import psycopg2
import numpy as np
from psycopg2 import Error
import face_recognition
import json
import io
from PIL import Image

DB_CONFIG = {
    "dbname": "travelers_db",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def face_image_to_bytes(face_image):
    if face_image is None:
        return None
    img_byte_arr = io.BytesIO()
    face_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def save_traveler_data(name, passenger_id, face_image=None, recognized_image=None, mrz_details=None, face_encoding=None):
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed.")
        return False
    cursor = conn.cursor()
    try:
        recognized_image_bytes = face_image_to_bytes(recognized_image) if recognized_image else None
        mrz_details_json = json.dumps(mrz_details) if mrz_details else None
        face_encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None
        
        print(f"Debug: Saving traveler data - Name: {name}, Passenger ID: {passenger_id}")
        cursor.execute("""
            INSERT INTO travelers (name, passenger_id, recognized_image, mrz_details, face_encoding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (passenger_id) DO UPDATE
            SET name = EXCLUDED.name,
                recognized_image = EXCLUDED.recognized_image,
                mrz_details = EXCLUDED.mrz_details,
                face_encoding = EXCLUDED.face_encoding
        """, (name, passenger_id, recognized_image_bytes, mrz_details_json, face_encoding_bytes))
        conn.commit()
        print("Debug: Successfully saved traveler data.")
        return True
    except Exception as e:
        print(f"❌ Error saving traveler data: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def save_traveler_entry(passenger_id, face_image=None, recognized_image=None, mrz_details=None):
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed.")
        return False
    cursor = conn.cursor()
    try:
        face_image_bytes = face_image_to_bytes(face_image) if face_image else None
        recognized_image_bytes = face_image_to_bytes(recognized_image) if recognized_image else None
        mrz_details_json = json.dumps(mrz_details) if mrz_details else None
        
        print(f"Debug: Saving traveler entry - Passenger ID: {passenger_id}")
        cursor.execute("""
            INSERT INTO traveler_entries (passenger_id, face_image, recognized_image, mrz_details, entry_timestamp)
            VALUES (%s, %s, %s, %s, NOW())
        """, (passenger_id, face_image_bytes, recognized_image_bytes, mrz_details_json))
        conn.commit()
        print("Debug: Successfully saved traveler entry.")
        return True
    except Exception as e:
        print(f"❌ Error saving traveler entry: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def get_traveler_entries(passenger_id):
    """Fetch all entries for a passenger from traveler_entries."""
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed.")
        return []
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT entry_id, face_image, recognized_image, mrz_details, entry_timestamp
        FROM traveler_entries
        WHERE passenger_id = %s
        ORDER BY entry_timestamp DESC;
        """, (passenger_id,))
        results = cursor.fetchall()
        entries = []
        for result in results:
            entry_id, face_image, recognized_image, mrz_details, entry_timestamp = result
            entries.append({
                "entry_id": entry_id,
                "face_image": face_image,
                "recognized_image": recognized_image,
                "mrz_details": mrz_details,
                "entry_timestamp": entry_timestamp
            })
        return entries
    except Exception as e:
        print(f"❌ Error fetching traveler entries: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def get_all_travelers_encodings():
    """Retrieve all face encodings from the database."""
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed.")
        return []
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name, passenger_id, face_encoding FROM travelers WHERE face_encoding IS NOT NULL")
        results = cursor.fetchall()
        encodings = []
        for name, passenger_id, encoding_bytes in results:
            encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            encodings.append({"name": name, "passenger_id": passenger_id, "encoding": encoding})
        return encodings
    except Exception as e:
        print(f"❌ Error fetching encodings: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def recognize_face(image):
    """Recognize a face from an uploaded image."""
    encodings = get_all_travelers_encodings()
    if not encodings:
        return "Unknown - no face encodings in database"
    
    unknown_image = np.array(image)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    
    if not unknown_encodings:
        return "Unknown - no face detected in image"
    
    unknown_encoding = unknown_encodings[0]
    for known in encodings:
        result = face_recognition.compare_faces([known["encoding"]], unknown_encoding, tolerance=0.6)
        if result[0]:
            return known["name"]
    return "Unknown - new face"

def get_all_traveler_entries(search_query=None):
    """Fetch all traveler entries, optionally filtered by passenger_id or name."""
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed.")
        return []
    cursor = conn.cursor()
    try:
        if search_query:
            query = """
            SELECT t.name, te.passenger_id, te.entry_timestamp, te.mrz_details->>'document_number' AS document_number,
                   te.mrz_details->>'surname' AS surname, te.mrz_details->>'given_name' AS given_name, te.face_image
            FROM traveler_entries te
            JOIN travelers t ON t.passenger_id = te.passenger_id
            WHERE t.name ILIKE %s OR te.passenger_id ILIKE %s
            ORDER BY te.entry_timestamp DESC;
            """
            cursor.execute(query, (f"%{search_query}%", f"%{search_query}%"))
        else:
            query = """
            SELECT t.name, te.passenger_id, te.entry_timestamp, te.mrz_details->>'document_number' AS document_number,
                   te.mrz_details->>'surname' AS surname, te.mrz_details->>'given_name' AS given_name, te.face_image
            FROM traveler_entries te
            JOIN travelers t ON t.passenger_id = te.passenger_id
            ORDER BY te.entry_timestamp DESC;
            """
            cursor.execute(query)
        
        results = cursor.fetchall()
        entries = []
        for result in results:
            name, passenger_id, entry_timestamp, document_number, surname, given_name, face_image = result
            entries.append({
                "Name": name,
                "Passenger ID": passenger_id,
                "Scan Timestamp": entry_timestamp,
                "Document Number": document_number,
                "Surname": surname,
                "Given Name": given_name,
                "Face Image": face_image  # Raw bytes, will be converted in UI
            })
        return entries
    except Exception as e:
        print(f"❌ Error fetching all traveler entries: {e}")
        return []
    finally:
        cursor.close()
        conn.close()