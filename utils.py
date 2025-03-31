
# import os
# import json
# from PIL import Image
# import numpy as np

# DATA_FOLDER = "saved_entries"

# def get_traveler_data(passenger_id):
#     """
#     Retrieve traveler data if it exists.
#     """
#     file_path = os.path.join(DATA_FOLDER, f"{passenger_id}.json")
#     if os.path.exists(file_path):
#         with open(file_path, "r") as f:
#             return json.load(f)
#     return None

# def save_traveler_data(name, passenger_id, recognized_image, mrz_details):
#     """
#     Save traveler details and image.
#     """
#     os.makedirs(DATA_FOLDER, exist_ok=True)
    
#     # Save metadata as JSON
#     traveler_data = {
#         "name": name,
#         "passenger_id": passenger_id,
#         "mrz_details": mrz_details
#     }
    
#     with open(os.path.join(DATA_FOLDER, f"{passenger_id}.json"), "w") as f:
#         json.dump(traveler_data, f, indent=4)

#     # Save the image
#     image_path = os.path.join(DATA_FOLDER, f"{passenger_id}.png")
#     Image.fromarray(np.array(recognized_image)).save(image_path)



import os
import json
import psycopg2
from PIL import Image
import numpy as np
import io

DATA_FOLDER = "saved_entries"

# Database connection settings
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")  # Change if needed
DB_NAME = os.getenv("DB_NAME", "travelers_db")
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """Connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def save_traveler_data(name, passenger_id, recognized_image, mrz_details):
    """Save traveler details and image locally"""
    os.makedirs(DATA_FOLDER, exist_ok=True)

    traveler_data = {
        "name": name,
        "passenger_id": passenger_id,
        "mrz_details": mrz_details
    }

    # Save metadata as JSON
    with open(os.path.join(DATA_FOLDER, f"{passenger_id}.json"), "w") as f:
        json.dump(traveler_data, f, indent=4)

    # Save the image
    image_path = os.path.join(DATA_FOLDER, f"{passenger_id}.png")
    Image.fromarray(np.array(recognized_image)).save(image_path)
    
def get_traveler_data(passenger_id):
    """Retrieve traveler data from local storage."""
    file_path = os.path.join(DATA_FOLDER, f"{passenger_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None


def save_traveler_data_combined(name, passenger_id, recognized_image, mrz_details):
    """Save traveler details in PostgreSQL."""
    conn = psycopg2.connect(
        dbname="travelers_db",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    try:
        query = """
        INSERT INTO travelers (name, passenger_id, recognized_image, mrz_details)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (passenger_id) DO UPDATE 
        SET name = EXCLUDED.name, 
            recognized_image = EXCLUDED.recognized_image, 
            mrz_details = EXCLUDED.mrz_details;
        """

        # Convert image to byte format
        image_bytes = None
        if recognized_image:
            import io
            image_io = io.BytesIO()
            recognized_image.save(image_io, format="PNG")
            image_bytes = image_io.getvalue()

        cursor.execute(query, (name, passenger_id, image_bytes, json.dumps(mrz_details)))
        conn.commit()  

        print("✅ Traveler data saved successfully in DB!")

    except Exception as e:
        print(f"❌ Error saving traveler data: {e}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()