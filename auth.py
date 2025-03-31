import psycopg2
from psycopg2 import Error

# Database configuration (matching your setup)
DB_CONFIG = {
    "dbname": "travelers_db",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def register_user(email, password, role, staff_id=None):
    """Register a new user in the users table."""
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed."
    
    cursor = conn.cursor()
    try:
        # Check if email already exists
        cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return False, "Email already registered."

        # Check if staff_id is unique (for Staff role)
        if role == "Staff" and staff_id:
            cursor.execute("SELECT staff_id FROM users WHERE staff_id = %s", (staff_id,))
            if cursor.fetchone():
                return False, "A staff account with this ID already exists."

        # Insert new user
        cursor.execute(
            "INSERT INTO users (email, password, role, staff_id) VALUES (%s, %s, %s, %s) RETURNING id",
            (email, password, role, staff_id)
        )
        conn.commit()
        return True, "Registration successful!"
    
    except Error as e:
        print(f"Error registering user: {e}")
        return False, f"Database error: {e}"
    
    finally:
        cursor.close()
        conn.close()

def authenticate_user(email, password):
    """Authenticate a user from the users table."""
    conn = get_db_connection()
    if not conn:
        return False, None, None
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password, role, staff_id FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        
        if not result:
            return False, None, None
        
        stored_password, role, staff_id = result
        if stored_password != password:
            return False, None, None
        
        return True, role, staff_id
    
    except Error as e:
        print(f"Error authenticating user: {e}")
        return False, None, None
    
    finally:
        cursor.close()
        conn.close()