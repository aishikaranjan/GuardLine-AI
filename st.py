import streamlit as st
from PIL import Image
import re
from auth import register_user, authenticate_user  
import sys
import os
import main_app

st.set_page_config(page_title="GuardLine AI Dashboard", layout="wide")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "email" not in st.session_state:
    st.session_state["email"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "staff_id" not in st.session_state:
    st.session_state["staff_id"] = None

# Header and Sidebar
st.markdown("<h1 style='text-align: center;'>GuardLine AI Dashboard</h1>", unsafe_allow_html=True)
st.sidebar.image("logo.png", width=400)
st.sidebar.title("Menu")

if st.session_state["authenticated"]:
        # Reset problematic session state variables
    reset_keys = ["mrz_shown", "data_saved", "recognized_image", "mrz_details", "face_image"]
    for key in reset_keys:
        if key in st.session_state:
            del st.session_state[key]

    # session state keys
    st.session_state.setdefault("mrz_shown", False)
    st.session_state.setdefault("data_saved", False)
    st.session_state.setdefault("recognized_image", None)
    st.session_state.setdefault("mrz_details", None)
    st.session_state.setdefault("face_image", None)
    
    main_app.run_dashboard()
    st.stop()

menu = st.sidebar.radio("Navigation", ["Login", "Register"], index=0)

if menu == "Login":
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        auth_result, role, staff_id = authenticate_user(email, password)
        print(f"Debug: Login attempt - auth_result: {auth_result}, role: {role}, staff_id: {staff_id}")
        if auth_result:
            st.success("Login successful!")
            st.session_state["authenticated"] = True
            st.session_state["email"] = email
            st.session_state["role"] = role
            st.session_state["staff_id"] = staff_id
            print(f"Debug: Session state after login - role: {st.session_state['role']}")
            st.rerun()
        else:
            st.error("Invalid email or password.")

elif menu == "Register":
    st.subheader("Register")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    role = st.radio("Are you registering as:", ["Passenger", "Staff"])
    staff_id = None
    if role == "Staff":
        staff_id = st.text_input("Staff ID (6-digit)")
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match!")
        elif not re.match(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$', password):
            st.error("Password must be at least 8 characters long and contain both letters and numbers.")
        elif role == "Staff" and (not staff_id or not staff_id.isdigit() or len(staff_id) != 6):
            st.error("Invalid Staff ID. It must be a 6-digit number.")
        else:
            success, message = register_user(email, password, role, staff_id)
            if success:
                st.success("Registration successful! You can now log in.")
            else:
                st.error(message)