import streamlit as st
from PIL import Image
import io
import numpy as np
from db import save_traveler_data, save_traveler_entry, recognize_face, get_traveler_entries, get_all_traveler_entries, get_db_connection
from ocr_passport.streamlit_ocr import process_passport_image
import face_recognition
import json
import pandas as pd
import cv2
import imutils
import tempfile
import random
import os

def initialize_session_state():
    session_defaults = {
        "name": "",
        "passenger_id": "",
        "recognized_image": None,
        "mrz_details": None,
        "face_image": None,
        "face_encoding": None,
        "data_saved": False,
        "stored_mrz_details": None,
        "mrz_match": None,
        "face_match": None,
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    print("Debug: Session state initialized:", st.session_state)

def process_face_image(face_image):
    st.image(face_image, caption="Face Image", use_container_width=True)
    st.session_state["face_image"] = face_image
    
    # Compute face encoding
    face_image_array = np.array(face_image)
    encodings = face_recognition.face_encodings(face_image_array)
    if encodings:
        st.session_state["face_encoding"] = encodings[0]
        print("Debug: Face encoding computed successfully in Step 1. Encoding shape:", encodings[0].shape)
    else:
        st.session_state["face_encoding"] = None
        st.warning("No face detected in the image from Step 1.")
        print("Debug: No face detected in Step 1 image.")
    
    result = recognize_face(face_image)
    st.write(f"Recognized as: {result}")
    
    if "Unknown" in result or "new face" in result:
        st.warning("New Passenger Detected. Please enter details below.")
        st.session_state["name"] = ""
        st.session_state["passenger_id"] = ""
        st.session_state["stored_mrz_details"] = None
        print("Debug: New passenger detected. Resetting stored_mrz_details.")
    else:
        existing_data = get_traveler_data_by_name(result)
        print(f"Debug: Retrieved data for {result}: {existing_data}")
        if existing_data:
            st.session_state["name"] = existing_data["name"]
            st.session_state["passenger_id"] = existing_data["passenger_id"]
            st.session_state["stored_mrz_details"] = existing_data["mrz_details"]
            print(f"Debug: Stored MRZ details set: {st.session_state['stored_mrz_details']}")
        else:
            st.warning("Recognized but no data found. Please enter details.")
            print("Debug: Recognized passenger but no data found.")

def compare_faces(face_encoding_1, face_encoding_2):
    if face_encoding_1 is None or face_encoding_2 is None:
        print("Debug: One of the face encodings is None. face_encoding_1:", face_encoding_1, "face_encoding_2:", face_encoding_2)
        st.session_state["face_match"] = False
        print("Debug: face_match set to False due to None encoding.")
        return False
    matches = face_recognition.compare_faces([face_encoding_1], face_encoding_2, tolerance=0.65)  # Increased tolerance
    result = matches[0]
    st.session_state["face_match"] = result
    print(f"Debug: Face comparison result: {result}. face_match set to: {result}")
    return result

def compare_mrz_details(new_mrz_details):
    if not new_mrz_details:
        st.session_state["mrz_match"] = False
        st.warning("No MRZ details provided to compare.")
        print("Debug: No MRZ details provided to compare. mrz_match set to False.")
        return
    
    stored_mrz = st.session_state.get("stored_mrz_details")
    print(f"Debug: Comparing MRZ - Stored: {stored_mrz}, New: {new_mrz_details}")
    if stored_mrz and new_mrz_details:
        keys_to_compare = ["document_number", "surname", "given_name"]
        stored_subset = {key: stored_mrz.get(key) for key in keys_to_compare if key in stored_mrz}
        new_subset = {key: new_mrz_details.get(key) for key in keys_to_compare if key in new_mrz_details}
        
        if len(stored_subset) != len(keys_to_compare) or len(new_subset) != len(keys_to_compare):
            st.session_state["mrz_match"] = False
            st.warning("Incomplete MRZ data for comparison.")
            print("Debug: Incomplete MRZ data for comparison. mrz_match set to False.")
            return
        
        match = all(stored_subset[key] == new_subset[key] for key in keys_to_compare)
        st.session_state["mrz_match"] = match
        if match:
            st.success("MRZ details match the stored record! ✅")
            print("Debug: MRZ match successful. mrz_match set to True.")
        else:
            st.warning("MRZ details do not match the stored record.")
            print(f"Debug: MRZ mismatch. Stored MRZ: {stored_mrz}, New MRZ: {new_mrz_details}")
            print("Debug: mrz_match set to False.")
    else:
        st.session_state["mrz_match"] = False
        if not stored_mrz:
            st.info("No stored MRZ details to compare against. Proceed to save if this is a new passenger.")
            print("Debug: No stored MRZ details. mrz_match set to False (new passenger).")
        else:
            st.info("New MRZ details captured.")
            print("Debug: New MRZ details captured. mrz_match set to False.")

def detect_anomalies_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return [], [], []

    fgbg = cv2.createBackgroundSubtractorMOG2()
    abnormal_frames = []
    abnormal_frame_numbers = []
    processed_frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        _, binary_thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        binary_thresh = binary_thresh.astype(np.uint8)
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        abnormal_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red box
                cv2.putText(frame, "Strange Action", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                abnormal_detected = True

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame_rgb)

        if abnormal_detected and frame_count not in abnormal_frame_numbers:
            abnormal_frames.append(frame_rgb)
            abnormal_frame_numbers.append(frame_count)

    cap.release()
    cv2.destroyAllWindows()

    # Select up to 6 random abnormal frames for display
    selected_indices = sorted(random.sample(range(len(abnormal_frames)), min(6, len(abnormal_frames))))
    abnormal_frames = [abnormal_frames[i] for i in selected_indices]
    abnormal_frame_numbers = [abnormal_frame_numbers[i] for i in selected_indices]

    return abnormal_frames, abnormal_frame_numbers, processed_frames

def detect_anomalies_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error opening webcam.")
        return [], [], []

    fgbg = cv2.createBackgroundSubtractorMOG2()  # Motion detection
    frame_count = 0
    abnormal_frames = []
    abnormal_frame_numbers = []
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        _, binary_thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        abnormal_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum motion size threshold
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
                cv2.putText(frame, "Strange Action", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                abnormal_detected = True

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame_rgb)

        if abnormal_detected:
            abnormal_frames.append(frame_rgb)
            abnormal_frame_numbers.append(frame_count)

        if len(abnormal_frames) >= 6:  # Limit displayed abnormal frames
            break

    cap.release()
    cv2.destroyAllWindows()
    return abnormal_frames, abnormal_frame_numbers, processed_frames

def run_dashboard():
    initialize_session_state()
    
    st.sidebar.markdown(f"**Email:** {st.session_state.get('email', 'Unknown')}")
    if st.session_state.get('role') == "Staff":
        st.sidebar.markdown(f"**Staff ID:** {st.session_state.get('staff_id', 'Unknown')}")
    else:
        st.sidebar.markdown(f"**Role:** {st.session_state.get('role', 'Unknown')}")

    role = st.session_state.get('role', 'Passenger')
    if role == "Passenger":
        menu_options = ["Traveler Identity", "Logout"]
    elif role == "Staff":
        menu_options = ["Anomaly Surveillance", "Traveler History", "Logout"]
    
    menu = st.sidebar.radio("Navigation", menu_options)
    print(f"Debug: Role: {role}, Menu options: {menu_options}")

    if menu == "Traveler Identity" and role == "Passenger":
        st.subheader("Passenger Identity Verification")
        st.markdown("### Step 1: Passenger Recognition")
        input_method = st.radio("Choose Input Method", ["Upload Image", "Use Live Camera"])

        if input_method == "Upload Image":
            uploaded_face = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
            if uploaded_face:
                process_face_image(Image.open(uploaded_face))

        elif input_method == "Use Live Camera":
            camera_image = st.camera_input("Take a photo")
            if camera_image:
                process_face_image(Image.open(camera_image))

        st.markdown("### Step 2: Enter Passenger Details")
        st.session_state["name"] = st.text_input("Passenger Name", value=st.session_state["name"])
        st.session_state["passenger_id"] = st.text_input("Passenger ID", value=st.session_state["passenger_id"])

        if st.session_state["face_image"]:
            recognition_result = recognize_face(st.session_state["face_image"])
            if "Unknown" in recognition_result or "new face" in recognition_result:
                if st.button("Add New Face to Database") and st.session_state["name"] and st.session_state["passenger_id"]:
                    encodings = face_recognition.face_encodings(np.array(st.session_state["face_image"]))
                    if encodings:
                        save_traveler_data(
                            st.session_state["name"],
                            st.session_state["passenger_id"],
                            face_image=st.session_state["face_image"],
                            face_encoding=encodings[0]
                        )
                        st.success(f"Added {st.session_state['name']} to the database!")
                    else:
                        st.error("No face detected in the image.")

        st.markdown("### Step 3: MRZ Reader (Passport OCR)")
        ocr_input_method = st.radio("Select Method", ["Upload Image", "Real-Time Capture"])

        # Reset match flags before processing new passport image
        st.session_state["mrz_match"] = None
        st.session_state["face_match"] = None
        print("Debug: Reset mrz_match and face_match before passport processing.")

        if ocr_input_method == "Upload Image":
            uploaded_passport = st.file_uploader("Upload Passport Image", type=["jpg", "jpeg", "png"])
            if uploaded_passport:
                passport_image = Image.open(uploaded_passport)
                st.image(passport_image, caption="Uploaded Passport Image", use_container_width=True)
                with st.spinner("Processing passport image..."):
                    mrz_details, passport_image = process_passport_image(passport_image)
                    st.session_state["mrz_details"] = mrz_details
                    
                    # Extract face from passport image
                    passport_image_array = np.array(passport_image)
                    passport_encodings = face_recognition.face_encodings(passport_image_array)
                    print(f"Debug: Passport face encodings detected: {len(passport_encodings)} faces")
                    if passport_encodings:
                        passport_face_encoding = passport_encodings[0]
                        if st.session_state["face_encoding"] is not None:
                            face_match = compare_faces(st.session_state["face_encoding"], passport_face_encoding)
                            print(f"Debug: After face comparison (Upload), face_match: {st.session_state['face_match']}")
                            if face_match:
                                st.success("✅ Face matches the passport image!")
                            else:
                                st.error("❌ Face does not match the passport image. Please ensure the face image matches the passport.")
                        else:
                            st.warning("No face encoding available from Step 1 to compare. Please upload a clear face image in Step 1.")
                            st.session_state["face_match"] = False
                            print("Debug: No face encoding from Step 1 (Upload), face_match set to False.")
                    else:
                        st.warning("No face detected in the passport image. Please upload a passport image with a visible face.")
                        st.session_state["face_match"] = False
                        print("Debug: No face detected in passport image (Upload), face_match set to False.")
                    
                    if mrz_details:
                        st.subheader("Extracted MRZ Details")
                        st.json(mrz_details)
                        compare_mrz_details(mrz_details)
                        print(f"Debug: After MRZ comparison (Upload), mrz_match: {st.session_state['mrz_match']}")
                    else:
                        st.error("Failed to extract MRZ details from the passport image.")
                        st.session_state["mrz_match"] = False
                        print("Debug: MRZ extraction failed (Upload), mrz_match set to False.")

        elif ocr_input_method == "Real-Time Capture":
            passport_camera_image = st.camera_input("Capture Passport Image")
            if passport_camera_image:
                passport_image = Image.open(passport_camera_image)
                st.image(passport_image, caption="Captured Passport Image", use_container_width=True)
                with st.spinner("Processing passport image..."):
                    mrz_details, passport_image = process_passport_image(passport_image)
                    st.session_state["mrz_details"] = mrz_details
                    
                    # Extract face from passport image
                    passport_image_array = np.array(passport_image)
                    passport_encodings = face_recognition.face_encodings(passport_image_array)
                    print(f"Debug: Passport face encodings detected: {len(passport_encodings)} faces")
                    if passport_encodings:
                        passport_face_encoding = passport_encodings[0]
                        if st.session_state["face_encoding"] is not None:
                            face_match = compare_faces(st.session_state["face_encoding"], passport_face_encoding)
                            print(f"Debug: After face comparison (Real-Time), face_match: {st.session_state['face_match']}")
                            if face_match:
                                st.success("✅ Face matches the passport image!")
                            else:
                                st.error("❌ Face does not match the passport image. Please ensure the face image matches the passport.")
                        else:
                            st.warning("No face encoding available from Step 1 to compare. Please capture a clear face image in Step 1.")
                            st.session_state["face_match"] = False
                            print("Debug: No face encoding from Step 1 (Real-Time), face_match set to False.")
                    else:
                        st.warning("No face detected in the passport image. Please capture a passport image with a visible face.")
                        st.session_state["face_match"] = False
                        print("Debug: No face detected in passport image (Real-Time), face_match set to False.")
                    
                    if mrz_details:
                        st.subheader("Extracted MRZ Details")
                        st.json(mrz_details)
                        compare_mrz_details(mrz_details)
                        print(f"Debug: After MRZ comparison (Real-Time), mrz_match: {st.session_state['mrz_match']}")
                    else:
                        st.error("Failed to extract MRZ details from the passport image.")
                        st.session_state["mrz_match"] = False
                        print("Debug: MRZ extraction failed (Real-Time), mrz_match set to False.")

        # Final Verification and Save Data (without UI heading)
        print("Debug: Entering final verification logic.")
        print(f"Debug: Session state before verification: {st.session_state}")

        mrz_match = st.session_state.get("mrz_match")
        face_match = st.session_state.get("face_match")
        stored_mrz = st.session_state.get("stored_mrz_details")
        has_mrz_details = st.session_state.get("mrz_details") is not None

        print(f"Debug: Before button check - has_mrz_details: {has_mrz_details}, mrz_match: {mrz_match}, face_match: {face_match}, stored_mrz: {stored_mrz}")

        if not has_mrz_details:
            st.warning("No MRZ details available. Please upload a passport image.")
        elif face_match is False:
            st.warning("Face does not match the passport image. Please verify the passenger's identity.")
        elif stored_mrz is None and has_mrz_details:
            # New passenger: stored_mrz is None, but we have mrz_details
            st.success("New passenger verification complete!")
            if st.button("Save Data"):
                try:
                    face_image_data = st.session_state["face_image"]
                    recognized_image_data = st.session_state.get("recognized_image")  # Might be None
                    encodings = face_recognition.face_encodings(np.array(face_image_data)) if face_image_data else []
                    face_encoding = encodings[0] if encodings else None
                    
                    print("Debug: Calling save_traveler_data for new passenger")
                    success_data = save_traveler_data(
                        st.session_state["name"],
                        st.session_state["passenger_id"],
                        recognized_image=recognized_image_data,  # Use recognized_image if available
                        face_image=face_image_data,  # Fallback to face_image
                        mrz_details=st.session_state["mrz_details"],
                        face_encoding=face_encoding
                    )
                    print(f"Debug: save_traveler_data returned: {success_data}")
                    
                    if success_data:
                        print("Debug: Calling save_traveler_entry for new passenger")
                        success_entry = save_traveler_entry(
                            st.session_state["passenger_id"],
                            face_image=face_image_data,
                            recognized_image=recognized_image_data,
                            mrz_details=st.session_state["mrz_details"]
                        )
                        print(f"Debug: save_traveler_entry returned: {success_entry}")
                        
                        if success_entry:
                            st.session_state["data_saved"] = True
                            st.success("Data saved successfully! ✅")
                            st.success(f"Passenger {st.session_state['name']} is allowed to proceed! ✅")
                        else:
                            st.error("Failed to save traveler entry to the database.")
                    else:
                        st.error("Failed to save traveler data to the database.")
                except Exception as e:
                    st.error(f"Error saving data: {e}")
                    print(f"Debug: Exception in save_traveler_data/save_traveler_entry: {e}")

        elif stored_mrz is not None and mrz_match and face_match:
            # Recognized passenger: mrz_match and face_match must both be True
            st.success("All verifications passed for recognized passenger!")
            if st.button("Save Data"):
                try:
                    face_image_data = st.session_state["face_image"]
                    recognized_image_data = st.session_state.get("recognized_image")
                    
                    print("Debug: Calling save_traveler_entry for recognized passenger")
                    success = save_traveler_entry(
                        st.session_state["passenger_id"],
                        face_image=face_image_data,
                        recognized_image=recognized_image_data,
                        mrz_details=st.session_state["mrz_details"]
                    )
                    print(f"Debug: save_traveler_entry returned: {success}")
                    if success:
                        st.session_state["data_saved"] = True
                        st.success("Data saved successfully! ✅")
                        st.success(f"Passenger {st.session_state['name']} is allowed to proceed! ✅")
                    else:
                        st.error("Failed to save data to the database.")
                except Exception as Natasha:
                    st.error(f"Error saving data: {e}")
                    print(f"Debug: Exception in save_traveler_entry: {e}")
        else:
            if mrz_match is False:
                st.warning("Please upload the correct passport again.")
                st.session_state["mrz_details"] = None
            else:
                st.warning("Verification incomplete. Please ensure both MRZ and face match.")

    elif menu == "Anomaly Surveillance" and role == "Staff":
        st.markdown("""
            <h1 style='text-align: center; color: red;'>Video Anomaly Detection Model</h1>
            <br> <!-- Adds vertical space -->
        """, unsafe_allow_html=True)

        option = st.radio("Select Input Source:", ("Upload Video", "Use Webcam"), horizontal=True)

        if option == "Upload Video":
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                st.success("Video uploaded successfully. Processing...")

                abnormal_frames, abnormal_frame_numbers, processed_frames = detect_anomalies_video(tmp_file_path)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Abnormal Frames")
                    for count, abnormal_frame in enumerate(abnormal_frames):
                        st.image(abnormal_frame, caption=f"Frame {abnormal_frame_numbers[count]}", channels="RGB",
                                 use_container_width=True)

                with col2:
                    st.subheader("Live Video Feed")
                    for frame_rgb in processed_frames:
                        st.image(frame_rgb, caption="Live Video Feed", channels="RGB", use_container_width=True)

        elif option == "Use Webcam":
            st.success("Starting webcam anomaly detection...")

            abnormal_frames, abnormal_frame_numbers, processed_frames = detect_anomalies_webcam()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top Abnormal Frames")
                for count, abnormal_frame in enumerate(abnormal_frames):
                    st.image(abnormal_frame, caption=f"Frame {abnormal_frame_numbers[count]}", channels="RGB",
                             use_container_width=True)

            with col2:
                st.subheader("Live Webcam Feed")
                for frame_rgb in processed_frames:
                    st.image(frame_rgb, caption="Live Webcam Feed", channels="RGB", use_container_width=True)

    elif menu == "Traveler History" and role == "Staff":
        st.subheader("Traveler History")
        st.markdown("View and search the history of all traveler scans.")
        
        search_query = st.text_input("Search by Name or Passenger ID", "")
        entries = get_all_traveler_entries(search_query if search_query else None)
        
        if entries:
            table_data = [
                {k: v for k, v in entry.items() if k != "Face Image"}
                for entry in entries
            ]
            
            df = pd.DataFrame(table_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Traveler History as CSV",
                data=csv,
                file_name="traveler_history.csv",
                mime="text/csv",
            )
            
            for entry in entries:
                with st.expander(f"Traveler: {entry['Name']} (ID: {entry['Passenger ID']})"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if entry["Face Image"]:
                            try:
                                img_bytes = entry["Face Image"]
                                img = Image.open(io.BytesIO(img_bytes))
                                st.image(img, caption="Face Image", width=150)
                            except Exception as e:
                                st.warning(f"Could not display image: {e}")
                        else:
                            st.write("No image available.")
                    
                    with col2:
                        st.markdown(f"**Passenger ID:** {entry['Passenger ID']}")
                        st.markdown(f"**Scan Timestamp:** {entry['Scan Timestamp']}")
                        st.markdown(f"**Document Number:** {entry['Document Number']}")
                        st.markdown(f"**Surname:** {entry['Surname']}")
                        st.markdown(f"**Given Name:** {entry['Given Name']}")
        else:
            st.warning("No traveler history found matching your search.")

    elif menu == "Logout":
        st.session_state.clear()
        st.success("You have been logged out. Please log in again.")
        st.rerun()

def get_traveler_data_by_name(name):
    """Retrieve traveler data by name from the database."""
    conn = get_db_connection()
    if not conn:
        print("❌ Database connection failed.")
        return None
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name, passenger_id, mrz_details, face_encoding FROM travelers WHERE name = %s", (name,))
        result = cursor.fetchone()
        if result:
            name, passenger_id, mrz_details, face_encoding = result
            if isinstance(mrz_details, str):
                try:
                    mrz_details = json.loads(mrz_details)
                except json.JSONDecodeError as e:
                    print(f"❌ Error decoding MRZ details for {name}: {e}")
                    mrz_details = None
            return {
                "name": name,
                "passenger_id": passenger_id,
                "mrz_details": mrz_details,
                "face_encoding": face_encoding
            }
        return None
    except Exception as e:
        print(f"❌ Error fetching traveler data by name: {e}")
        return None
    finally:
        cursor.close()
        conn.close()