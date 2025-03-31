import streamlit as st
import cv2
import numpy as np
import imutils
import tempfile
import random
import os
from tensorflow.keras.models import load_model

        

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


def run_anomaly_dashboard():
    st.sidebar.title("Staff Dashboard")

    # Fetch user details from session state
    user_email = st.session_state.get('email', 'Unknown')
    staff_id = st.session_state.get('staff_id', 'N/A')

    # Display user information
    st.sidebar.markdown(f"**Email:** {user_email}")
    st.sidebar.markdown(f"**Staff ID:** {staff_id}")

    menu = st.sidebar.radio("Navigation", ["Video Anomaly Detection", "Logout"], index=0)

    if menu == "Video Anomaly Detection":
        # **Main Heading**
        st.markdown("""
            <h1 style='text-align: center; color: red;'>Video Anomaly Detection Model</h1>
            <br> <!-- Adds vertical space -->
        """, unsafe_allow_html=True)

        # **Move Input Source Selection Here (NOT in Sidebar)**
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

    elif menu == "Logout":
        st.session_state["authenticated"] = False
        st.success("You have been logged out. Redirecting to login...")

        # Ensure the redirection happens smoothly
        st.session_state["redirect"] = True
        st.rerun()

        # Redirect to login page (st.py)
        if st.session_state.get("redirect", False):
            st.session_state["redirect"] = False
            os.system("streamlit run C:/Users/aishi/Downloads/Capstone 2/Capstone 2/st.py")


if __name__ == "__main__":
    run_anomaly_dashboard()