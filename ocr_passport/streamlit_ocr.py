import streamlit as st
from PIL import Image
import numpy as np
from fastmrz.fastmrz import FastMRZ
import tempfile
import os
import pytesseract

# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/aishi/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

# Initialize MRZ Reader
mrz_reader = FastMRZ()

# ocr_passport/streamlit_ocr.py
import streamlit as st
from PIL import Image
import numpy as np
from fastmrz.fastmrz import FastMRZ
import tempfile
import os
import pytesseract

# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/aishi/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

# Initialize MRZ Reader
mrz_reader = FastMRZ()

# Function to process the captured or uploaded image
def process_passport_image(image):
    """
    Process a passport image to extract MRZ details.
    Args:
        image: PIL Image object of the passport.
    Returns:
        tuple: (mrz_details: dict, image: PIL Image) - MRZ details and the original image.
    """
    # Use a custom writable directory for temporary files
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it doesn't exist

    temp_file_path = os.path.join(temp_dir, "temp_passport_image.jpg")
    image.save(temp_file_path)  # Save the PIL image to the temp file
    
    # Process the image using FastMRZ
    try:
        mrz_details = mrz_reader.get_details(temp_file_path)
        # Ensure mrz_details is a dictionary
        if not mrz_details or not isinstance(mrz_details, dict):
            mrz_details = None
    except Exception as e:
        print(f"Error processing passport image with FastMRZ: {e}")
        mrz_details = None
    
    # Clean up the temporary file
    try:
        os.remove(temp_file_path)
    except Exception as e:
        print(f"Error removing temp file: {e}")
    
    return mrz_details, image
# # Streamlit App Layout
# st.title("MRZ Passport Reader with Real-Time Capture")
# st.write("Upload a passport image or capture one in real time to extract MRZ details.")

# # Add tabs for different input methods
# tab1, tab2 = st.tabs(["Upload Image", "Real-Time Capture"])

# with tab1:
#     # File uploader for passport image
#     uploaded_file = st.file_uploader("Upload Passport Image", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#         # Display uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Passport Image", use_column_width=True)

#         # Process the passport image
#         with st.spinner("Processing the image..."):
#             try:
#                 mrz_details = process_passport_image(image)
#                 # Display MRZ details
#                 st.subheader("MRZ Details")
#                 st.json(mrz_details)
#             except Exception as e:
#                 st.error(f"Error processing the image: {e}")

#         st.success("Processing complete!")

# with tab2:
#     # Real-time camera capture
#     captured_image = st.camera_input("Capture a photo of the passport")

#     if captured_image is not None:
#         # Display captured image
#         image = Image.open(captured_image)
#         st.image(image, caption="Captured Passport Image", use_column_width=True)

#         # Process the captured passport image
#         with st.spinner("Processing the image..."):
#             try:
#                 mrz_details = process_passport_image(image)
#                 # Display MRZ details
#                 st.subheader("MRZ Details")
#                 st.json(mrz_details)
#             except Exception as e:
#                 st.error(f"Error processing the image: {e}")

#         st.success("Processing complete!")
