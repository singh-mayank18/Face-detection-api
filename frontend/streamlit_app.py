import streamlit as st
import requests
from PIL import Image
import io

st.title("Face Detection Demo")


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

API_URL = "http://localhost:8000/detect-face"

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Detect Faces"):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Faces Detected: {result['faces_detected']}")
            st.json(result)
        else:
            st.error("API error! Check backend logs.")
