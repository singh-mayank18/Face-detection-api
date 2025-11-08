import streamlit as st
import uvicorn
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import hashlib
import requests
# FASTAPI BACKEND

app = FastAPI(title="Face Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

  #Store uploaded image hashes to detect duplicates
uploaded_hashes = set()

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_img_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    image_bytes = await file.read()

    #Check duplicates
    img_hash = get_img_hash(image_bytes)
    if img_hash in uploaded_hashes:
        return {
            "message": "Image already uploaded",
            "faces_detected": None,
            "face_boxes": None
        }

    uploaded_hashes.add(img_hash)

    #Face detection
    image = Image.open(BytesIO(image_bytes))
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return {
        "message": "New image processed",
        "faces_detected": len(faces),
        "face_boxes": faces.tolist() if len(faces) > 0 else []
    }

        #Function to run FastAPI in background thread
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()


# STREAMLIT FRONTEND

st.title("Face Detection App (API + UI Combined)")

API_URL = "http://localhost:8000/detect-face"   # Works in Streamlit Cloud

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Detect Faces"):
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Faces Detected: {result['faces_detected']}")
            st.json(result)
        else:
            st.error("API Error. Check logs.")
