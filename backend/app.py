from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import hashlib

app = FastAPI(title="Face Detection API")

#In-memory store of hashes
uploaded_hashes = set()

       #Load Haar Cascade Model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_img_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    #Read image bytes
    image_bytes = await file.read()

    #Generate hash for duplicate detection
    img_hash = get_img_hash(image_bytes)

    #Check if already uploaded
    if img_hash in uploaded_hashes:
        return {
            "message": "Image already uploaded",
            "faces_detected": None,
            "face_boxes": None
        }

    #Add hash to memory (mark as seen)
    uploaded_hashes.add(img_hash)

      #Read image for detection
    image = Image.open(BytesIO(image_bytes))
    img_array = np.array(image)

   #Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

      #Run face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        #Prepare output
    result = {
        "message": "New image processed",
        "faces_detected": len(faces),
        "face_boxes": faces.tolist() if len(faces) > 0 else []
    }

    return result
