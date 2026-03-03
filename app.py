# app.py
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile

# --------------------------
# Initialize GROQ client securely
# --------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------
# Load YOLOv8 model
# --------------------------
st.title("🕶️ Real-Time Object Detection & Info")
st.text("Show any object to your camera. YOLOv8 will detect it and give details!")

model = YOLO("yolov8n.pt")  # lightweight YOLOv8 model

# --------------------------
# Camera input
# --------------------------
camera_input = st.camera_input("Show object to your camera")

if camera_input:
    # Convert PIL image to OpenCV format
    image = np.array(Image.open(camera_input))
    results = model.predict(image, verbose=False)[0]

    # Draw bounding boxes and labels
    for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls_id)]} ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Display the image
    st.image(image, channels="RGB")

    # --------------------------
    # Sidebar options
    # --------------------------
    st.sidebar.title("Object Info Options")
    detected_object = model.names[int(results.boxes.cls[0])] if len(results.boxes.cls) > 0 else None

    if detected_object:
        option = st.sidebar.radio(
            "Choose an option:",
            ("Object Use", "Benefits & Drawbacks", "Voice Explanation")
        )

        if option == "Object Use":
            message = f"Explain the use of {detected_object} in simple words."
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama-3.3-70b-versatile",
            )
            st.subheader(f"Use of {detected_object}")
            st.write(response.choices[0].message.content)

        elif option == "Benefits & Drawbacks":
            message = f"What are the benefits and drawbacks of {detected_object}?"
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama-3.3-70b-versatile",
            )
            st.subheader(f"Benefits & Drawbacks of {detected_object}")
            st.write(response.choices[0].message.content)

        elif option == "Voice Explanation":
            message = f"Explain {detected_object} in a simple sentence for speaking."
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama-3.3-70b-versatile",
            )
            explanation_text = response.choices[0].message.content
            st.subheader(f"Voice Explanation of {detected_object}")
            st.write(explanation_text)

            # Convert text to speech
            tts = gTTS(explanation_text)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
