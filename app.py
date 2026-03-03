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
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("❌ GROQ API Key not found! Add it in Streamlit secrets.")
    st.stop()

# --------------------------
# Load YOLOv8 model
# --------------------------
# Currently:
model = YOLO("yolov8n.pt")

# Replace with small model for better accuracy:
model = YOLO("yolov8s.pt")  # YOLOv8 small model, more accurate
# --------------------------
# Streamlit page layout
# --------------------------
st.set_page_config(page_title="🎯 Real-Time Object Detector", page_icon="🕶️", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎯 YOLOv8 Object Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B0082;'>Show any object to your camera and explore it!</h3>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------
# Sidebar with 3 always-visible options
# --------------------------
st.sidebar.title("💡 Object Info Options")
option = st.sidebar.radio(
    "Choose what info you want:",
    ("Object Use", "Benefits & Drawbacks", "Voice Explanation")
)
st.sidebar.markdown("⚡ Instructions: Show an object to your camera. Choose an option from above to learn more.")

# --------------------------
# Camera input
# --------------------------
camera_input = st.camera_input("📸 Show an object to your camera")

if camera_input is not None:
    # Convert PIL to OpenCV image
    image = np.array(Image.open(camera_input))
    results = model.predict(image, verbose=False)[0]

    # Draw bounding boxes and labels
    for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls_id)]} ({score:.2f})"
        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (75, 0, 130), 3)
        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - 25), (x1 + w, y1), (75, 0, 130), -1)
        # Label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display processed image
    st.image(image, channels="RGB", caption="Detected Objects", use_column_width=True)

    # Get the first detected object
    detected_object = model.names[int(results.boxes.cls[0])] if len(results.boxes.cls) > 0 else None

    # If an object is detected
    if detected_object:
        st.markdown(f"### 🔹 Detected Object: {detected_object}")

        # Option 1: Object Use
        if option == "Object Use":
            message = f"Explain the use of {detected_object} in simple words."
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama-3.3-70b-versatile",
            )
            st.markdown("**Use Explanation:**")
            st.info(response.choices[0].message.content)

        # Option 2: Benefits & Drawbacks
        elif option == "Benefits & Drawbacks":
            message = f"What are the benefits and drawbacks of {detected_object}?"
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama-3.3-70b-versatile",
            )
            st.markdown("**Benefits & Drawbacks:**")
            st.warning(response.choices[0].message.content)

        # Option 3: Voice Explanation
        elif option == "Voice Explanation":
            message = f"Explain {detected_object} in a simple sentence for speaking."
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model="llama-3.3-70b-versatile",
            )
            explanation_text = response.choices[0].message.content
            st.markdown("**Voice Explanation:**")
            st.success(explanation_text)

            # Convert text to speech
            tts = gTTS(explanation_text)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
else:
    st.warning("📷 Camera input not detected. Please allow access or upload an image.")
