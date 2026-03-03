import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile

# ----------------------------
# Initialize GROQ client
# ----------------------------
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("GROQ API Key not found! Add it in Streamlit secrets.")
    st.stop()

# ----------------------------
# Load YOLOv8 small model
# ----------------------------
model = YOLO("yolov8s.pt")  # faster and more accurate than nano

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Object Detector", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B0082'>🎯 Smart Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#4B0082'>Show an object and get its info + voice</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("💡 Info Options")
option = st.sidebar.radio(
    "Choose info type:",
    ("Object Use", "Benefits & Drawbacks", "Voice Explanation")
)
st.sidebar.markdown("Instructions: Show an object to the camera. Optionally, type object name for accuracy.")

# Camera input
camera_input = st.camera_input("📸 Show an object to the camera")

if camera_input:
    # Convert image
    image = np.array(Image.open(camera_input))

    # YOLO detects bounding boxes
    results = model.predict(image, verbose=False)[0]

    # Draw boxes
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    st.image(image, channels="RGB", caption="Detected Objects", use_column_width=True)

    # Ask user to type object name (required for correct explanation)
    detected_object = st.text_input("Detected object (type it here for accurate info):")

    if detected_object:
        st.markdown(f"### 🔹 Object: {detected_object}")

        # Option 1: Use
        if option == "Object Use":
            response = client.chat.completions.create(
                messages=[{"role":"user","content":f"Explain the use of {detected_object} in simple words."}],
                model="llama-3.3-70b-versatile"
            )
            st.info(response.choices[0].message.content)

        # Option 2: Benefits & Drawbacks
        elif option == "Benefits & Drawbacks":
            response = client.chat.completions.create(
                messages=[{"role":"user","content":f"What are the benefits and drawbacks of {detected_object}?"}],
                model="llama-3.3-70b-versatile"
            )
            st.warning(response.choices[0].message.content)

        # Option 3: Voice
        elif option == "Voice Explanation":
            response = client.chat.completions.create(
                messages=[{"role":"user","content":f"Explain {detected_object} in simple words for speaking."}],
                model="llama-3.3-70b-versatile"
            )
            text_to_speak = response.choices[0].message.content
            st.success(text_to_speak)

            # Convert to speech
            tts = gTTS(text_to_speak)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")

else:
    st.warning("📷 Camera input not detected. Please allow camera or upload image.")
