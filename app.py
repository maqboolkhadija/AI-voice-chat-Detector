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
# Initialize GROQ securely
# --------------------------
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("❌ GROQ API Key not found! Add it in Streamlit secrets.")
    st.stop()

# --------------------------
# Load YOLOv8 model
# --------------------------
model = YOLO("yolov8s.pt")  # more accurate than yolov8n

# --------------------------
# Streamlit Page Layout
# --------------------------
st.set_page_config(page_title="🎯 Smart Object Detector", page_icon="🕶️", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎯 Smart Object Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B0082;'>Show any object and get info + voice output!</h3>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------
# Sidebar always visible
# --------------------------
st.sidebar.title("💡 Object Info Options")
option = st.sidebar.radio(
    "Choose what info you want:",
    ("Object Use", "Benefits & Drawbacks", "Voice Explanation")
)
st.sidebar.markdown("⚡ Instructions: Show an object to the camera. Choose an option to learn more.")

# --------------------------
# Camera input
# --------------------------
camera_input = st.camera_input("📸 Show an object to your camera")

if camera_input is not None:
    # Convert PIL to OpenCV
    image = np.array(Image.open(camera_input))

    # YOLO detection (just bounding boxes)
    results = model.predict(image, verbose=False)[0]

    # Draw colorful bounding boxes
    for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)  # magenta boxes
        label = f"Object"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - 25), (x1 + w, y1), (255, 0, 255), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    st.image(image, channels="RGB", caption="Detected Object", use_column_width=True)

    # --------------------------
    # Use GROQ to recognize & describe object
    # --------------------------
    # Take the first bounding box region as a reference
    if len(results.boxes.xyxy) > 0:
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[0])
        cropped_obj = image[y1:y2, x1:x2]
        pil_obj = Image.fromarray(cropped_obj)

        # Convert to bytes for GROQ (optional)
        # But simplest: we just ask GROQ to identify object in general
        prompt = "Identify this object in one word and describe it simply."

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        object_name = response.choices[0].message.content.split('\n')[0]  # Take first line as name
        st.markdown(f"### 🔹 Detected Object (GROQ): {object_name}")

        # --------------------------
        # Handle 3 options
        # --------------------------
        if option == "Object Use":
            prompt_use = f"Explain the use of {object_name} in simple words."
            res_use = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_use}],
                model="llama-3.3-70b-versatile",
            )
            st.info(res_use.choices[0].message.content)

        elif option == "Benefits & Drawbacks":
            prompt_ben = f"What are the benefits and drawbacks of {object_name}?"
            res_ben = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_ben}],
                model="llama-3.3-70b-versatile",
            )
            st.warning(res_ben.choices[0].message.content)

        elif option == "Voice Explanation":
            prompt_voice = f"Explain {object_name} in a simple sentence for speaking."
            res_voice = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_voice}],
                model="llama-3.3-70b-versatile",
            )
            text_to_speak = res_voice.choices[0].message.content
            st.success(text_to_speak)
            # Convert to audio
            tts = gTTS(text_to_speak)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")

else:
    st.warning("📷 Camera input not detected. Please allow access or upload an image.")
