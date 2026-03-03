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
# Initialize GROQ client
# --------------------------
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("❌ GROQ API Key not found! Add it in Streamlit secrets.")
    st.stop()

# --------------------------
# Load YOLOv8 medium model for better accuracy
# --------------------------
# Replace this:
model = YOLO("yolov8n.pt")  

# With this:
model = YOLO("yolov8m.pt")  # medium model → better accuracy for all objects
# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Smart Object Detector", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B0082'>🎯 Smart Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#4B0082'>Automatically detects objects, explains them, and plays voice!</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar with 3 options
st.sidebar.title("💡 Info Options")
option = st.sidebar.radio(
    "Choose info type:",
    ("Object Use", "Benefits & Drawbacks", "Voice Explanation")
)
st.sidebar.markdown("⚡ Show an object to the camera. It will be automatically recognized.")

# Camera input
camera_input = st.camera_input("📸 Show an object to your camera")

if camera_input is not None:
    # Convert PIL to OpenCV image
    image = np.array(Image.open(camera_input))

    # --------------------------
    # Increase resolution for YOLO inference
    # --------------------------
    max_dim = 960  # Resize longest side to 960px
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # YOLO detection
    results = model.predict(image, imgsz=640, verbose=False)[0]

    if len(results.boxes.xyxy) == 0:
        st.warning("No objects detected. Try again with better lighting or reposition the object.")
    else:
        # Draw bounding boxes and identify objects
        for idx, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cropped_obj = image[y1:y2, x1:x2]

            # --------------------------
            # Automatic object recognition using GROQ
            # --------------------------
            prompt = (
                "You are a smart assistant. "
                "Identify this object in one word or short phrase. "
                "The object may be small, thin, or uncommon like comb, marker, charger, glass bottle, etc."
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            object_name = response.choices[0].message.content.split("\n")[0]

            # Label on image
            cv2.putText(image, object_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --------------------------
            # Display info according to sidebar
            # --------------------------
            if option == "Object Use":
                msg_use = f"Explain the use of {object_name} in simple words."
                res_use = client.chat.completions.create(
                    messages=[{"role": "user", "content": msg_use}],
                    model="llama-3.3-70b-versatile"
                )
                st.info(res_use.choices[0].message.content)

            elif option == "Benefits & Drawbacks":
                msg_ben = f"What are the benefits and drawbacks of {object_name}?"
                res_ben = client.chat.completions.create(
                    messages=[{"role": "user", "content": msg_ben}],
                    model="llama-3.3-70b-versatile"
                )
                st.warning(res_ben.choices[0].message.content)

            elif option == "Voice Explanation":
                msg_voice = f"Explain {object_name} in simple words for speaking."
                res_voice = client.chat.completions.create(
                    messages=[{"role": "user", "content": msg_voice}],
                    model="llama-3.3-70b-versatile"
                )
                explanation_text = res_voice.choices[0].message.content
                st.success(explanation_text)

                tts = gTTS(explanation_text)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")

        # Display final image
        st.image(image, channels="RGB", caption="Detected Objects", use_column_width=True)

else:
    st.warning("📷 Camera input not detected. Please allow camera or upload image.")
