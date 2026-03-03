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
model = YOLO("yolov8s.pt")  # better accuracy than nano

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Object Detector", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B0082'>🎯 Smart Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#4B0082'>Automatically detects object, explains it, and plays voice!</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar always visible
st.sidebar.title("💡 Info Options")
option = st.sidebar.radio(
    "Choose info type:",
    ("Object Use", "Benefits & Drawbacks", "Voice Explanation")
)
st.sidebar.markdown("⚡ Show an object in front of your camera and select an option to learn more.")

# Camera input
camera_input = st.camera_input("📸 Show an object to the camera")

if camera_input is not None:
    # Convert PIL to OpenCV
    image = np.array(Image.open(camera_input))

    # YOLO detection (bounding boxes)
    results = model.predict(image, verbose=False)[0]

    # Process each detected object
    if len(results.boxes.xyxy) == 0:
        st.warning("No objects detected. Try again.")
    else:
        for idx, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            # Crop object region
            cropped = image[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cropped)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # ----------------------------
            # Use GROQ to identify object
            # ----------------------------
            prompt = (
                "You are a smart assistant. "
                "Identify this object in simple words (1-2 words). "
                "This object looks like: [describe visually as best as possible]."
            )

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )

            object_name = response.choices[0].message.content.split("\n")[0]

            # Label the box
            cv2.putText(image, object_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ----------------------------
            # Show explanations according to sidebar
            # ----------------------------
            if option == "Object Use":
                prompt_use = f"Explain the use of {object_name} in simple words."
                res_use = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_use}],
                    model="llama-3.3-70b-versatile"
                )
                st.info(res_use.choices[0].message.content)

            elif option == "Benefits & Drawbacks":
                prompt_ben = f"What are the benefits and drawbacks of {object_name}?"
                res_ben = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_ben}],
                    model="llama-3.3-70b-versatile"
                )
                st.warning(res_ben.choices[0].message.content)

            elif option == "Voice Explanation":
                prompt_voice = f"Explain {object_name} in simple words for speaking."
                res_voice = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_voice}],
                    model="llama-3.3-70b-versatile"
                )
                text_to_speak = res_voice.choices[0].message.content
                st.success(text_to_speak)
                # Convert to speech
                tts = gTTS(text_to_speak)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")

        # Show final image with bounding boxes and labels
        st.image(image, channels="RGB", caption="Detected Objects", use_column_width=True)

else:
    st.warning("📷 Camera input not detected. Please allow access or upload image.")
