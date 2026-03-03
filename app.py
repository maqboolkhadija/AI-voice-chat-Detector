import os
import cv2
import base64
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile

# --------------------------
# Initialize GROQ Vision client
# --------------------------
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("❌ GROQ API Key not found! Add it in Streamlit secrets.")
    st.stop()

# Vision model name that supports image input
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --------------------------
# Load YOLO for bounding boxes only
# --------------------------
model = YOLO("yolov8m.pt")  # medium for better box detection

# --------------------------
# UI Setup
# --------------------------
st.set_page_config(page_title="Smart Vision Object Detector", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B0082'>🎯 Smart Vision Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#4B0082'>Detects and names real objects accurately!</h4>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("Info Options")
option = st.sidebar.radio("Choose info type:", ("Object Use","Benefits & Drawbacks","Voice Explanation"))

# Camera
camera_input = st.camera_input("📸 Show an object for detection")

if camera_input:
    # Read and convert image
    raw = Image.open(camera_input)
    image_np = np.array(raw)

    # Resize for YOLO
    max_dim = 960
    h, w = image_np.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        image_np = cv2.resize(image_np, (int(w*scale), int(h*scale)), cv2.INTER_CUBIC)

    # Run YOLO
    results = model.predict(image_np, imgsz=640, verbose=False)[0]

    if len(results.boxes.xyxy) == 0:
        st.warning("No objects detected. Try better lighting.")

    # Draw boxes and detect names using Groq Vision
    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_np, (x1,y1), (x2,y2), (255,0,255), 3)

        # Crop object from image
        crop = image_np[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop)
        
        # Encode cropped object to base64
        buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        pil_crop.save(buffered.name)
        with open(buffered.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        img_data = f"data:image/jpeg;base64,{b64}"

        # Ask Groq Vision what this object is
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What object is this?"},
                        {"type": "image_url", "image_url": {"url": img_data}}
                    ],
                }
            ],
            temperature=0.7,
            max_completion_tokens=150
        )
        # Extract object name from the answer
        obj_name = response.choices[0].message.content.strip().split("\n")[0]
        cv2.putText(image_np, obj_name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Based on selected info option:
        if option == "Object Use":
            prompt = f"Explain in simple words what {obj_name} is used for."
        elif option == "Benefits & Drawbacks":
            prompt = f"List benefits and drawbacks of {obj_name}."
        else:  # Voice
            prompt = f"Explain {obj_name} simply so it can be spoken out loud."

        info_resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=200
        )
        info_text = info_resp.choices[0].message.content

        # UI Output
        st.write(f"**Detected Object:** {obj_name}")
        st.write(info_text)

        # If voice option selected
        if option == "Voice Explanation":
            tts = gTTS(info_text)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            st.audio(tmp.name, format="audio/mp3")

    # Show final image
    st.image(image_np, channels="RGB", use_column_width=True)
else:
    st.warning("📷 Camera input not detected. Allow camera or upload an image.")
