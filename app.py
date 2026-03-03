import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile

# Initialize GROQ client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Load YOLO medium model
# Replace this:
model = YOLO("yolov8n.pt")  

# With this:
model = YOLO("yolov8m.pt")  # medium model → better accuracy for all objects
st.title("🎯 Safe Object Detector")
option = st.sidebar.radio("Choose info type:", ("Object Use","Benefits & Drawbacks","Voice Explanation"))

camera_input = st.camera_input("📸 Show an object")

if camera_input:
    image = np.array(Image.open(camera_input))
    h, w = image.shape[:2]
    max_dim = 960
    scale = max_dim / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    results = model.predict(image, imgsz=640, verbose=False)[0]

    if len(results.boxes.xyxy) == 0:
        st.warning("No objects detected.")
    else:
        for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,255),3)
            cv2.putText(image, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            # Use GROQ LLM with YOLO label (no image) for explanation
            if option != "Voice Explanation":
                prompt_text = {
                    "Object Use": f"Explain what {label} is used for.",
                    "Benefits & Drawbacks": f"List benefits and drawbacks of {label}."
                }[option]

                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":prompt_text}],
                    temperature=0.7,
                    max_completion_tokens=200
                )
                st.write(f"**Detected Object:** {label}")
                st.write(resp.choices[0].message.content)
            else:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":f"Explain {label} simply for voice."}],
                    temperature=0.7,
                    max_completion_tokens=200
                )
                explanation_text = resp.choices[0].message.content
                st.write(f"**Detected Object:** {label}")
                st.write(explanation_text)
                tts = gTTS(explanation_text)
                tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp_mp3.name)
                st.audio(tmp_mp3.name, format="audio/mp3")

        st.image(image, channels="RGB", use_column_width=True)
else:
    st.warning("📷 Camera input not detected.")
