from groq import Groq
import streamlit as st
import tempfile
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from gtts import gTTS

# Initialize GROQ client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Load YOLO model
# Replace this:
model = YOLO("yolov8n.pt")  

# With this:
model = YOLO("yolov8m.pt")  # medium model → better accuracy for all objects
st.title("🎯 Smart Object Detector")

option = st.sidebar.radio("Choose info type:", ("Object Use", "Benefits & Drawbacks", "Voice Explanation"))

camera_input = st.camera_input("📸 Show an object")

if camera_input:
    image = np.array(Image.open(camera_input))

    # Resize for clarity
    h, w = image.shape[:2]
    max_dim = 960
    scale = max_dim / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    results = model.predict(image, imgsz=640, verbose=False)[0]

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,255), 3)

        # Crop object and save temporarily
        crop = image[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        pil_crop.save(tmp_file.name)

        # Send cropped file to GROQ
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role":"user",
                "content":[
                    {"type":"text","text":"Identify this object accurately in 1-2 words."},
                    {"type":"image_file","image_file":{"file_path": tmp_file.name}}
                ]
            }],
            temperature=0.7,
            max_completion_tokens=150
        )
        obj_name = response.choices[0].message.content.strip().split("\n")[0]
        cv2.putText(image, obj_name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        # Generate explanation
        if option != "Voice Explanation":
            prompt_text = {
                "Object Use": f"Explain what {obj_name} is used for.",
                "Benefits & Drawbacks": f"List benefits and drawbacks of {obj_name}."
            }[option]

            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role":"user","content":prompt_text}],
                temperature=0.7,
                max_completion_tokens=200
            )
            st.write(f"**Detected Object:** {obj_name}")
            st.write(resp.choices[0].message.content)

        else:  # Voice
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role":"user","content":f"Explain {obj_name} simply for voice."}],
                temperature=0.7,
                max_completion_tokens=200
            )
            explanation_text = resp.choices[0].message.content
            st.write(f"**Detected Object:** {obj_name}")
            st.write(explanation_text)
            tts = gTTS(explanation_text)
            tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_mp3.name)
            st.audio(tmp_mp3.name, format="audio/mp3")

    st.image(image, channels="RGB", use_column_width=True)
else:
    st.warning("📷 Camera input not detected. Allow camera or upload image.")
