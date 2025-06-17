import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load model
model = load_model("ASLmodelF.h5")

# Class labels
classes = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','Space','Del','Nothing'
]

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit page config
st.set_page_config(page_title="Sign Language Translator", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #ffe6f0;
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #ff3399;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🖐️ Sign Language to Text</div>", unsafe_allow_html=True)

# Initialize session state
if 'predicted_word' not in st.session_state:
    st.session_state.predicted_word = ""
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()

start_camera = st.button("Start Camera")
clear_btn = st.button("Clear All")
FRAME_WINDOW = st.image([])
letter_display = st.empty()
text_box = st.empty()

if clear_btn:
    st.session_state.predicted_word = ""

if start_camera:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label = "Nothing"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords) * w), int(min(y_coords) * h)
            x2, y2 = int(max(x_coords) * w), int(max(y_coords) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:
                input_data = np.array(landmarks).reshape(1, 42)
                prediction = model.predict(input_data, verbose=0)
                index = np.argmax(prediction)
                label = classes[index]

        current_time = time.time()
        if label != "Nothing" and (current_time - st.session_state.last_time) > 2.5:
            if label == "Space":
                st.session_state.predicted_word += " "
            elif label == "Del":
                st.session_state.predicted_word = st.session_state.predicted_word[:-1]
            else:
                st.session_state.predicted_word += label
            st.session_state.last_time = current_time

        letter_display.markdown(f"<h2 style='text-align: center; color: blue;'>[ {label} ]</h2>", unsafe_allow_html=True)
        text_box.text_area("Translated Text", value=st.session_state.predicted_word, height=150)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
