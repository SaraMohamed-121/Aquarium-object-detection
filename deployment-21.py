import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Aquarium Project - YOLO Object Detection",
    page_icon="🌊",
    layout="wide"
)

page_bg_img = '''
<style>
.stApp {
    background-image: url("https://www.example.com/path_to_your_background_image.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("🐟 Aquarium Project: YOLO Object Detection")
st.markdown("""
This application detects aquatic creatures like dolphins, fish, starfish, puffins, etc., using the YOLO model.
These creatures play a significant role in tourism by attracting divers to beautiful underwater scenes, such as the Blue Hole in Dahab.
""")

model = YOLO(r'C:\Users\user\Downloads\vision applai\project\moodeel\best (1).pt')

base_dir = st.text_input('Enter the base directory for the label files:', r'C:\Users\user\Downloads\vision applai\project\aquarium.v2-release.darknet')
test_labels_path = os.path.join(base_dir, 'test', '_darknet.labels')

def read_labels(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

try:
    test_labels = read_labels(test_labels_path)
    st.subheader("Test Labels")
    st.write(f"Have {len(test_labels)} classes:\n")
    st.write(test_labels)
except Exception as e:
    st.error(f"Error reading labels: {e}")

img_files = st.file_uploader(label="Choose image files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
conf_threshold = st.slider('Confidence Threshold:', 0.0, 1.0, 0.25)

if img_files:
    for img_file in img_files:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        open_cv_image = cv2.imdecode(file_bytes, 1)
        
        results = model.predict(source=open_cv_image, conf=conf_threshold)

        for result in results:
            boxes = result.boxes.xyxy  
            scores = result.boxes.conf  
            classes = result.boxes.cls  

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)  
                label = f"{model.names[int(cls)]}: {score:.2f}"

                color = (0, 255, 0)  # Green by default, but you could vary this

                thickness = 2
                cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), color, thickness)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                label_background = (x1, y1 - label_size[1] - 10, x1 + label_size[0] + 10, y1)
                cv2.rectangle(open_cv_image, label_background[:2], label_background[2:], color, -1)
                cv2.putText(open_cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        st.write("Prediction Results:\n")
        st.image(open_cv_image, caption="Detected Objects", use_column_width=True)
