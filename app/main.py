import streamlit as st
import torch
import cv2
import numpy as np
import os

from models.deeplabv3plus import get_model



MODEL_PATH = "runs/checkpoints/best_model.pth"
IMG_WIDTH = 960
IMG_HEIGHT = 544
THRESHOLD = 0.7

st.set_page_config(
    page_title="Falcon Segmentation Dashboard",
    layout="wide"
)

st.title("Falcon Segmentation Dashboard")


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found")
        return None

    model = get_model(1)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return model


model = load_model()

if model is None:
    st.stop()

st.success("Model loaded successfully")


def predict_image(original_img):

    original_h, original_w = original_img.shape[:2]


    img = cv2.resize(original_img, (IMG_WIDTH, IMG_HEIGHT))

    tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(output)

    prob_map = probability.cpu().numpy()[0][0]


    mask = (prob_map > THRESHOLD).astype(np.uint8)


    mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


    colored_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)

    colored_mask[mask == 1] = [255,0,0]

    # overlay
    overlay = original_img.copy()

    overlay[mask == 1] = [0,0,255]

    blended = cv2.addWeighted(
        original_img,
        0.8,
        overlay,
        0.4,
        0
    )

    return colored_mask, blended


uploaded = st.file_uploader(
    "Upload an image",
    type=["png","jpg","jpeg"]
)

if uploaded is not None:

    bytes_data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

    mask, overlay = predict_image(image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Segmentation Mask")
        st.image(mask, use_container_width=True)

    with col3:
        st.subheader("Overlay Result")
        st.image(overlay, use_container_width=True)

    st.success("Segmentation completed successfully")


st.markdown("---")
st.write("Falcon Semantic Segmentation | DeepLabV3+ | Streamlit")
