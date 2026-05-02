"""
ui_streamlit.py
---------------
Interactive user interface for the CNN-LSTM Image Captioning project.
"""

import os
import json
import time
import torch
import streamlit as st
from PIL import Image

from model import CNNtoLSTM
from eval import generate_caption
from vocab_builder import load_vocabulary
from dataset import get_transforms

st.set_page_config(
    page_title="Image Captioning - CNN-LSTM",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Image Captioning with CNN-LSTM (PyTorch)")
st.markdown(
    """
    This demo uses a Convolutional Neural Network (CNN) as an encoder
    and a Long Short-Term Memory (LSTM) as a decoder to generate
    natural language captions from images.

    ---
    """
)


@st.cache_resource
def load_model(model_path: str, vocab_path: str, device: str = "cuda"):
    """Load the trained model and vocabulary."""
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    vocab = load_vocabulary(vocab_path)

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = CNNtoLSTM(
        embed_size=model_cfg.get("embed_dim", 300),
        hidden_size=model_cfg.get("hidden_dim", 512),
        vocab_size=len(vocab),
        num_layers=model_cfg.get("num_layers", 1),
        dropout=model_cfg.get("dropout", 0.3),
        train_cnn=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, vocab, device


st.sidebar.header("Configuration")

model_path = st.sidebar.text_input(
    "Model weights (.pt):",
    value="checkpoints/model_mini_demo.pt"
)
vocab_path = st.sidebar.text_input(
    "Vocabulary file (.json):",
    value="data/vocab.json"
)
captions_json = st.sidebar.text_input(
    "Reference captions (optional, .json):",
    value="data/Flickr8k_text/test_captions.json"
)
top_k = st.sidebar.slider("Top-k sampling", min_value=1, max_value=10, value=1, step=1)
max_len = st.sidebar.slider("Maximum caption length", min_value=10, max_value=30, value=20, step=2)
device_choice = st.sidebar.radio("Device", options=["cuda", "cpu"], index=0)

try:
    model, vocab, device = load_model(model_path, vocab_path, device_choice)
    st.sidebar.success("Model successfully loaded.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

transform = get_transforms(img_size=299)

st.subheader("Upload an image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            start_time = time.time()
            image_tensor = transform(image).unsqueeze(0).to(device)
            pred_caption = generate_caption(
                model=model,
                image_tensor=image_tensor,
                vocab=vocab,
                max_len=max_len,
                top_k=top_k,
            )
            elapsed = time.time() - start_time

        st.success("Caption generated successfully.")
        st.markdown(f"**Generated Caption:** {pred_caption}")
        st.caption(f"Inference time: {elapsed:.2f} seconds")

        if os.path.exists(captions_json):
            with open(captions_json, "r", encoding="utf-8") as f:
                refs = json.load(f)
            filename = os.path.basename(uploaded_file.name)
            if filename in refs:
                st.markdown("**Reference Caption (one of five):**")
                st.info(refs[filename][0])
            else:
                st.info("No reference caption available for this image.")

st.markdown("---")
st.caption(
    """
    Developed by Giuseppe Dimonte
    MSc in Computer Engineering - Artificial Intelligence Curriculum
    University of Parma
    """
)
