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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Image Captioning Demo",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 92% 12%, rgba(31, 105, 211, 0.08), transparent 23%),
            radial-gradient(circle at 8% 82%, rgba(47, 142, 224, 0.06), transparent 25%),
            linear-gradient(145deg, #f9fbff 0%, #f1f6fc 52%, #edf4fb 100%);
    }

    .block-container {
        max-width: 1160px;
        padding-top: 4rem;
        padding-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0c2d68;
        line-height: 1.15;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 0.82rem;
        color: #586985;
        margin-bottom: 0.85rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .status-ok {
        color: #17733a;
        background: #edf9f1;
        border: 1px solid #bee6c9;
    }

    .status-error {
        color: #a12622;
        background: #fff0ef;
        border: 1px solid #efc1bd;
    }

    .panel {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbe6f3;
        border-radius: 20px;
        padding: 1.15rem;
        box-shadow: 0 8px 26px rgba(27, 57, 104, 0.08);
        min-height: 100%;
    }

    .panel-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: #0d2f6b;
        margin-bottom: 0.7rem;
    }

    .caption-box {
        background: linear-gradient(180deg, #f4fbf3 0%, #edf8ec 100%);
        border: 1px solid #acd8a8;
        border-radius: 10px;
        padding: 1.1rem;
        min-height: 166px;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .caption-text {
        color: #1e2a3a;
        font-size: 1.14rem;
        font-weight: 700;
        line-height: 1.42;
    }

    .caption-placeholder {
        color: #7a879a;
        font-size: 1.05rem;
        font-weight: 500;
    }

    .info-card {
        background: transparent;
        border: 0;
        border-radius: 0;
        padding: 0.35rem 0.75rem;
        box-shadow: none;
        text-align: center;
    }

    .info-label {
        color: #7a879a;
        font-size: 0.88rem;
        margin-bottom: 0.28rem;
    }

    .info-value {
        color: #0c2d68;
        font-size: 1rem;
        font-weight: 800;
    }

    /* Give the three primary work areas the compact slide-like card treatment. */
    div[data-testid="stHorizontalBlock"]:has(.stFileUploader) > div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid #d5e2f1;
        border-radius: 12px;
        padding: 0.72rem 0.75rem 0.62rem;
        box-shadow: 0 5px 18px rgba(27, 57, 104, 0.07);
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #8bb7ee;
        border-radius: 9px;
        background: #f8fbff;
        padding: 0.35rem;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: 0;
        min-height: 130px;
    }

    div[data-testid="stImage"] img {
        border-radius: 9px;
        height: 265px;
        object-fit: cover;
    }

    div[data-testid="stFileUploader"] section {
        padding: 0.35rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.info-card) {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #d5e2f1;
        border-radius: 10px;
        padding: 0.55rem 0.3rem;
        box-shadow: 0 4px 14px rgba(27, 57, 104, 0.05);
    }

    div[data-testid="stHorizontalBlock"]:has(.info-card) > div[data-testid="column"] + div[data-testid="column"] {
        border-left: 1px solid #dbe6f3;
    }

    .footer-box {
        margin-top: 1.5rem;
        text-align: center;
        color: #738097;
        font-size: 0.9rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 2.45rem;
        font-weight: 700;
        font-size: 0.92rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2f6fb 0%, #e9f0f8 100%);
        border-right: 1px solid #dce6f2;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 3.5rem 1rem 1.2rem;
        }

        .hero-title {
            font-size: 1.5rem;
        }

        .hero-subtitle {
            font-size: 0.9rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path: str, vocab_path: str, device: str = "cuda"):
    """
    Load trained CNN-LSTM model and vocabulary.

    Supports:
    - full checkpoint containing "model_state"
    - raw PyTorch state_dict
    """

    device = torch.device(
        device if torch.cuda.is_available() and device == "cuda"
        else "cpu"
    )

    vocab = load_vocabulary(vocab_path)

    ckpt = torch.load(
        model_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]

        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})

        embed_size = model_cfg.get("embed_dim", 300)
        hidden_size = model_cfg.get("hidden_dim", 512)
        num_layers = model_cfg.get("num_layers", 1)
        dropout = model_cfg.get("dropout", 0.3)

    else:
        # Raw state_dict
        state_dict = ckpt

        # Infer architecture dimensions from weights
        embed_size = state_dict["decoder.embed.weight"].shape[1]

        hidden_size = state_dict["decoder.lstm.weight_hh_l0"].shape[1]

        num_layers = 1
        while f"decoder.lstm.weight_ih_l{num_layers}" in state_dict:
            num_layers += 1

        dropout = 0.3

    checkpoint_vocab_size = state_dict["decoder.embed.weight"].shape[0]
    if len(vocab) != checkpoint_vocab_size:
        raise ValueError(
            f"Vocabolario incompatibile con il checkpoint: il modello richiede "
            f"{checkpoint_vocab_size} token, ma '{vocab_path}' ne contiene "
            f"{len(vocab)}. Per model_mini_demo.pt usa data/_tmp_vocab.json; "
            "per i checkpoint degli esperimenti usa data/vocab.json."
        )

    model = CNNtoLSTM(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers,
        dropout=dropout,
        train_cnn=False,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, vocab, device


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Configurazione")

model_path = st.sidebar.text_input(
    "Model weights (.pt)",
    value="checkpoints/ablation_finetune_cpu/model_best.pt"
)

vocab_path = st.sidebar.text_input(
    "Vocabulary file (.json)",
    value="data/vocab.json"
)

captions_json = st.sidebar.text_input(
    "Reference captions (.json, opzionale)",
    value="data/Flickr8k_text/test_captions.json"
)

st.sidebar.markdown("---")

beam_width = st.sidebar.slider(
    "Ampiezza beam search",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

max_len = st.sidebar.slider(
    "Maximum caption length",
    min_value=10,
    max_value=30,
    value=20,
    step=2
)

default_device_index = 0 if torch.cuda.is_available() else 1

device_choice = st.sidebar.radio(
    "Device",
    options=["cuda", "cpu"],
    index=default_device_index
)

st.sidebar.caption(
    "CUDA disponibile"
    if torch.cuda.is_available()
    else "CUDA non disponibile: verrà utilizzata la CPU."
)


# ============================================================
# LOAD MODEL
# ============================================================

model = None
vocab = None
device = None
load_error = None

try:
    model, vocab, device = load_model(
        model_path,
        vocab_path,
        device_choice
    )
    model_loaded = True

except Exception as e:
    model_loaded = False
    load_error = str(e)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([6, 1])

with header_left:
    st.markdown(
        '<div class="hero-title">🖼️ Image Captioning Demo</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">
        Inferenza in tempo reale con architettura
        <b>CNN-LSTM</b> per la generazione automatica
        di descrizioni testuali a partire da immagini.
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    if model_loaded:
        st.markdown(
            '<div class="status-pill status-ok">● Modello pronto</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-pill status-error">● Errore modello</div>',
            unsafe_allow_html=True
        )


if not model_loaded:
    st.error(
        f"Errore durante il caricamento del modello: {load_error}"
    )
    st.stop()


# ============================================================
# TRANSFORM
# ============================================================

transform = get_transforms(img_size=299)


# ============================================================
# SESSION STATE
# ============================================================

if "pred_caption" not in st.session_state:
    st.session_state.pred_caption = None

if "elapsed" not in st.session_state:
    st.session_state.elapsed = None

if "reference_captions" not in st.session_state:
    st.session_state.reference_captions = []


# ============================================================
# MAIN LAYOUT
# ============================================================

col_upload, col_preview, col_caption = st.columns(
    [0.95, 1.35, 1.0],
    gap="large"
)


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

with col_upload:

    st.markdown(
        '<div class="panel-title">📤 Carica immagine</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Seleziona un'immagine",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.caption(
        "Formati supportati: JPG, JPEG, PNG"
    )

    generate_button = st.button(
        "✨ Genera didascalia",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )


# ------------------------------------------------------------
# PREVIEW
# ------------------------------------------------------------

image = None

with col_preview:

    st.markdown(
        '<div class="panel-title">🖼️ Anteprima immagine</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

    else:

        st.info(
            "Carica un'immagine per visualizzare l'anteprima."
        )


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

if (
    generate_button
    and uploaded_file is not None
    and image is not None
):

    with st.spinner(
        "Analisi dell'immagine e generazione della didascalia..."
    ):

        start_time = time.time()

        image_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )

        pred_caption = generate_caption(
            model=model,
            image_tensor=image_tensor,
            vocab=vocab,
            max_len=max_len,
            beam_width=beam_width,
        )

        elapsed = time.time() - start_time

        st.session_state.pred_caption = pred_caption
        st.session_state.elapsed = elapsed
        st.session_state.reference_captions = []

        # ----------------------------------------------------
        # Reference caption
        # ----------------------------------------------------

        if os.path.exists(captions_json):

            try:

                with open(
                    captions_json,
                    "r",
                    encoding="utf-8"
                ) as f:

                    refs = json.load(f)

                filename = os.path.basename(
                    uploaded_file.name
                )

                if filename in refs:
                    st.session_state.reference_captions = refs[filename]

            except Exception:
                pass


# ------------------------------------------------------------
# CAPTION RESULT
# ------------------------------------------------------------

with col_caption:

    st.markdown(
        '<div class="panel-title">💬 Didascalia generata</div>',
        unsafe_allow_html=True
    )

    if st.session_state.pred_caption:

        safe_caption = (
            str(st.session_state.pred_caption)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if safe_caption.strip().lower() == "&lt;unk&gt;":
            safe_caption = "Nessuna descrizione riconoscibile"

        st.markdown(
            f"""
            <div class="caption-box">
                <div class="caption-text">
                    “{safe_caption}”
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="caption-box">
                <div class="caption-placeholder">
                    Carica un'immagine e genera la didascalia.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# REFERENCE CAPTION
# ============================================================

if st.session_state.reference_captions:

    st.markdown("### 📚 Didascalie di riferimento")

    st.caption("Sono le descrizioni umane associate all'immagine nel test set Flickr8k.")
    for index, reference in enumerate(st.session_state.reference_captions, start=1):
        st.info(f"{index}. {reference}")


# ============================================================
# MODEL INFO
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="panel-title">ℹ️ Informazioni modello</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Architettura</div>
            <div class="info-value">CNN-LSTM</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Encoder</div>
            <div class="info-value">InceptionV3</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Device</div>
            <div class="info-value">{str(device).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with info4:

    elapsed_text = (
        f"{st.session_state.elapsed:.2f} s"
        if st.session_state.elapsed is not None
        else "—"
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Tempo inferenza</div>
            <div class="info-value">{elapsed_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-box">
    Developed by <b>Giuseppe Dimonte</b><br>
    MSc in Computer Engineering – Artificial Intelligence Curriculum<br>
    University of Parma
    </div>
    """,
    unsafe_allow_html=True
)