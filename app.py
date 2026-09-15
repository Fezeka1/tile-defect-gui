"""
Streamlit GUI for tile/glass surface defect inspection.
 
Run with:
    streamlit run app.py
 
Looks for (in order of preference):
    model_scripted.pt   -- TorchScript-optimized model (faster, see optimize_speed.py)
    optimized_autoencoder.pth -- retrained/optimized checkpoint (see train.py)
    best_baseline_autoencoder.pth -- original baseline checkpoint
 
And for calibration.json (produced by evaluate.py) to pre-fill the
recommended anomaly threshold. Everything still works without these --
sensible defaults are used and clearly labeled as such.
"""
import json
import os
import time
 
import numpy as np
import streamlit as st
import torch
from PIL import Image
 
from dataset import encoder_transform, target_transform
from model import load_autoencoder
 
st.set_page_config(page_title="Tile/Glass Defect Inspector", layout="wide")
 
CANDIDATE_MODELS = [
    ("model_scripted.pt", "torchscript"),
    ("optimized_autoencoder.pth", "state_dict"),
    ("best_baseline_autoencoder.pth", "state_dict"),
]
CALIBRATION_PATH = "calibration.json"
 
 
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    for path, kind in CANDIDATE_MODELS:
        if os.path.exists(path):
            if kind == "torchscript":
                model = torch.jit.load(path, map_location=device)
                model.eval()
            else:
                model = load_autoencoder(path, device=device)
            return model, path
    return None, None
 
 
@st.cache_data
def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return None
 
 
def run_inference(model, pil_image):
    encoder_input = encoder_transform(pil_image).unsqueeze(0)
    target = target_transform(pil_image).unsqueeze(0)
 
    start = time.time()
    with torch.no_grad():
        reconstruction = model(encoder_input)
    latency_ms = (time.time() - start) * 1000
 
    pixel_error = ((reconstruction - target) ** 2).mean(dim=1).squeeze(0).numpy()  # [H, W]
    image_error = float(pixel_error.mean())
 
    recon_np = reconstruction.squeeze(0).permute(1, 2, 0).numpy()
    recon_np = np.clip(recon_np, 0, 1)
 
    return recon_np, pixel_error, image_error, latency_ms
 
 
def heatmap_overlay(target_pil, pixel_error):
    import matplotlib.cm as cm
 
    target_np = np.array(target_pil.resize((128, 128))) / 255.0
    normed = (pixel_error - pixel_error.min()) / (pixel_error.max() - pixel_error.min() + 1e-8)
    heat = cm.get_cmap("inferno")(normed)[..., :3]
    overlay = 0.55 * target_np + 0.45 * heat
    return np.clip(overlay, 0, 1)
 
 
def main():
    st.title("Tile / Glass Surface Defect Inspector")
    st.caption("Autoencoder-based anomaly detection — trained only on defect-free surfaces; "
               "defects are flagged via reconstruction error.")
 
    model, model_path = load_model()
    if model is None:
        st.error(
            "No model checkpoint found. Place one of "
            f"{[c[0] for c in CANDIDATE_MODELS]} in this directory."
        )
        return
 
    calibration = load_calibration()
    default_threshold = 0.01
    roc_auc_note = None
    if calibration:
        default_threshold = calibration.get("recommended_threshold", default_threshold)
        roc_auc_note = calibration.get("image_level_roc_auc")
 
    with st.sidebar:
        st.subheader("Model")
        st.write(f"Loaded: `{model_path}`")
        if roc_auc_note is not None:
            st.write(f"Held-out image-level ROC-AUC: **{roc_auc_note:.3f}**")
 
        st.subheader("Anomaly threshold")
        threshold = st.slider(
            "Reconstruction-error threshold",
            min_value=0.0, max_value=0.05,
            value=float(default_threshold), step=0.0005, format="%.4f",
        )
        if calibration:
            st.caption(
                f"Calibrated from held-out good images "
                f"(95th percentile = {calibration['thresholds']['unsupervised_percentile']:.4f}). "
                "Lower = more sensitive (more false alarms). "
                "Higher = fewer false alarms (may miss subtle defects)."
            )
        else:
            st.caption("No calibration.json found — using an uncalibrated default. "
                       "Run evaluate.py first for a data-driven threshold.")
 
    uploaded_files = st.file_uploader(
        "Upload one or more tile/glass images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
 
    if not uploaded_files:
        st.info("Upload an image to run inspection.")
        return
 
    for uploaded_file in uploaded_files:
        pil_image = Image.open(uploaded_file).convert("RGB")
        recon_np, pixel_error, image_error, latency_ms = run_inference(model, pil_image)
        is_defective = image_error > threshold
 
        st.divider()
        st.subheader(uploaded_file.name)
 
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(pil_image, caption="Original", use_container_width=True)
        with col2:
            st.image(recon_np, caption="Reconstruction", use_container_width=True)
        with col3:
            overlay = heatmap_overlay(pil_image, pixel_error)
            st.image(overlay, caption="Anomaly heatmap", use_container_width=True)
        with col4:
            verdict = "DEFECTIVE" if is_defective else "GOOD"
            color = "red" if is_defective else "green"
            st.markdown(f"### :{color}[{verdict}]")
            st.metric("Reconstruction error", f"{image_error:.5f}")
            st.metric("Threshold", f"{threshold:.5f}")
            st.metric("Inference latency", f"{latency_ms:.1f} ms")
 
 
if __name__ == "__main__":
    main()
