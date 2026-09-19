# Tile Defect Inspector

An autoencoder-based anomaly detection system for identifying cracks, chips, and blemishes on ceramic tile surfaces.

**Live demo:** https://tile-defect-gui-qenwnefbtwywdy4yjtrsxp.streamlit.app/

## Features

- Upload a tile image and get an instant GOOD / DEFECTIVE verdict
- Anomaly heatmap showing where the model detected irregularities
- Adjustable detection sensitivity (Lenient / Balanced / Strict)
- Result history for previously checked images

## How It Works

The model is trained only on defect-free tile images. It learns to reconstruct what a normal tile looks like, so when it's shown a defective tile, the reconstruction is noticeably worse. That reconstruction error is used to flag the image as defective.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Fezeka1/tile-defect-gui.git
   cd tile-defect-gui
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the App

```
streamlit run app.py
```

This opens the app in your browser. It automatically loads `best_baseline_autoencoder.pth` and `calibration.json` from the project folder.

## Retraining / Re-evaluating the Model

The dataset is not included in this repository. Download the MVTec AD "tile" category, then:

```
python evaluate.py --data-root /path/to/tile --checkpoint best_baseline_autoencoder.pth --out calibration.json
python train.py --data-root /path/to/tile --out optimized_autoencoder.pth --augment
```

## Tech Stack

- PyTorch / torchvision
- scikit-learn
- Streamlit
- Pillow, NumPy, Matplotlib
