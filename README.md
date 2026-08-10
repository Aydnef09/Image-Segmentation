# Classical Image Segmentation — Coin Detection

A comparative study of five classical segmentation configurations across three method families, evaluated on a ten-image coin dataset with pixel-accurate ground-truth masks.

## What it does
- **Thresholding:** global Otsu, adaptive Gaussian (default + tuned)
- **Clustering:** K-Means (k=2) in grey / RGB / Lab feature spaces
- **Watershed:** marker-controlled (Otsu seed → distance transform → peak markers → flooding)
- **Pre-processing:** none / blur / CLAHE / CLAHE+blur, with ablation
- **Evaluation:** IoU, Dice, Precision, Recall, F1 + full 10-image heatmap
- **Failure analysis:** why adaptive thresholding produces ring artefacts on filled discs

## Key result
Otsu + CLAHE and K-Means tie at mean IoU 0.838 (Otsu is 20× faster, zero parameters). Watershed is the only method that separates individual coin instances. Default adaptive thresholding is structurally unsuited to filled circular objects.

## Run
```bash
pip install opencv-python-headless scikit-image scikit-learn numpy scipy matplotlib
python segmentation.py
```
Generates the dataset from scikit-image's coins photo + variants, then runs all methods and writes figures.
