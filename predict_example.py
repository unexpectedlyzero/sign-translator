"""
Contoh pemakaian model hasil training untuk memprediksi huruf dari landmark tangan.
Cocok dipakai langsung dengan output MediaPipe Hands (21 titik x,y per tangan).
"""

from pathlib import Path

import numpy as np
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "sign_language_model.pkl"
DATA_PATH = SCRIPT_DIR.parent / "Datafull train.csv"
N_LANDMARKS = 21


def normalize_landmarks(X):
    """Harus sama persis dengan normalisasi saat training."""
    X = X.reshape(-1, N_LANDMARKS, 2).astype(np.float64)
    wrist = X[:, 0:1, :]
    X_centered = X - wrist
    dists = np.linalg.norm(X_centered, axis=2)
    scale = dists.max(axis=1, keepdims=True)
    scale[scale == 0] = 1e-6
    X_norm = X_centered / scale[:, :, None]
    return X_norm.reshape(-1, N_LANDMARKS * 2)


def predict_letter(landmarks_flat, bundle):
    """
    landmarks_flat: list/array 42 angka -> [0x,0y,1x,1y,...,20x,20y]
    sesuai urutan titik MediaPipe Hands.
    """
    model = bundle["model"]
    le = bundle["label_encoder"]

    X = np.array(landmarks_flat, dtype=np.float64).reshape(1, -1)
    X_norm = normalize_landmarks(X)

    pred_idx = model.predict(X_norm)[0]
    letter = le.inverse_transform([pred_idx])[0]

    # kalau model punya predict_proba, tampilkan juga tingkat keyakinannya
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_norm)[0]
        confidence = float(proba[pred_idx])

    return letter, confidence


if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)
    print(f"Model yang dimuat: {bundle['model_name']} (akurasi test: {bundle['test_accuracy']:.2%})")

    # contoh: ambil satu baris dari CSV asli sebagai dummy input
    import pandas as pd
    df = pd.read_csv(DATA_PATH, sep=";").dropna()
    sample = df.iloc[0]
    landmarks = sample[bundle["feature_cols"]].values
    true_label = sample["char"]

    letter, conf = predict_letter(landmarks, bundle)
    print(f"Prediksi: {letter} (confidence: {conf:.2%}) | Label asli: {true_label}")
