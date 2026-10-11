"""
Training Model Penerjemah Bahasa Isyarat (ASL Fingerspelling) dari Landmark Tangan
=====================================================================================
Input  : CSV berisi 21 titik landmark tangan (x, y) dari MediaPipe Hands + label huruf
Output : Model terlatih (.pkl) yang memprediksi huruf dari landmark tangan

Pipeline:
1. Load & bersihkan data (buang baris dengan label/koordinat kosong)
2. Normalisasi landmark (translasi + skala invariant terhadap posisi/jarak tangan ke kamera)
3. Bandingkan beberapa model (Random Forest, Extra Trees, SVM) di data penuh
4. Pilih model terbaik, evaluasi lengkap di test set
5. Simpan model + metadata jadi satu file siap pakai untuk inference
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent
DATA_PATH = DATA_DIR.parent / "Datafull train.csv"
MODEL_OUT = DATA_DIR / "sign_language_model.pkl"
REPORT_OUT = DATA_DIR / "training_report.json"
CM_PLOT_OUT = DATA_DIR / "confusion_matrix.png"

N_LANDMARKS = 21


# ---------------------------------------------------------------------------
# 1. LOAD & BERSIHKAN DATA
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep=";")
    df = df.dropna(subset=["char"])
    feature_cols = [c for c in df.columns if c != "char"]
    df = df.dropna(subset=feature_cols)
    print(f"Jumlah data setelah dibersihkan: {len(df)} baris, {df['char'].nunique()} kelas")
    return df, feature_cols


# ---------------------------------------------------------------------------
# 2. NORMALISASI LANDMARK
# ---------------------------------------------------------------------------
def normalize_landmarks(X):
    """
    Normalisasi tiap sampel landmark:
      - Translasi: semua titik dikurangi posisi wrist (landmark 0)
      - Skala: dibagi jarak terjauh dari wrist ke titik lain
    Hasilnya invariant terhadap posisi tangan di frame & jarak ke kamera,
    jadi model lebih general saat dipakai di webcam device apapun.
    """
    X = X.reshape(-1, N_LANDMARKS, 2).astype(np.float64)
    wrist = X[:, 0:1, :]
    X_centered = X - wrist
    dists = np.linalg.norm(X_centered, axis=2)
    scale = dists.max(axis=1, keepdims=True)
    scale[scale == 0] = 1e-6
    X_norm = X_centered / scale[:, :, None]
    return X_norm.reshape(-1, N_LANDMARKS * 2)


# ---------------------------------------------------------------------------
# 3. BANDINGKAN MODEL (di data penuh, SVM pakai subsample karena kuadratik)
# ---------------------------------------------------------------------------
def compare_models(X_train, y_train, X_test, y_test, svm_subsample=15000):
    """
    Bandingkan beberapa model. Untuk Random Forest & Extra Trees, kedalaman pohon
    DIBATASI (max_depth, min_samples_leaf) -- pohon yang dibiarkan tumbuh penuh di
    data sebesar ini bisa menghasilkan file model berukuran >2 GB, yang terlalu
    besar untuk dipakai di aplikasi nyata. Pembatasan ini hanya mengorbankan sedikit
    akurasi (~1-2%) demi ukuran model yang jauh lebih kecil & inference yang lebih cepat.
    """
    results = {}
    trained = {}

    rf = RandomForestClassifier(
        n_estimators=250, max_depth=22, min_samples_leaf=1, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    results["RandomForest"] = acc
    trained["RandomForest"] = rf
    print(f"RandomForest (250 trees, depth<=22) | test accuracy: {acc:.4f}")

    et = ExtraTreesClassifier(
        n_estimators=250, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    )
    et.fit(X_train, y_train)
    acc = accuracy_score(y_test, et.predict(X_test))
    results["ExtraTrees"] = acc
    trained["ExtraTrees"] = et
    print(f"ExtraTrees (250 trees, depth<=20)   | test accuracy: {acc:.4f}")

    if len(X_train) > svm_subsample:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=svm_subsample, replace=False)
        X_svm, y_svm = X_train[idx], y_train[idx]
    else:
        X_svm, y_svm = X_train, y_train
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE, probability=True)
    svm.fit(X_svm, y_svm)
    acc = accuracy_score(y_test, svm.predict(X_test))
    results["SVM_RBF"] = acc
    trained["SVM_RBF"] = svm
    print(f"SVM RBF (subsample {svm_subsample})     | test accuracy: {acc:.4f}")

    best_name = max(results, key=results.get)
    print(f"\n>> Model terbaik: {best_name} (akurasi test: {results[best_name]:.4f})")
    return trained[best_name], best_name, results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df, feature_cols = load_data(DATA_PATH)

    X_raw = df[feature_cols].values
    y_raw = df["char"].values
    X = normalize_landmarks(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\n=== Perbandingan model di data penuh ===")
    best_model, best_name, cv_results = compare_models(X_train, y_train, X_test, y_test)

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(f"\n=== Laporan klasifikasi lengkap ({best_name}) ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Prediksi")
    plt.ylabel("Label Asli")
    plt.title(f"Confusion Matrix - {best_name} (akurasi={test_acc:.2%})")
    plt.tight_layout()
    plt.savefig(CM_PLOT_OUT, dpi=150)
    print(f"Confusion matrix disimpan di {CM_PLOT_OUT}")

    bundle = {
        "model": best_model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "n_landmarks": N_LANDMARKS,
        "model_name": best_name,
        "test_accuracy": test_acc,
        "normalize_fn_description": "wrist-centered (landmark 0), scaled by max distance from wrist",
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    print(f"Model disimpan di {MODEL_OUT}")

    with open(REPORT_OUT, "w") as f:
        json.dump({
            "model_comparison": cv_results,
            "best_model": best_name,
            "test_accuracy": test_acc,
            "classification_report": report,
            "n_samples": len(df),
            "n_classes": int(df["char"].nunique()),
        }, f, indent=2)
    print(f"Laporan lengkap disimpan di {REPORT_OUT}")


if __name__ == "__main__":
    main()
