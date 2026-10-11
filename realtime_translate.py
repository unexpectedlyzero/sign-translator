"""
Demo Real-time: Penerjemah Bahasa Isyarat dari Webcam
========================================================
Alur: Webcam -> MediaPipe Hands (deteksi 21 landmark) -> Model terlatih -> Huruf

Cara pakai:
    python realtime_translate.py

Tekan 'q' untuk keluar.

Catatan: script ini butuh webcam yang bisa diakses OpenCV (index 0 secara default).
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "sign_language_model.pkl"
N_LANDMARKS = 21
CONF_THRESHOLD = 0.5  # di bawah ini, dianggap "tidak yakin" dan tidak ditampilkan


def normalize_landmarks(X):
    """Harus identik dengan normalisasi saat training."""
    X = X.reshape(-1, N_LANDMARKS, 2).astype(np.float64)
    wrist = X[:, 0:1, :]
    X_centered = X - wrist
    dists = np.linalg.norm(X_centered, axis=2)
    scale = dists.max(axis=1, keepdims=True)
    scale[scale == 0] = 1e-6
    X_norm = X_centered / scale[:, :, None]
    return X_norm.reshape(-1, N_LANDMARKS * 2)


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    le = bundle["label_encoder"]
    print(f"Model dimuat: {bundle['model_name']} (akurasi test: {bundle['test_accuracy']:.2%})")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa membuka webcam. Cek koneksi kamera / index device.")
        cap.release()
        hands.close()
        return

    try:
        print("Tekan 'q' untuk keluar.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label_text = ""
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                # urutan harus sama persis dengan kolom CSV: 0x,0y,1x,1y,...,20x,20y
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append(lm.x * w)
                    coords.append(lm.y * h)
                coords = np.array(coords, dtype=np.float64).reshape(1, -1)

                X_norm = normalize_landmarks(coords)
                pred_idx = model.predict(X_norm)[0]
                letter = le.inverse_transform([pred_idx])[0]

                confidence = 1.0
                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(X_norm)[0][pred_idx]

                if confidence >= CONF_THRESHOLD:
                    label_text = f"{letter} ({confidence:.0%})"

            cv2.rectangle(frame, (0, 0), (220, 70), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            cv2.imshow("Penerjemah Bahasa Isyarat", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
