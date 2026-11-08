from pathlib import Path
import joblib
import spacy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

nlp = spacy.load("en_core_web_lg")
scaler = MinMaxScaler()

MODEL_PATH = Path(__file__).parent / "model" / "kneigh_model.joblib"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    samples = [
        ("I really enjoyed this movie, great acting.", 1),
        ("The food was cold and the staff was rude.", 0),
        ("Absolutely fantastic experience, would recommend.", 1),
    ]

    print(f"{'text':<55} {'label':<6} {'predicted_label'}")
    print("-" * 80)
    for text, label in samples:
        text_vector = nlp(text).vector

        text_2d = np.stack([text_vector])
        text_scaled = scaler.fit_transform(text_2d)
        pred = int(model.predict([text_vector]))
        print(f"{text:<55} {label:<6} {pred}")

if __name__ == "__main__":
    main()