"""
Train a TF-IDF + Random Forest classifier for harmful prompt detection.

Input files:
    data/train_split.csv
    data/test_split.csv

Output files:
    models/random_forest_vectorizer.pkl
    models/random_forest_classifier.pkl
    outputs/random_forest_test_results.csv
    outputs/random_forest_passed_prompts.csv
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score


# =============================================================================
# Configuration
# =============================================================================

SEED = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

TRAIN_PATH = DATA_DIR / "train_split.csv"
AUGMENTED_TRAIN_PATH = DATA_DIR / "train_split_augmented.csv"
TEST_PATH = DATA_DIR / "test_split.csv"

VECTORIZER_OUTPUT_PATH = MODEL_DIR / "random_forest_vectorizer.pkl"
MODEL_OUTPUT_PATH = MODEL_DIR / "random_forest_classifier.pkl"
RESULTS_OUTPUT_PATH = OUTPUT_DIR / "random_forest_test_results.csv"
PASSED_OUTPUT_PATH = OUTPUT_DIR / "random_forest_passed_prompts.csv"

USE_AUGMENTED_TRAINING_DATA = False


# =============================================================================
# Helper Functions
# =============================================================================

def evaluate_predictions(true_labels, predicted_labels) -> dict:
    """Return binary classification metrics."""
    return {
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, zero_division=0),
    }


# =============================================================================
# Training and Evaluation
# =============================================================================

def main() -> None:
    """Train and evaluate the Random Forest classifier."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = AUGMENTED_TRAIN_PATH if USE_AUGMENTED_TRAINING_DATA else TRAIN_PATH
    train_df = pd.read_csv(train_path).dropna(subset=["prompt"])
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])

    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )

    train_features = vectorizer.fit_transform(train_df["prompt"])
    train_labels = train_df["label"]

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=SEED,
    )

    random_forest_model.fit(train_features, train_labels)

    test_features = vectorizer.transform(test_df["prompt"])
    harmful_probabilities = random_forest_model.predict_proba(test_features)[:, 1]
    predicted_labels = (harmful_probabilities >= 0.5).astype(int)

    results_df = test_df.copy()
    results_df["predicted_label"] = predicted_labels
    results_df["prob_harmful"] = harmful_probabilities.round(4)
    results_df["blocked_by_random_forest"] = predicted_labels

    metrics = evaluate_predictions(test_df["label"], predicted_labels)

    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)

    passed_df = results_df[results_df["blocked_by_random_forest"] == 0]
    passed_df[["prompt", "data_type", "label", "prob_harmful"]].to_csv(
        PASSED_OUTPUT_PATH,
        index=False,
    )

    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)
    joblib.dump(random_forest_model, MODEL_OUTPUT_PATH)

    print("Random Forest training complete.")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
