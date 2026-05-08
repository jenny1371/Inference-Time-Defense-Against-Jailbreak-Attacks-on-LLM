"""
Evaluate the saved TF-IDF + Random Forest classifier on the test set.

Input files:
    data/test_split.csv
    models/random_forest_vectorizer.pkl
    models/random_forest_classifier.pkl

Output file:
    outputs/random_forest_test_results.csv
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

TEST_PATH = DATA_DIR / "test_split.csv"
VECTORIZER_PATH = MODEL_DIR / "random_forest_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "random_forest_classifier.pkl"
RESULTS_OUTPUT_PATH = OUTPUT_DIR / "random_forest_test_results.csv"


# =============================================================================
# Reporting
# =============================================================================

def print_evaluation_report(results_df: pd.DataFrame) -> None:
    """Print overall and per-sample-type metrics."""
    true_labels = results_df["label"]
    predicted_labels = results_df["predicted_label"]

    print("Random Forest test results")
    print(f"Precision: {precision_score(true_labels, predicted_labels, zero_division=0):.4f}")
    print(f"Recall: {recall_score(true_labels, predicted_labels, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predicted_labels, zero_division=0):.4f}")

    print("\nBreakdown by sample type")
    for sample_type in sorted(results_df["data_type"].unique()):
        subset_df = results_df[results_df["data_type"] == sample_type]
        accuracy = (subset_df["label"] == subset_df["predicted_label"]).mean()
        f1 = f1_score(
            subset_df["label"],
            subset_df["predicted_label"],
            zero_division=0,
        )
        print(f"{sample_type:<25} n={len(subset_df):>4} | Accuracy={accuracy:.3f} | F1={f1:.3f}")


# =============================================================================
# Evaluation
# =============================================================================

def main() -> None:
    """Run Random Forest evaluation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = joblib.load(VECTORIZER_PATH)
    random_forest_model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])

    test_features = vectorizer.transform(test_df["prompt"])
    harmful_probabilities = random_forest_model.predict_proba(test_features)[:, 1]
    predicted_labels = (harmful_probabilities >= 0.5).astype(int)

    results_df = test_df.copy()
    results_df["predicted_label"] = predicted_labels
    results_df["prob_harmful"] = harmful_probabilities.round(4)
    results_df["blocked_by_random_forest"] = predicted_labels

    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print_evaluation_report(results_df)


if __name__ == "__main__":
    main()
