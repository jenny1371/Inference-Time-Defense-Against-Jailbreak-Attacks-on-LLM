"""
Visualize feature importance for the saved TF-IDF + Random Forest classifier.

Input files:
    models/random_forest_vectorizer.pkl
    models/random_forest_classifier.pkl

Output file:
    figures/random_forest_feature_importance.png
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = Path("models")
FIGURE_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")

VECTORIZER_PATH = MODEL_DIR / "random_forest_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "random_forest_classifier.pkl"
FIGURE_OUTPUT_PATH = FIGURE_DIR / "random_forest_feature_importance.png"
TABLE_OUTPUT_PATH = OUTPUT_DIR / "random_forest_top_features.csv"

TOP_N_FEATURES = 20


# =============================================================================
# Explainability
# =============================================================================

def main() -> None:
    """Save a feature importance plot and CSV table."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = joblib.load(VECTORIZER_PATH)
    random_forest_model = joblib.load(MODEL_PATH)

    feature_names = vectorizer.get_feature_names_out()
    feature_importances = random_forest_model.feature_importances_

    top_indices = np.argsort(feature_importances)[-TOP_N_FEATURES:]
    top_features = feature_names[top_indices]
    top_values = feature_importances[top_indices]

    top_features_df = pd.DataFrame(
        {
            "feature": list(reversed(top_features)),
            "importance": list(reversed(top_values)),
        }
    )
    top_features_df.to_csv(TABLE_OUTPUT_PATH, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(top_features)), list(reversed(top_values)))
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Top 20 Important TF-IDF Features")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved figure: {FIGURE_OUTPUT_PATH}")
    print(f"Saved table: {TABLE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
