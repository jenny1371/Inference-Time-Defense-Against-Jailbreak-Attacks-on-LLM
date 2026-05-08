"""
Generate adversarial examples for the TF-IDF + Random Forest classifier using TextAttack.

Input files:
    data/test_split.csv
    data/train_split.csv
    models/random_forest_vectorizer.pkl
    models/random_forest_classifier.pkl

Output files:
    outputs/random_forest_attack_results.csv
    data/train_split_augmented.csv
"""

from pathlib import Path
import logging

import joblib
import pandas as pd
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import PWWSRen2019
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

TRAIN_PATH = DATA_DIR / "train_split.csv"
TEST_PATH = DATA_DIR / "test_split.csv"
VECTORIZER_PATH = MODEL_DIR / "random_forest_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "random_forest_classifier.pkl"
ATTACK_RESULTS_PATH = OUTPUT_DIR / "random_forest_attack_results.csv"
AUGMENTED_TRAIN_PATH = DATA_DIR / "train_split_augmented.csv"

TARGET_DATA_TYPE = "adversarial_benign"

logging.getLogger("textattack").setLevel(logging.WARNING)


# =============================================================================
# TextAttack Wrapper
# =============================================================================

class TfidfRandomForestWrapper(ModelWrapper):
    """TextAttack-compatible wrapper for a TF-IDF + Random Forest pipeline."""

    def __init__(self, vectorizer, random_forest_model):
        self.vectorizer = vectorizer
        self.random_forest_model = random_forest_model
        self.model = random_forest_model

    def __call__(self, text_list: list[str]):
        features = self.vectorizer.transform(text_list)
        return self.random_forest_model.predict_proba(features)


# =============================================================================
# Attack Runner
# =============================================================================

def main() -> None:
    """Run TextAttack and append successful adversarial examples to training data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = joblib.load(VECTORIZER_PATH)
    random_forest_model = joblib.load(MODEL_PATH)
    model_wrapper = TfidfRandomForestWrapper(vectorizer, random_forest_model)

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])
    target_df = test_df[test_df["data_type"] == TARGET_DATA_TYPE]

    dataset = Dataset(
        [(row["prompt"], row["label"]) for _, row in target_df.iterrows()]
    )

    attack = PWWSRen2019.build(model_wrapper)
    attack_args = AttackArgs(
        num_examples=-1,
        log_to_csv=str(ATTACK_RESULTS_PATH),
        checkpoint_interval=50,
        disable_stdout=True,
        silent=True,
    )

    print(f"Running Random Forest attack on {len(target_df)} samples.")
    attacker = Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

    attacked_df = pd.read_csv(ATTACK_RESULTS_PATH)
    successful_df = attacked_df[attacked_df["result_type"] == "Successful"]

    if successful_df.empty:
        print("No successful Random Forest adversarial examples were generated.")
        return

    new_samples_df = pd.DataFrame(
        {
            "prompt": successful_df["perturbed_text"],
            "label": successful_df["ground_truth_output"],
            "data_type": "adversarial_benign_augmented",
        }
    )

    train_df = pd.read_csv(TRAIN_PATH)
    augmented_train_df = pd.concat([train_df, new_samples_df], ignore_index=True)
    augmented_train_df.to_csv(AUGMENTED_TRAIN_PATH, index=False)

    success_rate = len(successful_df) / len(attacked_df)
    print(f"Attack success rate: {success_rate:.3f}")
    print(f"Saved attack results: {ATTACK_RESULTS_PATH}")
    print(f"Saved augmented training data: {AUGMENTED_TRAIN_PATH}")


if __name__ == "__main__":
    main()
