"""
Evaluate a two-stage harmful prompt detection pipeline.

Stage 1: TF-IDF + Random Forest
Stage 2: BERT classifier on prompts passed by Stage 1

Input files:
    data/test_split.csv
    models/random_forest_vectorizer.pkl
    models/random_forest_classifier.pkl
    models/best_bert_classifier.pt

Output file:
    outputs/pipeline_test_results.csv
"""

from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "bert-base-uncased"

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

TEST_PATH = DATA_DIR / "test_split.csv"
RF_VECTORIZER_PATH = MODEL_DIR / "random_forest_vectorizer.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_classifier.pkl"
BERT_CHECKPOINT_PATH = MODEL_DIR / "best_bert_classifier.pt"
PIPELINE_RESULTS_PATH = OUTPUT_DIR / "pipeline_test_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 25
CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)

BASELINE_RESULTS = {
    "LLaMA-8B Naive": {"precision": "N/A", "recall": "~0", "f1": "0.0972"},
    "LLaMA-8B Strict": {"precision": "0.8980", "recall": "0.6423", "f1": "0.7489"},
    "BERT Only": {"precision": "0.9780", "recall": "0.9773", "f1": "0.9777"},
}


# =============================================================================
# Dataset Class
# =============================================================================

class BertPromptDataset(Dataset):
    """Dataset for BERT inference on prompts passed by the Random Forest layer."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer):
        self.texts = (CLASSIFICATION_PREFIX + dataframe["prompt"].astype(str)).tolist()
        self.labels = dataframe["label"].tolist()
        self.indices = dataframe.index.tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded_input = self.tokenizer(
            self.texts[index],
            max_length=MAX_SEQUENCE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "row_index": self.indices[index],
        }


# =============================================================================
# Helper Functions
# =============================================================================

def print_sample_type_breakdown(results_df: pd.DataFrame, prediction_column: str) -> None:
    """Print accuracy and F1 by data_type."""
    print("\nBreakdown by sample type")
    for sample_type in sorted(results_df["data_type"].unique()):
        subset_df = results_df[results_df["data_type"] == sample_type]
        accuracy = (subset_df["label"] == subset_df[prediction_column]).mean()
        sample_f1 = f1_score(
            subset_df["label"],
            subset_df[prediction_column],
            zero_division=0,
        )
        print(f"{sample_type:<25} n={len(subset_df):>4} | Accuracy={accuracy:.3f} | F1={sample_f1:.3f}")


def print_model_comparison(
    rf_precision: float,
    rf_recall: float,
    rf_f1: float,
    pipeline_precision: float,
    pipeline_recall: float,
    pipeline_f1: float,
) -> None:
    """Print a compact model comparison table."""
    print("\nModel comparison on the test set")
    print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")

    for model_name, metrics in BASELINE_RESULTS.items():
        print(
            f"{model_name:<30} "
            f"{metrics['precision']:>10} "
            f"{metrics['recall']:>10} "
            f"{metrics['f1']:>10}"
        )

    print(f"{'TF-IDF + Random Forest':<30} {rf_precision:>10.4f} {rf_recall:>10.4f} {rf_f1:>10.4f}")
    print(f"{'RF + BERT Pipeline':<30} {pipeline_precision:>10.4f} {pipeline_recall:>10.4f} {pipeline_f1:>10.4f}")


# =============================================================================
# Pipeline Evaluation
# =============================================================================

def main() -> None:
    """Run the two-stage evaluation pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])
    true_labels = test_df["label"].tolist()

    vectorizer = joblib.load(RF_VECTORIZER_PATH)
    random_forest_model = joblib.load(RF_MODEL_PATH)

    test_features = vectorizer.transform(test_df["prompt"])
    rf_predictions = random_forest_model.predict(test_features)
    rf_harmful_probabilities = random_forest_model.predict_proba(test_features)[:, 1]

    rf_blocked_indices = set(test_df.index[rf_predictions == 1].tolist())
    rf_passed_df = test_df[rf_predictions == 0].copy()

    rf_precision = precision_score(true_labels, rf_predictions, zero_division=0)
    rf_recall = recall_score(true_labels, rf_predictions, zero_division=0)
    rf_f1 = f1_score(true_labels, rf_predictions, zero_division=0)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    bert_model.load_state_dict(torch.load(BERT_CHECKPOINT_PATH, map_location=DEVICE))
    bert_model.to(DEVICE)
    bert_model.eval()

    bert_loader = DataLoader(
        BertPromptDataset(rf_passed_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    bert_predictions_by_index = {}
    bert_harmful_probabilities_by_index = {}

    with torch.no_grad():
        for batch in bert_loader:
            outputs = bert_model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for row_index, prediction, harmful_probability in zip(
                batch["row_index"].numpy(),
                predictions.cpu().numpy(),
                probabilities[:, 1].cpu().numpy(),
            ):
                bert_predictions_by_index[int(row_index)] = int(prediction)
                bert_harmful_probabilities_by_index[int(row_index)] = float(harmful_probability)

    pipeline_predictions = []
    bert_predictions = []
    bert_harmful_probabilities = []

    for row_index in test_df.index:
        if row_index in rf_blocked_indices:
            pipeline_predictions.append(1)
            bert_predictions.append(None)
            bert_harmful_probabilities.append(None)
        else:
            pipeline_predictions.append(bert_predictions_by_index.get(row_index, 0))
            bert_predictions.append(bert_predictions_by_index.get(row_index, 0))
            bert_harmful_probabilities.append(bert_harmful_probabilities_by_index.get(row_index))

    results_df = test_df.copy()
    results_df["random_forest_predicted_label"] = rf_predictions
    results_df["random_forest_prob_harmful"] = rf_harmful_probabilities.round(4)
    results_df["bert_predicted_label"] = bert_predictions
    results_df["bert_prob_harmful"] = bert_harmful_probabilities
    results_df["pipeline_predicted_label"] = pipeline_predictions
    results_df.to_csv(PIPELINE_RESULTS_PATH, index=False)

    pipeline_precision = precision_score(true_labels, pipeline_predictions, zero_division=0)
    pipeline_recall = recall_score(true_labels, pipeline_predictions, zero_division=0)
    pipeline_f1 = f1_score(true_labels, pipeline_predictions, zero_division=0)

    print("Pipeline evaluation complete.")
    print(f"Prompts blocked by Random Forest: {sum(rf_predictions)}/{len(test_df)}")
    print(f"Prompts passed to BERT: {len(rf_passed_df)}/{len(test_df)}")
    print("\nFinal pipeline results")
    print(
        classification_report(
            true_labels,
            pipeline_predictions,
            target_names=["Benign", "Harmful"],
        )
    )

    confusion = confusion_matrix(true_labels, pipeline_predictions)
    print(
        "Confusion matrix: "
        f"TN={confusion[0][0]} FP={confusion[0][1]} "
        f"FN={confusion[1][0]} TP={confusion[1][1]}"
    )

    print_sample_type_breakdown(results_df, "pipeline_predicted_label")
    print_model_comparison(
        rf_precision,
        rf_recall,
        rf_f1,
        pipeline_precision,
        pipeline_recall,
        pipeline_f1,
    )


if __name__ == "__main__":
    main()
