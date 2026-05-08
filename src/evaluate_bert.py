"""
Evaluate the saved BERT classifier on the test set.

Input files:
    data/test_split.csv
    models/best_bert_classifier.pt

Output file:
    outputs/bert_test_results.csv
"""

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
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
CHECKPOINT_PATH = MODEL_DIR / "best_bert_classifier.pt"
RESULTS_OUTPUT_PATH = OUTPUT_DIR / "bert_test_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 25

CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)


# =============================================================================
# Dataset Class
# =============================================================================

class HarmfulPromptDataset(Dataset):
    """Torch dataset for BERT evaluation."""

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
# Evaluation
# =============================================================================

def main() -> None:
    """Run BERT evaluation and save row-level predictions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(
        HarmfulPromptDataset(test_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predicted_labels_by_index = {}
    harmful_probabilities_by_index = {}

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for row_index, prediction, probability in zip(
                batch["row_index"].numpy(),
                predictions.cpu().numpy(),
                probabilities[:, 1].cpu().numpy(),
            ):
                predicted_labels_by_index[int(row_index)] = int(prediction)
                harmful_probabilities_by_index[int(row_index)] = float(probability)

    results_df = test_df.copy()
    results_df["predicted_label"] = [predicted_labels_by_index[index] for index in test_df.index]
    results_df["prob_harmful"] = [harmful_probabilities_by_index[index] for index in test_df.index]
    results_df["prob_harmful"] = results_df["prob_harmful"].round(4)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)

    true_labels = results_df["label"]
    predicted_labels = results_df["predicted_label"]

    print("BERT test results")
    print(f"Precision: {precision_score(true_labels, predicted_labels, zero_division=0):.4f}")
    print(f"Recall: {recall_score(true_labels, predicted_labels, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predicted_labels, zero_division=0):.4f}")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=["Benign", "Harmful"],
        )
    )


if __name__ == "__main__":
    main()
