"""
Fine-tune BERT for binary harmful prompt classification.

Input files:
    data/train_split.csv
    data/val_split.csv

Output files:
    models/best_bert_classifier.pt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

TRAIN_PATH = DATA_DIR / "train_split.csv"
AUGMENTED_TRAIN_PATH = DATA_DIR / "train_split_augmented.csv"
VALIDATION_PATH = DATA_DIR / "val_split.csv"
CHECKPOINT_OUTPUT_PATH = MODEL_DIR / "best_bert_classifier.pt"

USE_AUGMENTED_TRAINING_DATA = False
TRAIN_SAMPLE_LIMIT = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)


torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Dataset Class
# =============================================================================

class HarmfulPromptDataset(Dataset):
    """Torch dataset for harmful prompt classification."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer):
        self.texts = dataframe["input_text"].tolist()
        self.labels = dataframe["label"].tolist()
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
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


# =============================================================================
# Helper Functions
# =============================================================================

def build_model_input(text: str) -> str:
    """Add the task prefix used for consequence-based classification."""
    return CLASSIFICATION_PREFIX + str(text)


def evaluate_model(
    model: BertForSequenceClassification,
    data_loader: DataLoader,
) -> tuple[list[int], list[int]]:
    """Run model inference and return true labels and predictions."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_predictions


# =============================================================================
# Training
# =============================================================================

def main() -> None:
    """Fine-tune BERT and save the best checkpoint by validation F1."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = AUGMENTED_TRAIN_PATH if USE_AUGMENTED_TRAINING_DATA else TRAIN_PATH
    train_df = pd.read_csv(train_path).dropna(subset=["prompt"])
    validation_df = pd.read_csv(VALIDATION_PATH).dropna(subset=["prompt"])

    if TRAIN_SAMPLE_LIMIT:
        train_df = train_df.head(TRAIN_SAMPLE_LIMIT)

    train_df["input_text"] = train_df["prompt"].apply(build_model_input)
    validation_df["input_text"] = validation_df["prompt"].apply(build_model_input)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = HarmfulPromptDataset(train_df, tokenizer)
    validation_dataset = HarmfulPromptDataset(validation_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].tolist(),
    )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_training_steps = len(train_loader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_training_steps // 10,
        num_training_steps=total_training_steps,
    )

    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_validation_f1 = 0.0
    best_epoch = 0

    print("Starting BERT fine-tuning.")
    print(f"Device: {DEVICE}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        cumulative_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            cumulative_loss += loss.item()

        average_training_loss = cumulative_loss / len(train_loader)
        validation_labels, validation_predictions = evaluate_model(model, validation_loader)

        validation_precision = precision_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )
        validation_recall = recall_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )
        validation_f1 = f1_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Loss: {average_training_loss:.4f} | "
            f"Precision: {validation_precision:.4f} | "
            f"Recall: {validation_recall:.4f} | "
            f"F1: {validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_OUTPUT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_OUTPUT_PATH, map_location=DEVICE))
    validation_labels, validation_predictions = evaluate_model(model, validation_loader)

    print(f"Training complete. Best validation F1: {best_validation_f1:.4f} at epoch {best_epoch}.")
    print("Final validation report:")
    print(
        classification_report(
            validation_labels,
            validation_predictions,
            target_names=["Benign", "Harmful"],
        )
    )


if __name__ == "__main__":
    main()
