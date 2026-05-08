"""
Run interactive inference with the saved BERT harmful prompt classifier.

Input file:
    models/best_bert_classifier.pt
"""

from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "bert-base-uncased"
MODEL_DIR = Path("models")
CHECKPOINT_PATH = MODEL_DIR / "best_bert_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
CLASSIFICATION_THRESHOLD = 0.5

CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)


# =============================================================================
# Model Loading
# =============================================================================

def load_model() -> tuple[BertTokenizer, BertForSequenceClassification]:
    """Load the tokenizer and fine-tuned BERT classifier."""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# =============================================================================
# Prediction
# =============================================================================

def predict_prompt(
    prompt_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> dict:
    """Predict whether a prompt is benign or harmful."""
    encoded_input = tokenizer(
        CLASSIFICATION_PREFIX + prompt_text,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded_input["input_ids"].to(DEVICE),
            attention_mask=encoded_input["attention_mask"].to(DEVICE),
        )
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    harmful_probability = probabilities[1].item()
    benign_probability = probabilities[0].item()
    predicted_label = "HARMFUL" if harmful_probability >= CLASSIFICATION_THRESHOLD else "BENIGN"

    return {
        "label": predicted_label,
        "harmful_probability": harmful_probability,
        "benign_probability": benign_probability,
    }


# =============================================================================
# Command Line Interface
# =============================================================================

def main() -> None:
    """Start an interactive prompt classification session."""
    tokenizer, model = load_model()

    print("Model loaded successfully.")
    print("Type 'quit' to exit.\n")

    while True:
        prompt_text = input("Prompt: ").strip()

        if prompt_text.lower() == "quit":
            break

        if not prompt_text:
            continue

        result = predict_prompt(prompt_text, tokenizer, model)

        print(f"\nPrediction: {result['label']}")
        print(f"P(harmful): {result['harmful_probability']:.4f}")
        print(f"P(benign): {result['benign_probability']:.4f}\n")


if __name__ == "__main__":
    main()
