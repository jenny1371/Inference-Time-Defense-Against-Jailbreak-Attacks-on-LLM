"""
Estimate global BERT token importance using Gradient x Input.

Input files:
    data/test_split.csv
    models/best_bert_classifier.pt

Output files:
    figures/bert_token_importance.png
    outputs/bert_token_importance.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FIGURE_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")

TEST_PATH = DATA_DIR / "test_split.csv"
CHECKPOINT_PATH = MODEL_DIR / "best_bert_classifier.pt"
FIGURE_OUTPUT_PATH = FIGURE_DIR / "bert_token_importance.png"
TABLE_OUTPUT_PATH = OUTPUT_DIR / "bert_token_importance.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
N_SAMPLES = 500
MIN_TOKEN_COUNT = 3
TOP_N_TOKENS = 20
TARGET_CLASS = 1

CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


# =============================================================================
# Attribution
# =============================================================================

def get_gradient_input_attributions(
    prompt_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    target_class: int = TARGET_CLASS,
) -> tuple[list[str], np.ndarray, torch.Tensor]:
    """Compute token-level Gradient x Input scores for one prompt."""
    encoded_input = tokenizer(
        CLASSIFICATION_PREFIX + prompt_text,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded_input["input_ids"].to(DEVICE)
    attention_mask = encoded_input["attention_mask"].to(DEVICE)

    embedding_layer = model.bert.embeddings.word_embeddings
    input_embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
    )

    target_probability = torch.softmax(outputs.logits, dim=1)[0, target_class]
    target_probability.backward()

    gradients = input_embeddings.grad.squeeze(0)
    embeddings = input_embeddings.squeeze(0).detach()
    attribution_scores = (gradients * embeddings).sum(dim=-1).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    mask = attention_mask[0].cpu().numpy()
    tokens = [token for token, mask_value in zip(tokens, mask) if mask_value == 1]
    attribution_scores = attribution_scores[: len(tokens)]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tokens, attribution_scores, probabilities


# =============================================================================
# Global Token Importance
# =============================================================================

def main() -> None:
    """Compute and save average token importance over a sampled test subset."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])
    sampled_df = test_df.sample(n=min(N_SAMPLES, len(test_df)), random_state=SEED)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    token_score_sums = {}
    token_counts = {}

    print(f"Computing BERT attributions for {len(sampled_df)} samples.")

    for row_number, (_, row) in enumerate(sampled_df.iterrows(), start=1):
        if row_number % 50 == 0:
            print(f"Processed {row_number}/{len(sampled_df)} samples.")

        try:
            tokens, scores, _ = get_gradient_input_attributions(
                str(row["prompt"]),
                tokenizer,
                model,
            )
        except RuntimeError as error:
            print(f"Skipped one sample due to attribution error: {error}")
            continue

        for token, score in zip(tokens, scores):
            if token in SPECIAL_TOKENS or token.startswith("##"):
                continue
            token_score_sums[token] = token_score_sums.get(token, 0.0) + abs(float(score))
            token_counts[token] = token_counts.get(token, 0) + 1

    average_scores = {
        token: token_score_sums[token] / token_counts[token]
        for token in token_score_sums
        if token_counts[token] >= MIN_TOKEN_COUNT
    }

    top_tokens = sorted(
        average_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:TOP_N_TOKENS]

    importance_df = pd.DataFrame(top_tokens, columns=["token", "average_abs_gradient_input"])
    importance_df.to_csv(TABLE_OUTPUT_PATH, index=False)

    tokens, values = zip(*top_tokens)

    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(tokens)), list(reversed(values)))
    plt.xlabel("Average |Gradient x Input|")
    plt.title("BERT Top 20 Important Tokens")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved figure: {FIGURE_OUTPUT_PATH}")
    print(f"Saved table: {TABLE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
