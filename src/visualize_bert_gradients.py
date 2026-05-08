"""
Visualize selected BERT error cases using token-level Gradient x Input attribution.

Input files:
    outputs/bert_test_results.csv
    outputs/bert_attack_results.csv
    models/best_bert_classifier.pt

Output files:
    figures/bert_gradient_<case_type>_part<part_number>.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")

BERT_TEST_RESULTS_PATH = OUTPUT_DIR / "bert_test_results.csv"
BERT_ATTACK_RESULTS_PATH = OUTPUT_DIR / "bert_attack_results.csv"
CHECKPOINT_PATH = MODEL_DIR / "best_bert_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
N_SAMPLES_PER_CASE = 10
TOKENS_PER_ROW = 18
CASES_PER_FIGURE = 5
TARGET_CLASS = 1

CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)


# =============================================================================
# Attribution
# =============================================================================

def get_gradient_input_attributions(
    prompt_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> tuple[list[str], np.ndarray, torch.Tensor]:
    """Compute token-level Gradient x Input attribution for one prompt."""
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

    outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
    target_score = outputs.logits[0, TARGET_CLASS]
    target_score.backward()

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
# Plotting
# =============================================================================

def sample_cases(dataframe: pd.DataFrame, n_samples: int = N_SAMPLES_PER_CASE) -> pd.DataFrame:
    """Sample up to n cases with a fixed random seed."""
    return dataframe.sample(n=min(n_samples, len(dataframe)), random_state=SEED)


def render_case_on_axis(
    axis,
    tokens: list[str],
    scores: np.ndarray,
    probabilities: torch.Tensor,
    title: str,
) -> None:
    """Render one token attribution case onto a matplotlib axis."""
    max_abs_score = max(abs(scores.max()), abs(scores.min())) + 1e-8
    normalized_scores = scores / max_abs_score

    predicted_label = "HARMFUL" if probabilities[1] > 0.5 else "BENIGN"
    axis.set_title(
        f"{title} [{predicted_label}] P(harmful)={probabilities[1]:.3f} "
        f"P(benign)={probabilities[0]:.3f}",
        fontsize=8,
        pad=4,
        loc="left",
    )
    axis.axis("off")

    token_rows = [tokens[index : index + TOKENS_PER_ROW] for index in range(0, len(tokens), TOKENS_PER_ROW)]
    score_rows = [
        normalized_scores[index : index + TOKENS_PER_ROW]
        for index in range(0, len(normalized_scores), TOKENS_PER_ROW)
    ]

    row_height = 1.0 / (len(token_rows) + 1)
    column_width = 1.0 / (TOKENS_PER_ROW + 1)

    for row_index, (row_tokens, row_scores) in enumerate(zip(token_rows, score_rows)):
        y_position = 1.0 - (row_index + 1) * row_height

        for column_index, (token, score) in enumerate(zip(row_tokens, row_scores)):
            x_position = (column_index + 0.5) * column_width

            if score > 0:
                face_color = (1, 1 - score, 1 - score)
            else:
                face_color = (1 + score, 1 + score, 1)

            face_color = tuple(max(0.0, min(1.0, value)) for value in face_color)

            axis.text(
                x_position,
                y_position,
                token,
                transform=axis.transAxes,
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": face_color,
                    "edgecolor": "gray",
                    "linewidth": 0.4,
                },
            )


def save_case_figure(results_chunk: list, case_type: str, part_number: int) -> None:
    """Save a group of attribution visualizations."""
    figure_count = len(results_chunk)
    figure, axes = plt.subplots(
        figure_count,
        1,
        figsize=(20, 4 * figure_count),
        gridspec_kw={"hspace": 0.6},
    )

    if figure_count == 1:
        axes = [axes]

    for axis, (case_index, tokens, scores, probabilities) in zip(axes, results_chunk):
        render_case_on_axis(
            axis,
            tokens,
            scores,
            probabilities,
            title=f"{case_type} #{case_index}",
        )

    figure_path = FIGURE_DIR / f"bert_gradient_{case_type.lower()}_part{part_number}.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {figure_path}")


# =============================================================================
# Visualization Runner
# =============================================================================

def main() -> None:
    """Generate attribution figures for selected error cases."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    bert_test_df = pd.read_csv(BERT_TEST_RESULTS_PATH)
    bert_attack_df = pd.read_csv(BERT_ATTACK_RESULTS_PATH)

    false_positive_df = bert_test_df[
        (bert_test_df["label"] == 0) & (bert_test_df["predicted_label"] == 1)
    ]
    false_negative_df = bert_test_df[
        (bert_test_df["label"] == 1) & (bert_test_df["predicted_label"] == 0)
    ]
    adversarial_benign_false_positive_df = bert_attack_df[
        (bert_attack_df["label"] == 0) & (bert_attack_df["new_predicted_label"] == 1)
    ]

    case_groups = {
        "FP": (sample_cases(false_positive_df), "prompt"),
        "FN": (sample_cases(false_negative_df), "prompt"),
        "ADV_BEN_FP": (sample_cases(adversarial_benign_false_positive_df), "original_prompt"),
    }

    for case_type, (case_df, prompt_column) in case_groups.items():
        print(f"Generating attribution figures for {case_type}: {len(case_df)} cases.")
        results_buffer = []
        part_number = 1

        for case_index, (_, row) in enumerate(case_df.iterrows(), start=1):
            tokens, scores, probabilities = get_gradient_input_attributions(
                str(row[prompt_column]),
                tokenizer,
                model,
            )

            results_buffer.append((case_index, tokens, scores, probabilities))

            if len(results_buffer) == CASES_PER_FIGURE:
                save_case_figure(results_buffer, case_type, part_number)
                results_buffer = []
                part_number += 1

        if results_buffer:
            save_case_figure(results_buffer, case_type, part_number)


if __name__ == "__main__":
    main()
