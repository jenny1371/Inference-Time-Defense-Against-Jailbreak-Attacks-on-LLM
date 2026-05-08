"""
Generate adversarial examples for the BERT classifier using gradient-guided synonym substitution.

Input files:
    data/test_split.csv
    data/train_split.csv
    models/best_bert_classifier.pt

Output files:
    outputs/bert_attack_results.csv
    data/train_split_augmented.csv
"""

from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.corpus import wordnet
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "bert-base-uncased"
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

TRAIN_PATH = DATA_DIR / "train_split.csv"
TEST_PATH = DATA_DIR / "test_split.csv"
CHECKPOINT_PATH = MODEL_DIR / "best_bert_classifier.pt"
ATTACK_RESULTS_PATH = OUTPUT_DIR / "bert_attack_results.csv"
AUGMENTED_TRAIN_PATH = DATA_DIR / "train_split_augmented.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH = 256
CLASSIFICATION_THRESHOLD = 0.5
TARGET_CLASS = 1
TOP_K_TOKENS = 3
MAX_SYNONYMS = 5

CLASSIFICATION_PREFIX = (
    "Classify whether the following user request, if fulfilled, would cause harm: "
)
EXCLUDED_TOKENS = {"[CLS]", "[SEP]", "[PAD]", ":", ",", ".", "?", "!"}


# =============================================================================
# Model Utilities
# =============================================================================

def load_model() -> tuple[BertTokenizer, BertForSequenceClassification]:
    """Load tokenizer and fine-tuned BERT model."""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_probabilities(
    prompt_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> torch.Tensor:
    """Return class probabilities for one prompt."""
    encoded_input = tokenizer(
        CLASSIFICATION_PREFIX + prompt_text,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            input_ids=encoded_input["input_ids"].to(DEVICE),
            attention_mask=encoded_input["attention_mask"].to(DEVICE),
        ).logits

    return torch.softmax(logits, dim=1)[0]


# =============================================================================
# Gradient-Guided Attack
# =============================================================================

def get_gradient_scores(
    prompt_text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> tuple[list[str], list[float], torch.Tensor]:
    """Compute token-level gradient scores for the target class."""
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
    scores = (gradients * embeddings).sum(dim=-1).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    mask = attention_mask[0].cpu().numpy()
    tokens = [token for token, mask_value in zip(tokens, mask) if mask_value == 1]
    scores = scores[: len(tokens)]

    probabilities = predict_probabilities(prompt_text, tokenizer, model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tokens, scores, probabilities


def get_synonyms(word: str) -> list[str]:
    """Return a short list of WordNet synonyms."""
    synonyms = set()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word.lower():
                synonyms.add(synonym)

    return list(synonyms)[:MAX_SYNONYMS]


def gradient_guided_synonym_attack(
    prompt_text: str,
    true_label: int,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
) -> list[dict]:
    """Try synonym substitutions on high-attribution tokens."""
    tokens, scores, probabilities = get_gradient_scores(prompt_text, tokenizer, model)
    original_prediction = 1 if probabilities[1] >= CLASSIFICATION_THRESHOLD else 0

    valid_token_indices = [
        index
        for index, token in enumerate(tokens)
        if not token.startswith("##")
        and token not in EXCLUDED_TOKENS
        and index < len(scores)
    ]

    top_token_indices = sorted(
        valid_token_indices,
        key=lambda index: abs(scores[index]),
        reverse=True,
    )[:TOP_K_TOKENS]

    successful_attacks = []

    for token_index in top_token_indices:
        target_token = tokens[token_index]
        synonyms = get_synonyms(target_token)

        for synonym in synonyms:
            perturbed_prompt = prompt_text.lower().replace(target_token, synonym, 1)

            if perturbed_prompt == prompt_text.lower():
                continue

            new_probabilities = predict_probabilities(perturbed_prompt, tokenizer, model)
            new_prediction = 1 if new_probabilities[1] >= CLASSIFICATION_THRESHOLD else 0

            if new_prediction != original_prediction:
                successful_attacks.append(
                    {
                        "original_prompt": prompt_text,
                        "perturbed_prompt": perturbed_prompt,
                        "original_token": target_token,
                        "replacement_token": synonym,
                        "original_predicted_label": original_prediction,
                        "new_predicted_label": new_prediction,
                        "label": true_label,
                        "data_type": "adversarial_augmented",
                    }
                )
                break

    return successful_attacks


# =============================================================================
# Attack Runner
# =============================================================================

def main() -> None:
    """Generate adversarial examples and append them to the training data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nltk.download("wordnet", quiet=True)

    tokenizer, model = load_model()
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["prompt"])
    adversarial_benign_df = test_df[test_df["data_type"] == "adversarial_benign"]

    all_attacks = []
    successful_prompt_count = 0

    print(f"Running BERT attack on {len(adversarial_benign_df)} adversarial benign samples.")

    for sample_number, (_, row) in enumerate(adversarial_benign_df.iterrows(), start=1):
        if sample_number % 50 == 0:
            print(f"Processed {sample_number}/{len(adversarial_benign_df)} samples.")

        attacks = gradient_guided_synonym_attack(
            str(row["prompt"]),
            int(row["label"]),
            tokenizer,
            model,
        )

        if attacks:
            successful_prompt_count += 1
            all_attacks.extend(attacks)

    if not all_attacks:
        print("No successful BERT adversarial examples were generated.")
        return

    attack_df = pd.DataFrame(all_attacks)
    attack_df.to_csv(ATTACK_RESULTS_PATH, index=False)

    new_samples_df = attack_df[["perturbed_prompt", "label", "data_type"]].rename(
        columns={"perturbed_prompt": "prompt"}
    )

    train_df = pd.read_csv(TRAIN_PATH)
    augmented_train_df = pd.concat([train_df, new_samples_df], ignore_index=True)
    augmented_train_df.to_csv(AUGMENTED_TRAIN_PATH, index=False)

    success_rate = successful_prompt_count / len(adversarial_benign_df)
    print(f"Attack success rate: {successful_prompt_count}/{len(adversarial_benign_df)} = {success_rate:.3f}")
    print(f"Saved attack results: {ATTACK_RESULTS_PATH}")
    print(f"Saved augmented training data: {AUGMENTED_TRAIN_PATH}")


if __name__ == "__main__":
    main()
