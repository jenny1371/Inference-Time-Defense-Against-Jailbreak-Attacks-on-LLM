"""
Prepare the WildJailbreak dataset for binary harmful prompt classification.

Input files:
    data/train.tsv
    data/eval.tsv

Output files:
    data/train_split.csv
    data/val_split.csv
    data/test_split.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_HARMFUL = 20_000
N_BENIGN = 10_000

DATA_DIR = Path("data")
TRAIN_TSV_PATH = DATA_DIR / "train.tsv"
EVAL_TSV_PATH = DATA_DIR / "eval.tsv"

TRAIN_OUTPUT_PATH = DATA_DIR / "train_split.csv"
VALIDATION_OUTPUT_PATH = DATA_DIR / "val_split.csv"
TEST_OUTPUT_PATH = DATA_DIR / "test_split.csv"


# =============================================================================
# Helper Functions
# =============================================================================

def extract_prompt(row: pd.Series) -> str:
    """Select the correct prompt column based on the sample type."""
    if str(row["data_type"]).startswith("adversarial"):
        adversarial_prompt = row.get("adversarial", "")
        if pd.notna(adversarial_prompt) and str(adversarial_prompt).strip():
            return str(adversarial_prompt)
        return str(row.get("vanilla", ""))

    return str(row.get("vanilla", ""))


def stratified_sample(
    dataframe: pd.DataFrame,
    sample_size: int,
    stratification_column: str,
    seed: int,
) -> pd.DataFrame:
    """Sample rows while approximately preserving the distribution of a column."""
    group_proportions = dataframe[stratification_column].value_counts(normalize=True)
    sampled_groups = []
    allocated_count = 0
    group_names = group_proportions.index.tolist()

    for group_index, group_name in enumerate(group_names):
        if group_index == len(group_names) - 1:
            group_sample_size = sample_size - allocated_count
        else:
            group_sample_size = round(group_proportions[group_name] * sample_size)

        group_df = dataframe[dataframe[stratification_column] == group_name]
        group_sample_size = min(group_sample_size, len(group_df))
        allocated_count += group_sample_size

        sampled_groups.append(
            group_df.sample(group_sample_size, random_state=seed)
        )

    return pd.concat(sampled_groups).sample(frac=1, random_state=seed)


# =============================================================================
# Dataset Preparation
# =============================================================================

def main() -> None:
    """Create train, validation, and test splits."""
    train_raw_df = pd.read_csv(TRAIN_TSV_PATH, sep="\t")
    eval_raw_df = pd.read_csv(EVAL_TSV_PATH, sep="\t")

    combined_df = pd.concat(
        [train_raw_df, eval_raw_df],
        ignore_index=True,
    )

    combined_df["prompt"] = combined_df.apply(extract_prompt, axis=1)
    combined_df["label"] = combined_df["data_type"].apply(
        lambda sample_type: 1 if "harmful" in str(sample_type) else 0
    )

    combined_df = combined_df.dropna(subset=["prompt"])
    combined_df = combined_df[combined_df["prompt"].str.strip() != ""]
    combined_df = combined_df[["prompt", "data_type", "label"]].reset_index(drop=True)

    harmful_df = combined_df[combined_df["label"] == 1]
    benign_df = combined_df[combined_df["label"] == 0]

    harmful_sample_df = stratified_sample(
        harmful_df,
        N_HARMFUL,
        "data_type",
        SEED,
    )

    benign_sample_df = stratified_sample(
        benign_df,
        N_BENIGN,
        "data_type",
        SEED,
    )

    sampled_df = pd.concat(
        [harmful_sample_df, benign_sample_df]
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_df, temporary_df = train_test_split(
        sampled_df,
        test_size=0.30,
        random_state=SEED,
        stratify=sampled_df["data_type"],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_df["data_type"],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    validation_df.to_csv(VALIDATION_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("Dataset preparation complete.")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()
