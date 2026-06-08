import math
from collections import Counter
from enum import StrEnum
import pandas as pd
import os

class ValidLabels(StrEnum):
    HATESPEECH = "hatespeech"
    OFFENSIVE = "offensive"
    NORMAL = "normal"

class AgreementBin(StrEnum):
    HIGH_AGREEMENT = "high_agreement"
    MODERATE_DISAGREEMENT = "moderate_disagreement"
    HIGH_DISAGREEMENT = "high_disagreement"


TOTAL_ANNOTATORS = 3
VALID_LABEL_VALUES = {label.value for label in ValidLabels}
REQUIRED_BINS = {bin.value for bin in AgreementBin}
SAMPLES_PER_BIN = 500
RANDOM_SEED = 42

def tokens_to_text(tokens):
    return " ".join(tokens)


def extract_valid_labels_or_none(annotators):
    labels = [ann["label"].lower().strip() for ann in annotators]

    # total annotators per query should be 3, skip the row otherwise
    if len(labels) != TOTAL_ANNOTATORS:
        return None

    # if at least 1 label is not in the predefined, skip the row
    if not all(label in VALID_LABEL_VALUES for label in labels):
        return None

    return labels

def calculate_shannon_entropy(label_counts, annotators_count = None):
    if annotators_count is None:
        annotators_count = sum(label_counts.values())

    shannon_entropy = 0.0
    for count in label_counts.values():
        probability = count / annotators_count
        shannon_entropy -= probability * math.log(probability, 2)

    return shannon_entropy

def assign_to_bin(label_counts):
    most_frequent = sorted(label_counts.values(), reverse=True)[0]

    if most_frequent == 1:
        return AgreementBin.HIGH_DISAGREEMENT
    elif most_frequent == 2:
        return AgreementBin.MODERATE_DISAGREEMENT
    else:
        return AgreementBin.HIGH_AGREEMENT

if __name__ == "__main__":
    curr_path = os.path.dirname(__file__)
    dataset_path = os.path.join(curr_path, "dataset.json")
    df = pd.read_json(dataset_path)
    df = df.T.reset_index(drop=True)

    df["post_id"] = df["post_id"].astype(str)
    df = df.set_index("post_id")

    df["annotator_labels"] = df["annotators"].apply(extract_valid_labels_or_none)

    # keep only rows with exactly 3 annotators and valid labels
    df = df[df["annotator_labels"].notna()].copy()

    df["text"] = df["post_tokens"].apply(tokens_to_text)
    df["label_counts"] = df["annotator_labels"].apply(lambda labels: dict(Counter(labels)))
    df["n_annotators"] = df["annotator_labels"].apply(len)

    print(df[["text", "annotator_labels", "label_counts", "n_annotators"]].head())
    print(df["n_annotators"].value_counts())
    print(df["annotator_labels"].explode().value_counts())

    # calculate the shannon entropy for each query
    df["shannon_entropy"] = df['label_counts'].apply(lambda label_counts: calculate_shannon_entropy(label_counts, TOTAL_ANNOTATORS))
    print(df["shannon_entropy"].describe())
    print(df["shannon_entropy"].info())

    # define the stability bins, assign the query to a bin
    df["stability_bins"] = df['label_counts'].apply(lambda label_counts: assign_to_bin(label_counts))
    df["stability_bins"] = df["stability_bins"].astype(str)

    print("\nBin counts:")
    print(df["stability_bins"].value_counts())

    print("\nEntropy values:")
    print(df["shannon_entropy"].value_counts().sort_index())

    print("\nCross-check entropy by bin:")
    print(df.groupby("stability_bins")["shannon_entropy"].value_counts().sort_index())

    ##################

    available_bins = set(df["stability_bins"].dropna().unique())

    missing_bins = REQUIRED_BINS - available_bins
    if missing_bins:
        raise ValueError(f"Missing expected bins: {missing_bins}")

    # min(samples_bin_1, samples_bin_2, samples_bin_3)
    # bin_counts = df["stability_bins"].value_counts(ascending=True)
    # SAMPLES_PER_BIN = bin_counts.iloc[0]

    print(f"\nSamples per bin: {SAMPLES_PER_BIN}")
    # print(f"\nBin with lowest number of samples: {bin_counts.index[0]}")

    bin_counts = df["stability_bins"].value_counts()

    for bin_name in REQUIRED_BINS:
        if bin_counts[bin_name] < SAMPLES_PER_BIN:
            raise ValueError(
                f"Not enough rows in {bin_name}. "
                f"Needed {SAMPLES_PER_BIN}, found {bin_counts[bin_name]}."
            )

    sampled_df = (
        df.groupby("stability_bins", group_keys=False)
        .sample(n=SAMPLES_PER_BIN, random_state=RANDOM_SEED)
        .copy()
    )

    print("\nSampled rows per bin:")
    print(sampled_df["stability_bins"].value_counts())

    print("\nSampled dataframe shape:")
    print(sampled_df.shape)

    total_samples = SAMPLES_PER_BIN * len(REQUIRED_BINS)

    output_path = os.path.join(curr_path, f"hatexplain_sampled_{total_samples}.csv")
    sampled_df = sampled_df.reset_index()
    sampled_df.to_csv(output_path, index=False)

    print(f"\nSaved sampled dataset to: {output_path}")

    sampled_df.rename(columns={"post_id": "query_id"}, inplace=True)
    output_path_workflow_ready = os.path.join(curr_path, f"hatexplain_sampled_wflow_ready_{total_samples}.csv")

    # save a column-subset in the format needed for the workflow (but rows stay the same)
    sampled_df[["query_id", "text"]].to_csv(output_path_workflow_ready, index=False)

    print(f"\nSaved workflow-ready sampled dataset to: {output_path_workflow_ready}")
