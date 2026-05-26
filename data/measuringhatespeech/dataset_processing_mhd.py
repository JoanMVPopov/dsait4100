import datasets
import pandas as pd
import numpy as np
import json
from math import log2


dataset = datasets.load_dataset(
    "ucberkeley-dlab/measuring-hate-speech",
    "default"
)

df = dataset["train"].to_pandas()

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nHatespeech column overview:")
print(df["hatespeech"].describe())
print("\nHatespeech value counts:")
print(df["hatespeech"].value_counts(dropna=False).sort_index())

print("\nUnique comments:", df["comment_id"].nunique())
print("Unique annotators:", df["annotator_id"].nunique())
print("Rows per comment:")
print(df.groupby("comment_id").size().describe())



def shannon_entropy(labels):
    labels = pd.Series(labels).dropna()
    counts = labels.value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def variation_ratio(labels):
    labels = pd.Series(labels).dropna()
    counts = labels.value_counts()
    return float(1 - counts.max() / counts.sum())


def to_json_list(x):
    return json.dumps(list(x), ensure_ascii=False)


def add_entropy_bins(data, entropy_col="entropy"):
    q33 = data[entropy_col].quantile(0.33)
    q66 = data[entropy_col].quantile(0.66)

    print(f"\nEntropy tertile thresholds:")
    print(f"33rd percentile: {q33:.4f}")
    print(f"66th percentile: {q66:.4f}")

    def assign_bin(x):
        if x == 0:
            return "low_disagreement"
        elif x < q66:
            return "medium_disagreement"
        else:
            return "high_disagreement"

    data["disagreement_bin"] = data[entropy_col].apply(assign_bin)
    return data



global_df = (
    df.groupby("comment_id")
    .agg(
        text=("text", "first"),
        n_annotators=("annotator_id", "nunique"),
        annotator_ids=("annotator_id", list),
        hatespeech_labels=("hatespeech", list),
        entropy=("hatespeech", shannon_entropy),
        variation_ratio=("hatespeech", variation_ratio),
        mean_hatespeech=("hatespeech", "mean"),
        median_hatespeech=("hatespeech", "median"),
        std_hatespeech=("hatespeech", "std"),
    )
    .reset_index()
)

global_df = global_df[global_df["n_annotators"] >= 3].copy()
global_df = add_entropy_bins(global_df, "entropy")

global_df["annotator_ids"] = global_df["annotator_ids"].apply(to_json_list)
global_df["hatespeech_labels"] = global_df["hatespeech_labels"].apply(to_json_list)

global_df = global_df[
    [
        "comment_id",
        "text",
        "disagreement_bin",
        "n_annotators",
        "entropy",
        "variation_ratio",
        "mean_hatespeech",
        "median_hatespeech",
        "std_hatespeech",
        "annotator_ids",
        "hatespeech_labels",
    ]
]

print("\nGlobal bin counts:")
print(global_df["disagreement_bin"].value_counts())

global_df.to_csv("mhs_global_entropy_bins_fixed.csv", index=False)
print("\nSaved: mhs_global_entropy_bins_fixed.csv")



df["ideology_group"] = None

conservative_cols = [
    "annotator_ideology_extremeley_conservative",
    "annotator_ideology_conservative",
    "annotator_ideology_slightly_conservative",
]

liberal_cols = [
    "annotator_ideology_slightly_liberal",
    "annotator_ideology_liberal",
    "annotator_ideology_extremeley_liberal",
]

neutral_cols = [
    "annotator_ideology_neutral"
]

df.loc[df[conservative_cols].eq(1).any(axis=1), "ideology_group"] = "conservative"
df.loc[df[liberal_cols].eq(1).any(axis=1), "ideology_group"] = "liberal"
df.loc[df[neutral_cols].eq(1).any(axis=1), "ideology_group"] = "neutral"

df_ideo = df[df["ideology_group"].isin(["conservative", "liberal", "neutral"])].copy()

print("\nIdeology group annotation counts:")
print(df_ideo["ideology_group"].value_counts())

print("\nIdeology group comment coverage:")
print(df_ideo.groupby("ideology_group")["comment_id"].nunique())



persona_df = (
    df_ideo.groupby(["comment_id", "ideology_group"])
    .agg(
        text=("text", "first"),
        n_group_annotators=("annotator_id", "nunique"),
        annotator_ids=("annotator_id", list),
        hatespeech_labels=("hatespeech", list),
        entropy=("hatespeech", shannon_entropy),
        variation_ratio=("hatespeech", variation_ratio),
        mean_hatespeech=("hatespeech", "mean"),
        median_hatespeech=("hatespeech", "median"),
        std_hatespeech=("hatespeech", "std"),
    )
    .reset_index()
)

persona_df = persona_df[persona_df["n_group_annotators"] >= 3].copy()

persona_parts = []
for group_name, group_data in persona_df.groupby("ideology_group"):
    print(f"\nPersona group: {group_name}")
    group_data = add_entropy_bins(group_data.copy(), "entropy")
    persona_parts.append(group_data)

persona_df = pd.concat(persona_parts, ignore_index=True)

persona_df["annotator_ids"] = persona_df["annotator_ids"].apply(to_json_list)
persona_df["hatespeech_labels"] = persona_df["hatespeech_labels"].apply(to_json_list)

persona_df = persona_df[
    [
        "comment_id",
        "text",
        "ideology_group",
        "disagreement_bin",
        "n_group_annotators",
        "entropy",
        "variation_ratio",
        "mean_hatespeech",
        "median_hatespeech",
        "std_hatespeech",
        "annotator_ids",
        "hatespeech_labels",
    ]
]

print("\nPersona bin counts:")
print(
    persona_df
    .groupby(["ideology_group", "disagreement_bin"])
    .size()
    .unstack(fill_value=0)
)

persona_df.to_csv("mhs_persona_entropy_bins.csv", index=False)
print("\nSaved: mhs_persona_entropy_bins.csv")



N_PER_BIN = 1000
RANDOM_STATE = 42

global_min = np.min(global_df["disagreement_bin"].value_counts())
print("This is the minimum in bins")
print(global_min)

global_sample = (
    global_df
    .groupby("disagreement_bin", group_keys=False)
    .apply(lambda x: x.sample(max(global_min, N_PER_BIN), random_state=RANDOM_STATE))
)

persona_sample = (
    persona_df
    .groupby(["ideology_group", "disagreement_bin"], group_keys=False)
    .apply(lambda x: x.sample(min(len(x), N_PER_BIN), random_state=RANDOM_STATE))
)

global_sample.to_csv("mhs_global_entropy_bins_fixed_sampled.csv", index=False)
persona_sample.to_csv("mhs_persona_entropy_bins_fixed_sampled.csv", index=False)

print("\nSaved sampled files:")
print("mhs_global_entropy_bins_fixed_sampled.csv")
print("mhs_persona_entropy_bins_sampled.csv")

print("\nSampled global bin counts:")
print(global_sample["disagreement_bin"].value_counts())

print("\nSampled persona bin counts:")
print(
    persona_sample
    .groupby(["ideology_group", "disagreement_bin"])
    .size()
    .unstack(fill_value=0)
)