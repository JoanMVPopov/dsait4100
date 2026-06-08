import json
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from data.hatexplain.preprocess_hatexplain import VALID_LABEL_VALUES, calculate_shannon_entropy, AgreementBin
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    HUMAN_DATA_PATH = PROJECT_ROOT / "data" / "hatexplain" / "hatexplain_sampled_1500.csv"
    RESULTS_PATH = PROJECT_ROOT / "data" / "hatexplain" / "results_ollama_runs.jsonl"

    human_df = pd.read_csv(HUMAN_DATA_PATH)
    results_df = load_jsonl(RESULTS_PATH)

    print("Human metadata:")
    print(human_df.shape)
    print(human_df.columns)
    print(human_df.head())

    print("\nModel results:")
    print(results_df.shape)
    print(results_df.columns)
    print(results_df.head())

    human_df = human_df.rename(columns={"post_id": "query_id"})

    human_df["query_id"] = human_df["query_id"].astype(str)
    results_df["query_id"] = results_df["query_id"].astype(str)

    print(human_df[["query_id", "shannon_entropy", "stability_bins"]].head())
    print(results_df[["query_id", "model", "persona", "seed", "raw_output"]].head())

    result_ids = set(results_df["query_id"])
    human_ids = set(human_df["query_id"])

    print("Result IDs:", len(result_ids))
    print("Human IDs:", len(human_ids))
    print("Matching IDs:", len(result_ids & human_ids))
    print("Result IDs missing from human metadata:", len(result_ids - human_ids))
    print("Human IDs missing from results:", len(human_ids - result_ids))

    ######################

    results_df["raw_output_clean"] = results_df["raw_output"].astype(str).str.lower().str.strip()

    # ~ is for negation
    invalid_mask = ~results_df["raw_output_clean"].isin(VALID_LABEL_VALUES)
    invalid_count = invalid_mask.sum()

    print("\nInvalid label count:")
    print(invalid_count)

    print("\nInvalid label rate:")
    print(invalid_count / len(results_df))

    if invalid_count > 0:
        print("\nInvalid label examples:")
        print(results_df.loc[invalid_mask, ["query_id", "model", "persona", "seed", "raw_output"]].head(20))

    ########################

    valid_results_df = results_df[~invalid_mask].copy()

    # collapse the seed outputs into one row per query x model x persona
    llm_entropy_df = (
        valid_results_df
        .groupby(["query_id", "model", "persona"])
        .agg(
            seed_outputs=("raw_output_clean", list),
            n_valid_seeds=("raw_output_clean", "count"),
        )
        .reset_index()
    )

    llm_entropy_df["llm_label_counts"] = llm_entropy_df["seed_outputs"].apply(
        lambda labels: dict(Counter(labels))
    )

    llm_entropy_df["llm_entropy"] = llm_entropy_df["llm_label_counts"].apply(
        calculate_shannon_entropy
    )

    print("\nLLM entropy dataframe:")
    print(llm_entropy_df.head())

    print("\nValid seed counts:")
    print(llm_entropy_df["n_valid_seeds"].value_counts())

    print("\nLLM entropy summary:")
    print(llm_entropy_df["llm_entropy"].describe())

    print("\nExample seed outputs:")
    print(llm_entropy_df[["query_id", "model", "persona", "seed_outputs", "llm_label_counts", "llm_entropy"]].head(10))

    ########################
    # Keep only complete 5-seed groups

    print("\nRows before complete-seed filtering:")
    print(llm_entropy_df.shape)

    llm_entropy_df = llm_entropy_df[
        llm_entropy_df["n_valid_seeds"] == 5
        ].copy()

    print("\nRows after complete-seed filtering:")
    print(llm_entropy_df.shape)

    print("\nValid seed counts after filtering:")
    print(llm_entropy_df["n_valid_seeds"].value_counts())

    ########################
    # Merge LLM entropy with human disagreement metadata

    analysis_df = llm_entropy_df.merge(
        human_df[
            [
                "query_id",
                "shannon_entropy",
                "stability_bins",
                "annotator_labels",
                "label_counts",
            ]
        ],
        on="query_id",
        how="left",
    )

    print("\nAnalysis dataframe:")
    print(analysis_df.head())
    print(analysis_df.shape)

    print("\nMissing human entropy values:")
    print(analysis_df["shannon_entropy"].isna().sum())

    ########################
    # RQ1 uses only the raw baseline condition

    baseline_df = analysis_df[
        analysis_df["persona"] == "no_persona"
        ].copy()

    print("\nBaseline dataframe:")
    print(baseline_df.shape)

    pd.set_option('display.max_columns', None)
    print(baseline_df.head())

    print("\nBaseline rows per model:")
    print(baseline_df["model"].value_counts())

    print("\nBaseline rows per human bin:")
    print(baseline_df["stability_bins"].value_counts())

    ########################
    # RQ1: Correlation between human disagreement and LLM variability

    print("\nRQ1: Spearman correlation between human entropy and LLM entropy")

    # Intra-model variability
    # group by model, look at the entropy (derived from grouping by seed earlier) for each query
    for model_name, model_df in baseline_df.groupby("model"):
        rho, p_value = spearmanr(
            model_df["shannon_entropy"],
            model_df["llm_entropy"],
        )

        print("\nModel:", model_name)
        print("N:", len(model_df))
        print("Spearman rho:", rho)
        print("p-value:", p_value)

    ##################################################################
    # BOXPLOT 1
    # TODO: This might be optional, determine
    # Overall, do higher human-disagreement bins show higher LLM variability?
    # x-axis: high_agreement, moderate_disagreement, high_disagreement
    # y-axis: LLM entropy across the 5 seeds
    # data: all models pooled together

    bin_order = [
        AgreementBin.HIGH_AGREEMENT,
        AgreementBin.MODERATE_DISAGREEMENT,
        AgreementBin.HIGH_DISAGREEMENT
    ]

    baseline_df["stability_bins"] = pd.Categorical(
        baseline_df["stability_bins"],
        categories=bin_order,
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    baseline_df.boxplot(
        column="llm_entropy",
        by="stability_bins",
        ax=ax,
        grid=False,
    )

    ax.set_title("LLM output variability by human agreement bin")
    ax.set_xlabel("Human agreement bin")
    ax.set_ylabel("LLM entropy across seeds")

    plt.suptitle("")
    plt.tight_layout()
    plt.show()

    #####################################################################
    # BOXPLOT 2
    # Does this pattern differ between Qwen, Llama, and DeepSeek-R1?

    fig, axes = plt.subplots(
        1,
        len(baseline_df["model"].unique()),
        figsize=(15, 5),
        sharey=True,
    )

    for ax, (model_name, model_df) in zip(axes, baseline_df.groupby("model")):
        model_df.boxplot(
            column="llm_entropy",
            by="stability_bins",
            ax=ax,
            grid=False,
        )

        ax.set_title(model_name)
        ax.set_xlabel("Human agreement bin")
        ax.set_ylabel("LLM entropy across seeds")

    plt.suptitle("")
    plt.tight_layout()
    plt.show()

    #################################

    # TODO: OPTIONAL, determine if needed
    # Scatter plot of human entropy vs LLM entropy, per model

    fig, axes = plt.subplots(
        1,
        len(baseline_df["model"].unique()),
        figsize=(15, 5),
        sharey=True,
    )

    for ax, (model_name, model_df) in zip(axes, baseline_df.groupby("model")):
        ax.scatter(
            model_df["shannon_entropy"],
            model_df["llm_entropy"],
            alpha=0.7,
        )

        ax.set_title(model_name)
        ax.set_xlabel("Human entropy")
        ax.set_ylabel("LLM entropy across seeds")

    plt.tight_layout()
    plt.show()

