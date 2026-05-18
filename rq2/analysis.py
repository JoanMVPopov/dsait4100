import json
from collections import Counter
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, kruskal

from data.hatexplain.preprocess_hatexplain import VALID_LABEL_VALUES, calculate_shannon_entropy, AgreementBin


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    HUMAN_DATA_PATH = PROJECT_ROOT / "data" / "hatexplain" / "hatexplain_sampled_30.csv"
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
    # RQ1 and RQ2 use only the raw baseline condition

    baseline_df = analysis_df[
        analysis_df["persona"] == "no_persona"
        ].copy()

    print("\nBaseline dataframe:")
    print(baseline_df.shape)

    print("\nBaseline rows per model:")
    print(baseline_df["model"].value_counts())

    print("\nBaseline rows per human bin:")
    print(baseline_df["stability_bins"].value_counts())

    ###########################################################

    for model_name, model_df in baseline_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 5))

        high_agreement = model_df[
            model_df["stability_bins"] == AgreementBin.HIGH_AGREEMENT
            ]["llm_entropy"]

        high_disagreement = model_df[
            model_df["stability_bins"] == AgreementBin.HIGH_DISAGREEMENT
            ]["llm_entropy"]

        ax.hist(high_agreement, alpha=0.6, label="high agreement")
        ax.hist(high_disagreement, alpha=0.6, label="high disagreement")

        ax.set_title(f"LLM entropy distribution: {model_name}")
        ax.set_xlabel("LLM entropy across seeds")
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        plt.show()

    ########################
    # RQ2: Do high-disagreement samples produce more variable LLM outputs?

    print("\nRQ2: Method selection based on normality")

    for model_name, model_df in baseline_df.groupby("model"):
        high_agreement = model_df[
            model_df["stability_bins"] == AgreementBin.HIGH_AGREEMENT
            ]["llm_entropy"]

        high_disagreement = model_df[
            model_df["stability_bins"] == AgreementBin.HIGH_DISAGREEMENT
            ]["llm_entropy"]

        print("\nModel:", model_name)
        print("N high agreement:", len(high_agreement))
        print("N high disagreement:", len(high_disagreement))

        print("Mean LLM entropy, high agreement:", high_agreement.mean())
        print("Mean LLM entropy, high disagreement:", high_disagreement.mean())
        print("Median LLM entropy, high agreement:", high_agreement.median())
        print("Median LLM entropy, high disagreement:", high_disagreement.median())

        # Shapiro-Wilk normality tests
        shapiro_high_agreement = shapiro(high_agreement)
        shapiro_high_disagreement = shapiro(high_disagreement)

        print("Shapiro high agreement p-value:", shapiro_high_agreement.pvalue)
        print("Shapiro high disagreement p-value:", shapiro_high_disagreement.pvalue)

        normal_high_agreement = shapiro_high_agreement.pvalue > 0.05
        normal_high_disagreement = shapiro_high_disagreement.pvalue > 0.05

        both_normal = normal_high_agreement and normal_high_disagreement

        if both_normal:
            levene_result = levene(high_agreement, high_disagreement)
            equal_variance = levene_result.pvalue > 0.05

            t_stat, p_value = ttest_ind(
                high_agreement,
                high_disagreement,
                equal_var=equal_variance,
            )

            print("Levene p-value:", levene_result.pvalue)
            print("Selected test:", "independent t-test" if equal_variance else "Welch t-test")
            print("t statistic:", t_stat)
            print("p-value:", p_value)

        else:
            u_stat, p_value = mannwhitneyu(
                high_agreement,
                high_disagreement,
                alternative="two-sided",
            )

            print("Selected test: Mann-Whitney U")
            print("Mann-Whitney U:", u_stat)
            print("p-value:", p_value)

    ########################
    # TODO Optional: Kruskal-Wallis across all three human agreement bins
    # do we need?
    # if so, you need to account for possible normality
    # and use anova if suitable

    print("\nOptional: Kruskal-Wallis across all three bins")

    for model_name, model_df in baseline_df.groupby("model"):
        groups = [
            group["llm_entropy"]
            for _, group in model_df.groupby("stability_bins")
        ]

        stat, p_value = kruskal(*groups)

        print("\nModel:", model_name)
        print("Kruskal-Wallis H:", stat)
        print("p-value:", p_value)




