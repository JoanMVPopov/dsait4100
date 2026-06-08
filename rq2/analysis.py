import json
from collections import Counter
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, roc_curve, \
    precision_recall_curve

from data.hatexplain.preprocess_hatexplain import VALID_LABEL_VALUES, calculate_shannon_entropy, AgreementBin
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)

def calculate_variation_ratio(label_counts):
    if not label_counts:
        return 0.0
    total_votes = sum(label_counts.values())
    max_votes = max(label_counts.values())
    return round(1.0 - (max_votes / total_votes), 2)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # HUMAN_DATA_PATH = PROJECT_ROOT / "data" / "hatexplain" / "hatexplain_sampled_1500.csv"
    HUMAN_DATA_PATH = PROJECT_ROOT / "data" / "measuringhatespeech" / "hatexplain_sampled_1500.csv"
    # RESULTS_PATH = PROJECT_ROOT / "data" / "hatexplain" / "results_ollama_runs.jsonl"
    RESULTS_PATH = PROJECT_ROOT / "data" / "measuringhatespeech" / "results_ollama_runs_mhs.jsonl"

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

    # Calculate Variation Ratio and the Binary Instability flag
    llm_entropy_df["llm_vr"] = llm_entropy_df["llm_label_counts"].apply(
        calculate_variation_ratio
    )
    llm_entropy_df["is_unstable"] = (llm_entropy_df["llm_vr"] > 0).astype(int)

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

    print("\nBaseline dataframe ready for analysis:")
    print(baseline_df[["query_id", "model", "stability_bins", "llm_entropy", "llm_vr", "is_unstable"]].head())

    #####################################################################
    # STEP 2: PREDICTIVE ML ANALYSIS (LLM Instability as a Subjectivity Detector)
    #####################################################################

    print("\n" + "=" * 60)
    print("STEP 2: PREDICTIVE ML ANALYSIS (LLM Instability as Proxy)")
    print("=" * 60)

    # Define our Ground Truth Target (y_true):
    # 1 = Humans disagreed (Moderate or High Disagreement)
    # 0 = Humans were unanimous (High Agreement)
    baseline_df["target_human_disagreement"] = (
            baseline_df["stability_bins"] != AgreementBin.HIGH_AGREEMENT
    ).astype(int)

    for model_name, model_df in baseline_df.groupby("model"):
        # ground truth
        y_true = model_df["target_human_disagreement"]

        # the "Confidence Score" for AUC metrics (Variation Ratio)
        y_score = model_df["llm_vr"]

        # the Hard Prediction (If LLM flutters at all -> Predict 1)
        y_pred = model_df["is_unstable"]

        # --- Calculate Metrics ---
        # ROC-AUC measures how well VR ranks texts from least to most subjective
        roc_auc = roc_auc_score(y_true, y_score)

        # PR-AUC is often better than ROC when classes might be imbalanced
        pr_auc = average_precision_score(y_true, y_score)

        # Precision/Recall using the strict "VR > 0" threshold
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        print(f"\n--- Model: {model_name} ---")
        print(f"ROC-AUC: {roc_auc:.3f} (Random Baseline = 0.500)")
        print(f"PR-AUC:  {pr_auc:.3f}")
        print(f"Threshold: 'is_unstable == 1' (Any deviation across 5 seeds)")
        print(f"  -> Precision: {precision:.3f}  | (If LLM fluttered, how often was it actually subjective?)")
        print(f"  -> Recall:    {recall:.3f}  | (Out of all subjective texts, how many did the LLM catch?)")

    #####################################################################
    # STEP 2b: VISUALIZING THE PREDICTIVE POWER (ROC & PR CURVES)
    #####################################################################

    # Set up a 1x2 figure (ROC on the left, PR on the right)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate baseline for PR Curve (proportion of positive class)
    no_skill_pr = baseline_df["target_human_disagreement"].mean()

    for model_name, model_df in baseline_df.groupby("model"):
        y_true = model_df["target_human_disagreement"]
        y_score = model_df["llm_vr"]

        # --- Plot ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

        # --- Plot PR Curve ---
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax_pr.plot(recall_vals, precision_vals, lw=2, label=f"{model_name} (AUC = {pr_auc:.2f})")

    # --- Formatting ROC ---
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (Unanimous text incorrectly flagged)')
    ax_roc.set_ylabel('True Positive Rate (Subjective text correctly flagged)')
    ax_roc.set_title('ROC Curve: Predicting Human Disagreement')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # --- Formatting PR ---
    ax_pr.plot([0, 1], [no_skill_pr, no_skill_pr], color='navy', lw=2, linestyle='--',
               label=f"Random Guess ({no_skill_pr:.2f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (Out of all subjective texts, how many found?)')
    ax_pr.set_ylabel('Precision (If flagged, how likely is it truly subjective?)')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, alpha=0.3)

    plt.suptitle("LLM Instability (Variation Ratio) as a Detector for Human Disagreement", fontsize=14)
    plt.tight_layout()
    plt.show()

    #####################################################################
    # STEP 3: STATISTICAL PROOFS (Ordinal Trend & Bootstrapped Correlation)
    #####################################################################

    print("\n" + "=" * 60)
    print("STEP 3: STATISTICAL PROOFS (Spearman Trend & Bootstrapped Correlation)")
    print("=" * 60)

    # Number of bootstrap iterations for Confidence Intervals
    N_BOOTSTRAPS = 1000

    # Map the categorical bins to ordinal integers for the trend test
    bin_mapping = {
        AgreementBin.HIGH_AGREEMENT: 0,
        AgreementBin.MODERATE_DISAGREEMENT: 1,
        AgreementBin.HIGH_DISAGREEMENT: 2
    }

    for model_name, model_df in baseline_df.groupby("model"):
        print(f"\n--- Model: {model_name} ---")

        # ---------------------------------------------------------
        # A. Spearman Trend Test (Ordinal Bins vs LLM Entropy)
        # ---------------------------------------------------------
        # Convert categorical bins to [0, 1, 2]
        ordinal_bins = model_df["stability_bins"].map(bin_mapping).values
        llm_ent = model_df["llm_entropy"].values

        trend_rho, trend_p = spearmanr(ordinal_bins, llm_ent)

        print("1. Ordinal Trend Test (Across 3 Human Bins):")
        print(f"   Spearman rho: {trend_rho:.3f}")
        print(f"   p-value:      {trend_p:.4e}")
        if trend_p < 0.05 and trend_rho > 0:
            print("   -> Significant upward trend: LLM entropy increases as Human disagreement increases.")

        # ---------------------------------------------------------
        # B. Bootstrapped Spearman Correlation (Continuous vs Continuous)
        # ---------------------------------------------------------
        human_ent = model_df["shannon_entropy"].values

        # Base Spearman correlation
        base_rho, base_p = spearmanr(human_ent, llm_ent)

        # Bootstrapping loop to handle the "5-seed tied data" problem
        boot_rhos = []
        for _ in range(N_BOOTSTRAPS):
            # Sample indices with replacement
            idx = np.random.randint(0, len(model_df), len(model_df))
            rho, _ = spearmanr(human_ent[idx], llm_ent[idx])
            boot_rhos.append(rho)

        # 95% Confidence Interval
        ci_lower = np.percentile(boot_rhos, 2.5)
        ci_upper = np.percentile(boot_rhos, 97.5)

        print("\n2. Bootstrapped Spearman Correlation (Human Entropy vs LLM Entropy):")
        print(f"   Spearman rho: {base_rho:.3f}")
        print(f"   95% CI:       [{ci_lower:.3f}, {ci_upper:.3f}]")

        if ci_lower > 0:
            print("   -> Robust positive correlation (CI does not cross zero).")
        else:
            print("   -> Correlation is NOT statistically robust.")

    #####################################################################
    # STEP 4: VISUAL PROOF & QUALITATIVE MISMATCH EXTRACTION
    #####################################################################

    print("\n" + "=" * 60)
    print("STEP 4: QUALITATIVE MISMATCH EXTRACTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # A. The Stacked Bar Chart (Visualizing the Trend)
    # ---------------------------------------------------------
    # We will plot this for the first model in your list as an example,
    # but you can loop this for all models if you want.

    for model_name, model_df in baseline_df.groupby("model"):
        # Create a cross-tabulation of Human Bins vs LLM Variation Ratios
        crosstab = pd.crosstab(
            model_df["stability_bins"],
            model_df["llm_vr"],
            normalize='index'  # This turns counts into percentages (0.0 to 1.0)
        ) * 100

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # A nice color palette: Green for Stable (0.0), shifting to Reds for highly unstable
        cmap = sns.color_palette("RdYlGn_r", n_colors=len(crosstab.columns))

        crosstab.plot(kind='bar', stacked=True, color=cmap, ax=ax, edgecolor='black')

        ax.set_title(f"LLM Output Stability across Human Disagreement ({model_name})", fontsize=14)
        ax.set_xlabel("Human Agreement Bin", fontsize=12)
        ax.set_ylabel("Percentage of Queries (%)", fontsize=12)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        # Clean up the legend
        plt.legend(title="LLM Variation Ratio\n(0.0 = Unanimous)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # ---------------------------------------------------------
        # B. Qualitative Analysis: Finding the Mismatches
        # ---------------------------------------------------------
        print(f"\n--- Mismatches for {model_name} ---")

        # Mismatch Type 1: Overconfident Model
        # Humans had HIGH disagreement, but the model was UNANIMOUS (VR = 0)
        overconfident = model_df[
            (model_df["stability_bins"] == AgreementBin.HIGH_DISAGREEMENT) &
            (model_df["llm_vr"] == 0.0)
            ]

        # Mismatch Type 2: Overly Sensitive/Noisy Model
        # Humans had HIGH agreement, but the model FLUTTERED (VR > 0)
        noisy = model_df[
            (model_df["stability_bins"] == AgreementBin.HIGH_AGREEMENT) &
            (model_df["llm_vr"] > 0.0)
            ]

        print(f"1. Overconfident LLM (High Human Disag, Stable LLM): {len(overconfident)} queries found.")
        if len(overconfident) > 0:
            sample_queries = overconfident["query_id"].head(3).tolist()
            print(f"   Look up these query_ids to analyze: {sample_queries}")

        print(f"2. Noisy LLM (High Human Ag, Unstable LLM): {len(noisy)} queries found.")
        if len(noisy) > 0:
            sample_queries = noisy["query_id"].head(3).tolist()
            print(f"   Look up these query_ids to analyze: {sample_queries}")
