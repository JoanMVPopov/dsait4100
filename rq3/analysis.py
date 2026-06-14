import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kruskal, mannwhitneyu
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)


VALID_LABEL_VALUES = {"normal", "offensive", "hatespeech"}

PERSONA_ORDER = ["conservative", "liberal", "neutral"]

BIN_ORDER = [
    "low_disagreement",
    "medium_disagreement",
    "high_disagreement",
]

BIN_TO_ORDINAL = {
    "low_disagreement": 0,
    "medium_disagreement": 1,
    "high_disagreement": 2,
}

LABEL_TO_SCORE = {
    "normal": 0,
    "offensive": 1,
    "hatespeech": 2,
}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def normalize_persona(value):
    if pd.isna(value):
        return np.nan

    value = str(value).lower().strip()

    if "conservative" in value:
        return "conservative"
    if "liberal" in value:
        return "liberal"
    if "neutral" in value:
        return "neutral"
    if value in {"no_persona", "none", "baseline"}:
        return "no_persona"

    return value


def clean_label(value):
    """
    Strictly parse model outputs into:
    normal / offensive / hatespeech.

    To match the updated global RQ1/RQ2 analysis, only the first
    whitespace-separated token is accepted as the label. This prevents
    refusals or verbose explanations from being counted as valid labels
    just because they mention one of the target labels later in the text.
    """
    if pd.isna(value):
        return np.nan

    tokens = str(value).lower().strip().split()

    if tokens and tokens[0] in VALID_LABEL_VALUES:
        return tokens[0]

    return np.nan


def shannon_entropy_from_labels(labels):
    labels = pd.Series(labels).dropna()

    if len(labels) == 0:
        return np.nan

    counts = labels.value_counts()
    probs = counts / counts.sum()

    return float(-(probs * np.log2(probs)).sum())


def normalized_entropy_from_labels(labels):
    entropy = shannon_entropy_from_labels(labels)

    if pd.isna(entropy):
        return np.nan

    return float(entropy / np.log2(len(VALID_LABEL_VALUES)))


def variation_ratio_from_labels(labels):
    labels = pd.Series(labels).dropna()

    if len(labels) == 0:
        return np.nan

    counts = labels.value_counts()
    return round(float(1.0 - counts.max() / counts.sum()), 2)


def safe_spearman(x, y):
    x = pd.Series(x)
    y = pd.Series(y)

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan

    rho, p_value = spearmanr(x, y)
    return float(rho), float(p_value)


def bootstrap_spearman(x, y, n_bootstraps=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    mask = x.notna() & y.notna()
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, np.nan

    base_rho, base_p = safe_spearman(x, y)

    boot_rhos = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(x), len(x))
        rho, _ = safe_spearman(x.iloc[idx], y.iloc[idx])

        if not pd.isna(rho):
            boot_rhos.append(rho)

    if len(boot_rhos) == 0:
        return base_rho, np.nan, np.nan

    ci_low = float(np.percentile(boot_rhos, 2.5))
    ci_high = float(np.percentile(boot_rhos, 97.5))

    return base_rho, ci_low, ci_high


def safe_auc_metrics(y_true, y_score, y_pred):
    """
    Computes ROC-AUC, PR-AUC, precision, and recall safely.
    Returns nan if only one class is present.
    """
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)
    y_pred = pd.Series(y_pred)

    mask = y_true.notna() & y_score.notna() & y_pred.notna()

    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)
    y_pred = y_pred[mask].astype(int)

    if len(y_true) == 0 or y_true.nunique() < 2:
        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
        }

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def analyze_mhs_persona_dataset(
    HUMAN_DATA_PATH: Path,
    RESULTS_PATH: Path,
    OUTPUT_PATH: Path,
    required_n_seeds: int = 5,
    n_bootstraps: int = 1000,
):

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH = OUTPUT_PATH / "plots"
    TABLES_PATH = OUTPUT_PATH / "tables"

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.mkdir(parents=True, exist_ok=True)

    human_df = pd.read_csv(HUMAN_DATA_PATH)
    results_df = load_jsonl(RESULTS_PATH)

    print("\n" + "=" * 80)
    print("PERSONA-LEVEL MHS ANALYSIS")
    print("=" * 80)

    print("\nHuman persona metadata:")
    print(human_df.shape)
    print(human_df.columns.tolist())
    print(human_df.head())

    print("\nLLM results:")
    print(results_df.shape)
    print(results_df.columns.tolist())
    print(results_df.head())

    human_df = human_df.rename(
        columns={
            "ideology_group": "human_persona",
            "disagreement_bin": "human_disagreement_bin",
            "entropy": "human_entropy",
            "variation_ratio": "human_variation_ratio",
            "mean_hatespeech": "human_mean_hatespeech",
            "median_hatespeech": "human_median_hatespeech",
            "std_hatespeech": "human_std_hatespeech",
            "hatespeech_labels": "human_labels",
        }
    )

    human_df["query_id"] = human_df["query_id"].astype(str)
    human_df["human_persona"] = human_df["human_persona"].apply(normalize_persona)

    human_df = human_df[
        human_df["human_persona"].isin(PERSONA_ORDER)
    ].copy()

    human_df["human_bin_ordinal"] = human_df["human_disagreement_bin"].map(BIN_TO_ORDINAL)

    # Avoid accidental duplicated human rows.
    before = len(human_df)
    human_df = human_df.drop_duplicates(
        subset=["query_id", "human_persona"],
        keep="first",
    ).copy()
    after = len(human_df)

    print(f"\nDropped duplicate human query/persona rows: {before - after}")

    print("\nHuman rows per persona:")
    print(human_df["human_persona"].value_counts())

    print("\nHuman rows per persona/bin:")
    print(
        human_df
        .groupby(["human_persona", "human_disagreement_bin"])
        .size()
        .unstack(fill_value=0)
    )

    results_df["query_id"] = results_df["query_id"].astype(str)
    results_df["llm_persona"] = results_df["persona"].apply(normalize_persona)
    results_df["label_clean"] = results_df["raw_output"].apply(clean_label)

    invalid_mask = results_df["label_clean"].isna()

    print("\nInvalid LLM label count:", invalid_mask.sum())
    print("Invalid LLM label rate:", invalid_mask.mean())

    if invalid_mask.sum() > 0:
        print("\nInvalid examples:")
        print(
            results_df.loc[
                invalid_mask,
                ["query_id", "model", "persona", "seed", "raw_output"]
            ].head(20)
        )

    # ------------------------------------------------------------
    # Non-compliance audit
    # ------------------------------------------------------------
    # A generation is non-compliant when it fails strict first-token
    # label parsing. We compute this by matched model/persona/bin so
    # that refusals or verbose outputs are visible in the persona setup.
    noncomp_df = results_df.assign(noncompliant=invalid_mask).copy()

    noncomp_matched = noncomp_df.merge(
        human_df[["query_id", "human_persona", "human_disagreement_bin"]],
        left_on=["query_id", "llm_persona"],
        right_on=["query_id", "human_persona"],
        how="inner",
    )

    noncomp_table = (
        noncomp_matched
        .groupby(["model", "llm_persona", "human_disagreement_bin"])["noncompliant"]
        .mean()
        .unstack(fill_value=0.0)
    )

    print("\nPersona non-compliance rate per model/persona/bin:")
    print(noncomp_table.to_string())
    noncomp_table.to_csv(TABLES_PATH / "persona_noncompliance_by_bin.csv")

    print("\nAudit of non-compliant outputs (top distinct patterns):")
    print(
        results_df.loc[invalid_mask, "raw_output"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str[:80]
        .value_counts()
        .head(20)
    )

    valid_results_df = results_df[
        (~invalid_mask) &
        (results_df["llm_persona"].isin(PERSONA_ORDER))
    ].copy()

    dedupe_key = ["query_id", "model", "llm_persona", "seed"]

    duplicate_count = valid_results_df.duplicated(subset=dedupe_key).sum()
    print("\nDuplicate query/model/persona/seed rows:", duplicate_count)

    valid_results_df = (
        valid_results_df
        .sort_values(dedupe_key)
        .drop_duplicates(subset=dedupe_key, keep="first")
        .copy()
    )

    print("\nValid LLM rows after cleaning/deduplication:", valid_results_df.shape)

    llm_df = (
        valid_results_df
        .groupby(["query_id", "model", "llm_persona"])
        .agg(
            seed_outputs=("label_clean", list),
            seeds=("seed", lambda x: sorted(list(x))),
            n_valid_seeds=("label_clean", "count"),
        )
        .reset_index()
    )

    llm_df["llm_label_counts"] = llm_df["seed_outputs"].apply(
        lambda labels: dict(Counter(labels))
    )

    llm_df["llm_entropy"] = llm_df["seed_outputs"].apply(shannon_entropy_from_labels)
    llm_df["llm_entropy_norm"] = llm_df["seed_outputs"].apply(normalized_entropy_from_labels)
    llm_df["llm_variation_ratio"] = llm_df["seed_outputs"].apply(variation_ratio_from_labels)

    llm_df["is_unstable"] = (llm_df["llm_variation_ratio"] > 0).astype(int)

    llm_df["llm_majority_label"] = llm_df["llm_label_counts"].apply(
        lambda d: max(d.items(), key=lambda item: item[1])[0]
    )

    llm_df["llm_majority_share"] = llm_df["llm_label_counts"].apply(
        lambda d: max(d.values()) / sum(d.values())
    )

    llm_df["llm_scores"] = llm_df["seed_outputs"].apply(
        lambda labels: [LABEL_TO_SCORE[label] for label in labels]
    )

    llm_df["llm_mean_severity"] = llm_df["llm_scores"].apply(
        lambda scores: float(np.mean(scores))
    )

    llm_df["llm_std_severity"] = llm_df["llm_scores"].apply(
        lambda scores: float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    )

    print("\nLLM seed count distribution before filtering:")
    print(llm_df["n_valid_seeds"].value_counts().sort_index())

    llm_df = llm_df[
        llm_df["n_valid_seeds"] == required_n_seeds
    ].copy()

    print("\nLLM rows after complete-seed filtering:", llm_df.shape)

    print("\nLLM rows per model/persona:")
    print(
        llm_df
        .groupby(["model", "llm_persona"])
        .size()
        .unstack(fill_value=0)
    )

    analysis_df = llm_df.merge(
        human_df,
        left_on=["query_id", "llm_persona"],
        right_on=["query_id", "human_persona"],
        how="inner",
    )

    print("\nMatched persona analysis dataframe:")
    print(analysis_df.shape)

    print("\nMatched rows per model/persona:")
    print(
        analysis_df
        .groupby(["model", "llm_persona"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nMatched rows per persona/bin:")
    print(
        analysis_df
        .groupby(["llm_persona", "human_disagreement_bin"])
        .size()
        .unstack(fill_value=0)
    )

    analysis_df.to_csv(OUTPUT_PATH / "persona_matched_analysis_full.csv", index=False)
    llm_df.to_csv(OUTPUT_PATH / "persona_llm_entropy_by_query_model_persona.csv", index=False)

    corr_rows = []

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        rho_entropy, p_entropy = safe_spearman(
            model_df["human_entropy"],
            model_df["llm_entropy"],
        )

        rho_vr, p_vr = safe_spearman(
            model_df["human_variation_ratio"],
            model_df["llm_variation_ratio"],
        )

        rho_bin, p_bin = safe_spearman(
            model_df["human_bin_ordinal"],
            model_df["llm_entropy"],
        )

        rho_mean, p_mean = safe_spearman(
            model_df["human_mean_hatespeech"],
            model_df["llm_mean_severity"],
        )

        corr_rows.append(
            {
                "model": model_name,
                "persona": persona_name,
                "n": len(model_df),

                "rho_human_entropy_vs_llm_entropy": rho_entropy,
                "p_entropy": p_entropy,

                "rho_human_vr_vs_llm_vr": rho_vr,
                "p_variation_ratio": p_vr,

                "rho_human_bin_vs_llm_entropy": rho_bin,
                "p_bin_trend": p_bin,

                "rho_human_mean_vs_llm_mean_severity": rho_mean,
                "p_mean_severity": p_mean,

                "mean_human_entropy": model_df["human_entropy"].mean(),
                "mean_llm_entropy": model_df["llm_entropy"].mean(),
                "mean_human_vr": model_df["human_variation_ratio"].mean(),
                "mean_llm_vr": model_df["llm_variation_ratio"].mean(),
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values(["model", "persona"])

    print("\n" + "=" * 80)
    print("MATCHED PERSONA CORRELATIONS")
    print("=" * 80)
    print(corr_df.round(4))

    corr_df.to_csv(TABLES_PATH / "persona_matched_correlations.csv", index=False)
    corr_df.to_latex(
        TABLES_PATH / "persona_matched_correlations.tex",
        index=False,
        float_format="%.3f",
    )

    boot_rows = []

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        rho, ci_low, ci_high = bootstrap_spearman(
            model_df["human_entropy"],
            model_df["llm_entropy"],
            n_bootstraps=n_bootstraps,
        )

        boot_rows.append(
            {
                "model": model_name,
                "persona": persona_name,
                "n": len(model_df),
                "rho": rho,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_excludes_zero": (
                    (ci_low > 0 and ci_high > 0) or
                    (ci_low < 0 and ci_high < 0)
                ) if not pd.isna(ci_low) else False,
            }
        )

    boot_df = pd.DataFrame(boot_rows).sort_values(["model", "persona"])

    print("\n" + "=" * 80)
    print("BOOTSTRAPPED SPEARMAN CIs")
    print("=" * 80)
    print(boot_df.round(4))

    boot_df.to_csv(TABLES_PATH / "persona_bootstrapped_spearman.csv", index=False)
    boot_df.to_latex(
        TABLES_PATH / "persona_bootstrapped_spearman.tex",
        index=False,
        float_format="%.3f",
    )

    bin_summary_df = (
        analysis_df
        .groupby(["model", "llm_persona", "human_disagreement_bin"])
        .agg(
            n=("query_id", "count"),

            human_entropy_mean=("human_entropy", "mean"),
            human_entropy_std=("human_entropy", "std"),

            llm_entropy_mean=("llm_entropy", "mean"),
            llm_entropy_std=("llm_entropy", "std"),

            human_vr_mean=("human_variation_ratio", "mean"),
            llm_vr_mean=("llm_variation_ratio", "mean"),

            unstable_rate=("is_unstable", "mean"),

            human_mean_hatespeech=("human_mean_hatespeech", "mean"),
            llm_mean_severity=("llm_mean_severity", "mean"),
        )
        .reset_index()
        .sort_values(["model", "llm_persona", "human_disagreement_bin"])
    )

    print("\n" + "=" * 80)
    print("PERSONA BIN SUMMARY")
    print("=" * 80)
    print(bin_summary_df.round(4))

    bin_summary_df.to_csv(TABLES_PATH / "persona_bin_summary.csv", index=False)
    bin_summary_df.to_latex(
        TABLES_PATH / "persona_bin_summary.tex",
        index=False,
        float_format="%.3f",
    )

    bin_test_rows = []

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        entropy_groups = [
            model_df.loc[
                model_df["human_disagreement_bin"] == bin_name,
                "llm_entropy",
            ].dropna()
            for bin_name in BIN_ORDER
        ]

        group_sizes = [len(group) for group in entropy_groups]

        if all(size >= 2 for size in group_sizes):
            h_stat, h_p = kruskal(*entropy_groups)
        else:
            h_stat, h_p = np.nan, np.nan

        low_values = entropy_groups[0]
        high_values = entropy_groups[2]

        if len(low_values) >= 2 and len(high_values) >= 2:
            u_stat, u_p = mannwhitneyu(
                low_values,
                high_values,
                alternative="two-sided",
            )
            high_minus_low = float(high_values.mean() - low_values.mean())
        else:
            u_stat, u_p, high_minus_low = np.nan, np.nan, np.nan

        bin_test_rows.append(
            {
                "model": model_name,
                "persona": persona_name,

                "n_low": group_sizes[0],
                "n_medium": group_sizes[1],
                "n_high": group_sizes[2],

                "kruskal_H": h_stat,
                "kruskal_p": h_p,

                "mannwhitney_low_vs_high_U": u_stat,
                "mannwhitney_low_vs_high_p": u_p,

                "mean_llm_entropy_high_minus_low": high_minus_low,
            }
        )

    bin_tests_df = pd.DataFrame(bin_test_rows).sort_values(["model", "persona"])

    print("\n" + "=" * 80)
    print("PERSONA BIN TESTS")
    print("=" * 80)
    print(bin_tests_df.round(4))

    bin_tests_df.to_csv(TABLES_PATH / "persona_bin_tests.csv", index=False)
    bin_tests_df.to_latex(
        TABLES_PATH / "persona_bin_tests.tex",
        index=False,
        float_format="%.3f",
    )

    predictive_rows = []

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        y_true = (
            model_df["human_disagreement_bin"] != "low_disagreement"
        ).astype(int)

        y_score = model_df["llm_variation_ratio"]
        y_pred = model_df["is_unstable"]

        metrics = safe_auc_metrics(y_true, y_score, y_pred)

        predictive_rows.append(
            {
                "model": model_name,
                "persona": persona_name,
                "n": len(model_df),
                "positive_rate_human_disagreement": y_true.mean(),
                **metrics,
            }
        )

    predictive_df = pd.DataFrame(predictive_rows).sort_values(["model", "persona"])

    print("\n" + "=" * 80)
    print("PERSONA-LEVEL PREDICTIVE ANALYSIS")
    print("=" * 80)
    print(predictive_df.round(4))

    predictive_df.to_csv(TABLES_PATH / "persona_predictive_metrics.csv", index=False)

    persona_summary = (
        boot_df[["model", "persona", "n", "rho", "ci_low", "ci_high", "ci_excludes_zero"]]
        .merge(
            corr_df[
                [
                    "model",
                    "persona",
                    "rho_human_entropy_vs_llm_entropy",
                    "p_entropy",
                    "rho_human_vr_vs_llm_vr",
                    "p_variation_ratio",
                    "rho_human_bin_vs_llm_entropy",
                    "p_bin_trend",
                    "rho_human_mean_vs_llm_mean_severity",
                    "p_mean_severity",
                    "mean_human_entropy",
                    "mean_llm_entropy",
                    "mean_human_vr",
                    "mean_llm_vr",
                ]
            ],
            on=["model", "persona"],
            how="left",
        )
        .merge(
            bin_tests_df[
                [
                    "model",
                    "persona",
                    "n_low",
                    "n_medium",
                    "n_high",
                    "kruskal_p",
                    "mannwhitney_low_vs_high_p",
                    "mean_llm_entropy_high_minus_low",
                ]
            ],
            on=["model", "persona"],
            how="left",
        )
        .merge(
            predictive_df[
                [
                    "model",
                    "persona",
                    "positive_rate_human_disagreement",
                    "roc_auc",
                    "pr_auc",
                    "precision",
                    "recall",
                ]
            ],
            on=["model", "persona"],
            how="left",
        )
    )

    print("\n" + "=" * 80)
    print("COMBINED PERSONA RQ3 SUMMARY")
    print("=" * 80)
    print(persona_summary.round(4))

    persona_summary.to_csv(TABLES_PATH / "persona_rq3_summary.csv", index=False)
    persona_summary.to_latex(
        TABLES_PATH / "persona_rq3_summary.tex",
        index=False,
        float_format="%.3f",
    )

    behavior_rows = []

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        all_seed_labels = [
            label
            for labels in model_df["seed_outputs"]
            for label in labels
        ]

        label_counts = Counter(all_seed_labels)
        total = sum(label_counts.values())

        behavior_rows.append(
            {
                "model": model_name,
                "persona": persona_name,
                "n_query_persona_rows": len(model_df),
                "n_seed_outputs": total,

                "normal_rate": label_counts.get("normal", 0) / total if total else np.nan,
                "offensive_rate": label_counts.get("offensive", 0) / total if total else np.nan,
                "hatespeech_rate": label_counts.get("hatespeech", 0) / total if total else np.nan,

                "mean_llm_entropy": model_df["llm_entropy"].mean(),
                "mean_llm_variation_ratio": model_df["llm_variation_ratio"].mean(),
                "mean_llm_severity": model_df["llm_mean_severity"].mean(),
                "unstable_rate": model_df["is_unstable"].mean(),
            }
        )

    behavior_df = pd.DataFrame(behavior_rows).sort_values(["model", "persona"])

    print("\n" + "=" * 80)
    print("PERSONA LABEL BEHAVIOR")
    print("=" * 80)
    print(behavior_df.round(4))

    behavior_df.to_csv(TABLES_PATH / "persona_label_behavior.csv", index=False)
    behavior_df.to_latex(
        TABLES_PATH / "persona_label_behavior.tex",
        index=False,
        float_format="%.3f",
    )

    analysis_df["human_entropy_z"] = (
        analysis_df["human_entropy"] - analysis_df["human_entropy"].mean()
    ) / analysis_df["human_entropy"].std(ddof=0)

    analysis_df["llm_entropy_z"] = (
        analysis_df["llm_entropy"] - analysis_df["llm_entropy"].mean()
    ) / analysis_df["llm_entropy"].std(ddof=0)

    analysis_df["entropy_gap_z"] = (
        analysis_df["llm_entropy_z"] - analysis_df["human_entropy_z"]
    )

    overconfident_df = analysis_df[
        (analysis_df["human_disagreement_bin"] == "high_disagreement") &
        (analysis_df["llm_variation_ratio"] == 0.0)
    ].copy()

    noisy_df = analysis_df[
        (analysis_df["human_disagreement_bin"] == "low_disagreement") &
        (analysis_df["llm_variation_ratio"] > 0.0)
    ].copy()

    qual_cols = [
        "query_id",
        "text",
        "model",
        "llm_persona",
        "human_persona",
        "human_disagreement_bin",

        "human_entropy",
        "llm_entropy",
        "human_variation_ratio",
        "llm_variation_ratio",

        "human_mean_hatespeech",
        "llm_mean_severity",

        "seed_outputs",
        "llm_label_counts",
        "entropy_gap_z",
    ]

    overconfident_df[qual_cols].sort_values(
        ["model", "llm_persona", "human_entropy"],
        ascending=[True, True, False],
    ).to_csv(
        OUTPUT_PATH / "qual_overconfident_high_human_disagreement_stable_llm.csv",
        index=False,
    )

    noisy_df[qual_cols].sort_values(
        ["model", "llm_persona", "llm_entropy"],
        ascending=[True, True, False],
    ).to_csv(
        OUTPUT_PATH / "qual_noisy_low_human_disagreement_unstable_llm.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("QUALITATIVE MISMATCH COUNTS")
    print("=" * 80)

    print("\nOverconfident cases: high human disagreement, stable LLM")
    print(
        overconfident_df
        .groupby(["model", "llm_persona"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nNoisy cases: low human disagreement, unstable LLM")
    print(
        noisy_df
        .groupby(["model", "llm_persona"])
        .size()
        .unstack(fill_value=0)
    )

    # Save mismatch rates with denominators, matching the global analysis style.
    high_denoms = (
        analysis_df[analysis_df["human_disagreement_bin"] == "high_disagreement"]
        .groupby(["model", "llm_persona"])
        .size()
        .rename("n_high_disagreement")
        .reset_index()
    )

    low_denoms = (
        analysis_df[analysis_df["human_disagreement_bin"] == "low_disagreement"]
        .groupby(["model", "llm_persona"])
        .size()
        .rename("n_low_disagreement")
        .reset_index()
    )

    over_counts = (
        overconfident_df
        .groupby(["model", "llm_persona"])
        .size()
        .rename("overconfident")
        .reset_index()
    )

    noisy_counts = (
        noisy_df
        .groupby(["model", "llm_persona"])
        .size()
        .rename("noisy")
        .reset_index()
    )

    mismatch_summary = (
        high_denoms
        .merge(over_counts, on=["model", "llm_persona"], how="left")
        .merge(low_denoms, on=["model", "llm_persona"], how="left")
        .merge(noisy_counts, on=["model", "llm_persona"], how="left")
        .fillna(0)
    )

    mismatch_summary["overconfident_rate"] = (
        mismatch_summary["overconfident"] / mismatch_summary["n_high_disagreement"]
    )

    mismatch_summary["noisy_rate"] = (
        mismatch_summary["noisy"] / mismatch_summary["n_low_disagreement"]
    )

    print("\nPersona mismatch summary:")
    print(mismatch_summary.round(4))

    mismatch_summary.to_csv(TABLES_PATH / "persona_mismatch_summary.csv", index=False)
    mismatch_summary.to_latex(
        TABLES_PATH / "persona_mismatch_summary.tex",
        index=False,
        float_format="%.3f",
    )

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        safe_model = str(model_name).replace("/", "_").replace(":", "_")
        safe_persona = str(persona_name).replace("/", "_").replace(":", "_")

        model_df = model_df.copy()
        model_df["human_disagreement_bin"] = pd.Categorical(
            model_df["human_disagreement_bin"],
            categories=BIN_ORDER,
            ordered=True,
        )

        plt.figure(figsize=(7, 5))

        model_df.boxplot(
            column="llm_entropy",
            by="human_disagreement_bin",
            grid=False,
        )

        plt.title(f"LLM entropy by human bin\n{model_name} | {persona_name}")
        plt.suptitle("")
        plt.xlabel("Human subgroup disagreement bin")
        plt.ylabel("LLM entropy across seeds")
        plt.tight_layout()

        plt.savefig(
            PLOTS_PATH / f"boxplot_entropy_by_bin_{safe_model}_{safe_persona}.png",
            dpi=300,
        )

        plt.close()

    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        safe_model = str(model_name).replace("/", "_").replace(":", "_")
        safe_persona = str(persona_name).replace("/", "_").replace(":", "_")

        model_df = model_df.copy()
        model_df["human_disagreement_bin"] = pd.Categorical(
            model_df["human_disagreement_bin"],
            categories=BIN_ORDER,
            ordered=True,
        )

        crosstab = pd.crosstab(
            model_df["human_disagreement_bin"],
            model_df["llm_variation_ratio"],
            normalize="index",
        ) * 100

        fig, ax = plt.subplots(figsize=(9, 5))

        crosstab.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            edgecolor="black",
        )

        ax.set_title(f"LLM instability distribution\n{model_name} | {persona_name}")
        ax.set_xlabel("Human subgroup disagreement bin")
        ax.set_ylabel("Percentage of queries")
        ax.legend(
            title="LLM variation ratio",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            PLOTS_PATH / f"stacked_vr_by_bin_{safe_model}_{safe_persona}.png",
            dpi=300,
        )

        plt.close()


    for (model_name, persona_name), model_df in analysis_df.groupby(["model", "llm_persona"]):
        safe_model = str(model_name).replace("/", "_").replace(":", "_")
        safe_persona = str(persona_name).replace("/", "_").replace(":", "_")

        plt.figure(figsize=(6, 5))

        plt.scatter(
            model_df["human_entropy"],
            model_df["llm_entropy"],
            alpha=0.6,
        )

        rho, p_value = safe_spearman(
            model_df["human_entropy"],
            model_df["llm_entropy"],
        )

        plt.title(
            f"Human vs LLM entropy\n{model_name} | {persona_name}\n"
            f"Spearman rho={rho:.3f}, p={p_value:.3g}"
        )
        plt.xlabel("Human subgroup entropy")
        plt.ylabel("LLM entropy across seeds")
        plt.tight_layout()

        plt.savefig(
            PLOTS_PATH / f"scatter_human_vs_llm_entropy_{safe_model}_{safe_persona}.png",
            dpi=300,
        )

        plt.close()

    for model_name, model_df_all in analysis_df.groupby("model"):
        safe_model = str(model_name).replace("/", "_").replace(":", "_")

        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

        plotted_any = False

        for persona_name, model_df in model_df_all.groupby("llm_persona"):
            y_true = (
                model_df["human_disagreement_bin"] != "low_disagreement"
            ).astype(int)

            y_score = model_df["llm_variation_ratio"]

            if y_true.nunique() < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            precision_values, recall_values, _ = precision_recall_curve(y_true, y_score)

            roc_auc = roc_auc_score(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            pr_baseline = y_true.mean()

            ax_roc.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{persona_name} AUC={roc_auc:.2f}",
            )

            pr_line, = ax_pr.plot(
                recall_values,
                precision_values,
                linewidth=2,
                label=f"{persona_name} AUC={pr_auc:.2f}, no-skill={pr_baseline:.2f}",
            )

            # Draw a persona-specific PR baseline in the same color.
            ax_pr.axhline(
                pr_baseline,
                color=pr_line.get_color(),
                linewidth=1.3,
                linestyle=":",
                alpha=0.8,
            )

            plotted_any = True

        if plotted_any:
            ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random")
            ax_roc.set_title("ROC curve")
            ax_roc.set_xlabel("False positive rate")
            ax_roc.set_ylabel("True positive rate")
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)

            ax_pr.set_title("Precision-recall curve")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.legend()
            ax_pr.grid(True, alpha=0.3)

            plt.suptitle(f"LLM variation ratio as human disagreement detector\n{model_name}")
            plt.tight_layout()

            plt.savefig(
                PLOTS_PATH / f"persona_roc_pr_curves_{safe_model}.png",
                dpi=300,
            )

        plt.close()

    heat_df = corr_df.copy()
    heat_df["row"] = heat_df["model"] + " | " + heat_df["persona"]

    heat_values = heat_df.set_index("row")[["rho_human_entropy_vs_llm_entropy"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(heat_values))))

    image = ax.imshow(heat_values.values, aspect="auto")

    ax.set_yticks(np.arange(len(heat_values.index)))
    ax.set_yticklabels(heat_values.index)

    ax.set_xticks([0])
    ax.set_xticklabels(["Spearman rho"])

    for i in range(len(heat_values.index)):
        value = heat_values.iloc[i, 0]
        text = "nan" if pd.isna(value) else f"{value:.2f}"
        ax.text(0, i, text, ha="center", va="center")

    ax.set_title("Matched persona entropy correlations")
    plt.colorbar(image, ax=ax)
    plt.tight_layout()

    plt.savefig(
        PLOTS_PATH / "persona_matched_entropy_correlation_heatmap.png",
        dpi=300,
    )

    plt.close()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved full matched dataframe to: {OUTPUT_PATH / 'persona_matched_analysis_full.csv'}")
    print(f"Saved tables to: {TABLES_PATH}")
    print(f"Saved plots to: {PLOTS_PATH}")

    return {
        "analysis_df": analysis_df,
        "llm_df": llm_df,
        "corr_df": corr_df,
        "boot_df": boot_df,
        "bin_summary_df": bin_summary_df,
        "bin_tests_df": bin_tests_df,
        "predictive_df": predictive_df,
        "behavior_df": behavior_df,
        "noncomp_table": noncomp_table,
        "persona_summary": persona_summary,
        "mismatch_summary": mismatch_summary,
    }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    HUMAN_DATA_PATH_MHS_PERSONA = (
        PROJECT_ROOT
        / "data"
        / "measuringhatespeech"
        / "mhs_persona_entropy_bins_fixed_sampled.csv"
    )

    RESULTS_PATH_MHS_PERSONA = (
        PROJECT_ROOT
        / "data"
        / "measuringhatespeech"
        / "results_ollama_runs_mhs_perona.jsonl"
    )

    OUTPUT_PATH = (
        PROJECT_ROOT
        / "rq3"
        / "persona_analysis_outputs"
    )

    analyze_mhs_persona_dataset(
        HUMAN_DATA_PATH=HUMAN_DATA_PATH_MHS_PERSONA,
        RESULTS_PATH=RESULTS_PATH_MHS_PERSONA,
        OUTPUT_PATH=OUTPUT_PATH,
        required_n_seeds=5,
        n_bootstraps=1000,
    )