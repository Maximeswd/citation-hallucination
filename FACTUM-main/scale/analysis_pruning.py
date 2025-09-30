# ==============================================================================
# --- IMPORTS & SETUP ---
# ==============================================================================
import json
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from scipy.stats import pearsonr, ttest_rel, mannwhitneyu
import random
from collections import defaultdict, Counter
from itertools import product
import re
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import ruptures as rpt

from itertools import combinations
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import resample
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier


# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
def config():
    """
    Sets up the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="A full ablation pipeline to compare a single level of component pruning."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the full scores file (JSONL).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="hallucination_analysis_final",
        help="Base directory for all output files.",
    )
    parser.add_argument(
        "--run_ebm", action="store_true", help="If set, include the EBM model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--top_n_confident",
        type=int,
        default=10,
        help="Number of most confident predictions to analyze for plots.",
    )
    parser.add_argument(
        "--k_percent",
        type=int,
        required=True,
        help="The single pruning percentage (e.g., 10, 25) to run.",
    )
    return parser.parse_args()


# ==============================================================================
# --- STANDALONE DATA LOADING & FEATURE ENGINEERING ---
# ==============================================================================
def seed_worker(worker_seed):
    """Initializes the random seed for each parallel worker."""
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parallel_index_worker(args):
    """Parallel worker to create a byte-offset index for a chunk of the JSONL file."""
    filepath, chunk_start, chunk_end = args
    local_index = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(chunk_start)
        if chunk_start > 0:
            f.readline()  # Align to the next full line
        current_pos = f.tell()
        while current_pos < chunk_end:
            line_offset = current_pos
            line = f.readline()
            if not line:
                break
            if '"source_id"' in line:
                try:
                    doc = json.loads(line)
                    if source_id := doc.get("source_id"):
                        local_index[source_id].append(line_offset)
                except json.JSONDecodeError:
                    pass
            current_pos = f.tell()
    return local_index


def worker_process_source_group(source_ids_chunk, byte_offset_map, jsonl_path):
    """Parallel worker to load and process a chunk of source_id groups."""
    chunk_tokens = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for source_id in source_ids_chunk:
            for offset in byte_offset_map.get(source_id, []):
                try:
                    f.seek(offset)
                    line = f.readline()
                    doc = json.loads(line)
                    tokens_from_doc = doc.get("token_data", [])
                    for token_data in tokens_from_doc:
                        token_data["source_id"] = doc.get("source_id")
                    chunk_tokens.extend(tokens_from_doc)
                except (json.JSONDecodeError, IndexError):
                    continue
    return chunk_tokens


def deterministic_chunking(data, num_chunks):
    """Splits a list into a specified number of nearly equal-sized chunks."""
    k, m = divmod(len(data), num_chunks)
    return (
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)
    )


def load_full_imbalanced_data(
    filepath: str, num_workers: int, seed: int
) -> pd.DataFrame:
    """
    Loads the full imbalanced dataset from a JSONL file in two parallel passes.
    Pass 1: Builds a byte-offset index of source_ids.
    Pass 2: Loads the data in chunks based on the index.
    """
    print("Pass 1: Creating byte-offset index in parallel...")
    file_size = os.path.getsize(filepath)
    chunk_size = file_size // num_workers
    pool_args = [
        (filepath, i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)
    ]
    pool_args[-1] = (filepath, (num_workers - 1) * chunk_size, file_size)

    byte_offset_map = defaultdict(list)
    with Pool(processes=num_workers) as pool:
        for local_index in tqdm(
            pool.imap(parallel_index_worker, pool_args),
            total=len(pool_args),
            desc="Indexing file chunks",
        ):
            for source_id, offsets in local_index.items():
                byte_offset_map[source_id].extend(offsets)

    unique_source_ids = sorted(list(byte_offset_map.keys()))
    print(
        f"\nPass 2: Loading all data from {len(unique_source_ids)} source_id groups..."
    )

    source_id_chunks = list(deterministic_chunking(unique_source_ids, num_workers))
    worker_func = partial(
        worker_process_source_group,
        byte_offset_map=byte_offset_map,
        jsonl_path=filepath,
    )

    all_tokens = []
    with Pool(processes=num_workers, initializer=seed_worker, initargs=(seed,)) as pool:
        for tokens_from_chunk in tqdm(
            pool.imap(worker_func, source_id_chunks),
            total=len(source_id_chunks),
            desc="Processing data chunks",
        ):
            all_tokens.extend(tokens_from_chunk)

    if not all_tokens:
        raise ValueError("No token data was loaded. Check the file path and format.")

    df = pd.DataFrame(all_tokens)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def create_per_layer_features(df: pd.DataFrame):
    """
    Expands the list-based scores in the DataFrame into individual columns for each layer/head.
    Also dynamically determines the number of layers and heads from the data.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int), 0, 0

    sample_token = df.iloc[0]
    score_keys = [
        k
        for k, v in sample_token.items()
        if isinstance(v, list) and v and isinstance(v[0], (int, float))
    ]

    num_layers, num_heads = 0, 0
    if per_layer_key := next((k for k in score_keys if "per_layer" in k), None):
        num_layers = len(sample_token[per_layer_key])
    if num_layers == 0:
        raise ValueError("Could not determine number of layers from the data.")

    if per_head_key := next((k for k in score_keys if "per_head" in k), None):
        num_heads = len(sample_token[per_head_key]) // num_layers
    if num_heads == 0:
        raise ValueError("Could not determine number of heads from the data.")

    processed_records = []
    for _, token in tqdm(
        df.iterrows(), total=df.shape[0], desc="Engineering per-layer features"
    ):
        features = {}
        for key in score_keys:
            scores = token.get(key)
            if not isinstance(scores, list) or not scores:
                continue
            if len(scores) == num_layers * num_heads:  # Head-based feature
                for i in range(num_layers):
                    for h in range(num_heads):
                        features[f"{key}_layer_{i}_head_{h}"] = scores[
                            i * num_heads + h
                        ]
            elif len(scores) == num_layers:  # Layer-based feature
                for i in range(num_layers):
                    features[f"{key}_layer_{i}"] = scores[i]
        processed_records.append(features)

    return (
        pd.DataFrame(processed_records).fillna(0),
        df["label"].astype(int),
        num_layers,
        num_heads,
    )


def create_aggregate_features(df_per_layer: pd.DataFrame):
    """Creates aggregate features (mean, std, etc.) from per-layer features."""
    if df_per_layer.empty:
        return pd.DataFrame()
    all_feature_dfs = []
    prefixes = set()
    for col in df_per_layer.columns:
        match = re.match(r"^(.*_layer_)", col)
        if match:
            prefixes.add(re.sub(r"head_\d+_", "", match.group(1)))

    for prefix in tqdm(
        prefixes,
        desc="Engineering aggregate features",
        leave=False,
        bar_format="{l_bar}{bar:10}{r_bar}",
    ):
        metric_cols = sorted(
            [c for c in df_per_layer.columns if c.startswith(prefix)],
            key=lambda x: int(re.search(r"_layer_(\d+)", x).group(1)),
        )
        if len(metric_cols) < 4:
            continue

        metric_data = df_per_layer[metric_cols]
        x = np.arange(metric_data.shape[1])
        agg_prefix = re.sub(r"(_head_\d+)?_layer_$", "_", prefix)

        prefix_features = {
            f"{agg_prefix}mean": metric_data.mean(axis=1),
            f"{agg_prefix}std": metric_data.std(axis=1),
            f"{agg_prefix}max": metric_data.max(axis=1),
            f"{agg_prefix}sum": metric_data.sum(axis=1),
        }
        if len(x) > 1:
            prefix_features[f"{agg_prefix}slope"] = np.polyfit(x, metric_data.T, 1)[0]
        fft_coeffs = np.fft.rfft(metric_data, axis=1)
        for i in range(1, 7):
            if fft_coeffs.shape[1] > i:
                prefix_features[f"{agg_prefix}fft_mag_{i}"] = np.abs(fft_coeffs[:, i])
        all_feature_dfs.append(pd.DataFrame(prefix_features))

    if not all_feature_dfs:
        return pd.DataFrame()

    return pd.concat(all_feature_dfs, axis=1).fillna(0)


# ==============================================================================
# --- CORE PIPELINE FUNCTIONS ---
# ==============================================================================
def stage1a_rank_all_components(X_dev, y_dev, output_dir):
    """
    Calculates and saves the Pearson correlation of every single raw component
    with the hallucination label.
    """
    print("\n--- Stage 1A: Ranking all individual components for analysis ---")
    all_components = list(X_dev.columns)
    correlation_results = []

    for component in tqdm(all_components, desc="Ranking all raw components"):
        if X_dev[component].nunique() > 1 and np.std(X_dev[component]) > 0:
            r_val, p_val = pearsonr(y_dev, X_dev[component])
            correlation_results.append(
                {
                    "component_name": component,
                    "correlation_r": r_val,
                    "abs_correlation": np.abs(r_val),
                    "p_value": p_val,
                }
            )

    corr_df = pd.DataFrame(correlation_results).sort_values(
        by="abs_correlation", ascending=False
    )
    save_path = os.path.join(output_dir, "report_all_component_correlations.csv")
    corr_df.to_csv(save_path, index=False, float_format="%.5f")
    print(f"  -> Saved full component ranking to {save_path}")
    return corr_df


def select_top_k_components(X_data, y_data, k_percent, feature_concepts):
    """
    Selects the top k% of components for each feature concept based on correlation.
    """
    all_components = list(X_data.columns)
    correlation_results = []

    for component in all_components:
        if X_data[component].nunique() > 1 and np.std(X_data[component]) > 0:
            r_val, _ = pearsonr(y_data, X_data[component])
            correlation_results.append(
                {"component_name": component, "abs_correlation": np.abs(r_val)}
            )

    corr_df = pd.DataFrame(correlation_results).sort_values(
        by="abs_correlation", ascending=False
    )

    best_components = {}
    for concept, prefix in feature_concepts.items():
        concept_df = corr_df[corr_df["component_name"].str.startswith(prefix)]
        num_to_select = int(len(concept_df) * (k_percent / 100.0))
        num_to_select = max(
            1, num_to_select
        )  # Ensure at least one component is selected
        best_components[concept] = concept_df.head(num_to_select)[
            "component_name"
        ].tolist()

    return best_components


def create_pruned_aggregate_features(df_per_layer, selected_components):
    """
    Creates aggregate features using only the subset of selected (pruned) components.
    """
    all_selected_cols = [
        col for sublist in selected_components.values() for col in sublist
    ]
    valid_cols = [col for col in set(all_selected_cols) if col in df_per_layer.columns]

    if not valid_cols:
        return pd.DataFrame()

    return create_aggregate_features(df_per_layer[valid_cols])


def get_candidate_shortlist(X_dev, y_dev, models, feature_concepts, top_n=5):
    """
    Finds the top N most predictive aggregate features for each concept to reduce
    the search space for the "dream team" search.
    """
    print("\n--- Stage 1: Finding Top-N Candidate Features for each Concept ---")
    candidate_features = defaultdict(dict)

    for model_name, model in models.items():
        print(f"  -> Generating single-feature performance ranks for {model_name}...")
        single_feature_performance = []
        for feature in tqdm(
            X_dev.columns, desc=f"Ranking features with {model_name}", leave=False
        ):
            scores = cross_val_score(
                clone(model), X_dev[[feature]], y_dev, cv=3, scoring="roc_auc"
            )
            single_feature_performance.append(
                {"Feature": feature, "auc": np.mean(scores)}
            )

        summary_df = pd.DataFrame(single_feature_performance).sort_values(
            by="auc", ascending=False
        )

        for concept, prefix in feature_concepts.items():
            concept_df = summary_df[summary_df["Feature"].str.startswith(prefix)]
            candidate_features[model_name][concept] = concept_df.head(top_n)[
                "Feature"
            ].tolist()

    return candidate_features


def find_dream_team_features(
    X_dev, y_dev, model, candidate_features_for_model, concepts_to_combine
):
    """
    Performs an exhaustive search over the candidate features to find the
    best-performing combination ("dream team") for a given model and set of concepts.
    """
    print(
        f"\n--- Stage 2: Finding 'Dream Team' for {len(concepts_to_combine)} concepts ---"
    )
    candidate_lists = [
        candidate_features_for_model.get(c, []) for c in concepts_to_combine
    ]
    if not all(candidate_lists):
        return []

    all_combinations = list(product(*candidate_lists))
    print(f"  -> Testing {len(all_combinations)} total combinations...")

    best_combination, best_score = None, -1

    for feature_combination in tqdm(
        all_combinations, desc="Evaluating combinations", leave=False
    ):
        feature_list = list(set(feature_combination))  # Remove duplicates
        scores = cross_val_score(
            clone(model), X_dev[feature_list], y_dev, cv=3, scoring="roc_auc"
        )
        current_score = np.mean(scores)

        if current_score > best_score:
            best_score, best_combination = current_score, feature_list

    print(f"  -> Dream Team found with CV AUC: {best_score:.4f}")
    return best_combination


def validate_dream_teams(
    dream_teams, X_full, y_full, output_dir, FEATURE_CONCEPT_PREFIXES
):
    """
    Creates a report validating the selected "dream team" features by showing their
    correlation with the hallucination label on the full dataset.
    """
    print("\n--- Stage 3: Validating 'Dream Team' Features via Correlation ---")
    validation_records = []

    all_selected_features = set()
    for model_family in dream_teams.values():
        for feature_list in model_family.values():
            if feature_list:
                all_selected_features.update(feature_list)

    correlation_map = {}
    for feature in sorted(list(all_selected_features)):
        if feature in X_full.columns:
            r_val, p_val = pearsonr(y_full, X_full[feature])
            correlation_map[feature] = {"r": r_val, "p_value": p_val}

    for model_name, combos in dream_teams.items():
        for combo_name, feature_list in combos.items():
            if not feature_list:
                continue
            for feature in feature_list:
                concept = next(
                    (
                        c.upper()
                        for c, p in FEATURE_CONCEPT_PREFIXES.items()
                        if feature.startswith(p)
                    ),
                    "Unknown",
                )
                if corr_data := correlation_map.get(feature):
                    validation_records.append(
                        {
                            "Classifier": model_name,
                            "Model_Combination": combo_name,
                            "Mechanistic_Concept": concept,
                            "Selected_Feature": feature,
                            "Correlation_r": corr_data["r"],
                            "P_Value": corr_data["p_value"],
                            "Is_Significant_p<0.05": "Yes"
                            if corr_data["p_value"] < 0.05
                            else "No",
                        }
                    )

    if not validation_records:
        return

    validation_df = pd.DataFrame(validation_records)
    save_path = os.path.join(output_dir, "report_dream_team_validation.csv")
    validation_df.to_csv(save_path, index=False, float_format="%.5f")
    print(f"  -> Saved unified feature validation report to {save_path}")


def get_feature_importances(model, feature_names):
    """
    Extracts feature importances from a trained model, with specific handling
    for scikit-learn models and ExplainableBoostingClassifier.
    """
    classifier = model
    if isinstance(model, Pipeline):
        classifier = model.steps[-1][1]

    if hasattr(classifier, "coef_"):
        return pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(classifier.coef_[0])}
        )
    elif hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(
            {"feature": feature_names, "importance": classifier.feature_importances_}
        )
    elif isinstance(classifier, ExplainableBoostingClassifier):
        return pd.DataFrame(
            {"feature": feature_names, "importance": classifier.term_importances()}
        )

    return pd.DataFrame()


def run_final_cv_and_analysis(
    X_per_layer_full,
    y_full,
    models,
    dream_teams,
    k,
    FEATURE_CONCEPT_PREFIXES,
    feature_sets_to_build,
    args,
):
    """
    Runs the main 10x10 repeated stratified cross-validation.
    On each fold, it performs top-k pruning on the train set, builds aggregate features,
    trains the models, and evaluates on the test set.
    Collects and returns test fold data for qualitative analysis.
    """
    print(f"\n--- Stage 4 (k={k}%): Running Final 10x10 CV with On-the-Fly Pruning ---")
    n_splits, n_repeats = 10, 10
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=args.seed
    )

    all_scores = defaultdict(lambda: defaultdict(list))
    all_importances_list = []
    all_selections_list = []
    all_holdout_data = defaultdict(list)  

    pbar = tqdm(
        rskf.split(X_per_layer_full, y_full),
        total=n_splits * n_repeats,
        desc=f"Overall CV Progress for k={k}%",
    )

    for train_idx, test_idx in pbar:
        X_train_per_layer, X_test_per_layer = (
            X_per_layer_full.iloc[train_idx],
            X_per_layer_full.iloc[test_idx],
        )
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        # Balance the training set for this fold using downsampling
        train_df_fold = pd.concat([X_train_per_layer, y_train.rename("label")], axis=1)
        minority_class = train_df_fold[train_df_fold["label"] == 1]
        majority_class = train_df_fold[train_df_fold["label"] == 0]
        if len(minority_class) == 0:
            continue
        majority_downsampled = resample(
            majority_class,
            replace=False,
            n_samples=len(minority_class),
            random_state=args.seed,
        )
        train_df_balanced = pd.concat([minority_class, majority_downsampled])
        X_train_balanced_per_layer, y_train_balanced = (
            train_df_balanced.drop("label", axis=1),
            train_df_balanced["label"],
        )

        for model_name, model in models.items():
            for combo_name, dream_team_features in dream_teams[model_name].items():
                if not dream_team_features:
                    continue

                concepts_in_combo = feature_sets_to_build[combo_name]
                feature_concepts_subset = {
                    c: p
                    for c, p in FEATURE_CONCEPT_PREFIXES.items()
                    if c in concepts_in_combo
                }

                # Perform pruning based on the balanced training data for this fold
                selected_components = select_top_k_components(
                    X_train_balanced_per_layer,
                    y_train_balanced,
                    k,
                    feature_concepts_subset,
                )
                all_selections_list.append(selected_components)

                # Create aggregate features from the pruned components
                X_train_pruned_agg = create_pruned_aggregate_features(
                    X_train_balanced_per_layer, selected_components
                )
                X_test_pruned_agg = create_pruned_aggregate_features(
                    X_test_per_layer, selected_components
                )

                final_feature_list = [
                    f
                    for f in dream_team_features
                    if f in X_train_pruned_agg.columns
                    and f in X_test_pruned_agg.columns
                ]
                if not final_feature_list:
                    continue

                current_model = clone(model)
                current_model.fit(
                    X_train_pruned_agg[final_feature_list], y_train_balanced
                )
                probas = current_model.predict_proba(
                    X_test_pruned_agg[final_feature_list]
                )[:, 1]
                preds = (probas > 0.5).astype(int)

                # Store scores
                scores = all_scores[(model_name, combo_name)]
                scores["AUC"].append(roc_auc_score(y_test, probas))
                scores["PCC"].append(pearsonr(y_test, probas)[0])
                scores["Precision"].append(
                    precision_score(y_test, preds, zero_division=0)
                )
                scores["Recall"].append(recall_score(y_test, preds, zero_division=0))
                scores["F1"].append(f1_score(y_test, preds, zero_division=0))

                # Store feature importances
                imp_df = get_feature_importances(current_model, final_feature_list)
                if not imp_df.empty:
                    imp_df["model"], imp_df["combo"] = model_name, combo_name
                    all_importances_list.append(imp_df)

                # Store minimal holdout data for plotting
                holdout_fold_df = X_test_pruned_agg[final_feature_list].copy()
                holdout_fold_df["label"] = y_test
                holdout_fold_df["probability"] = probas
                all_holdout_data[(model_name, combo_name)].append(holdout_fold_df)

    return (
        all_scores,
        pd.concat(all_importances_list) if all_importances_list else pd.DataFrame(),
        all_selections_list,
        all_holdout_data,
    )


def report_significance_and_importances(
    all_scores, all_importances_df, models, output_dir
):
    """
    Performs statistical tests to compare model combinations and aggregates
    feature importances, saving both as reports.
    """
    print("\n--- Stage 5: Performing Statistical Analysis and Reporting ---")

    final_results = []
    metrics = ["AUC", "PCC", "Precision", "Recall", "F1"]

    # Define all pairwise comparisons to be made
    comparisons_to_make = [
        ("Baseline", "Ours 1"),
        ("Baseline", "Ours 2"),
        ("Ours 1", "Ours 2"), 
    ]

    for model_name in models.keys():
        model_tests = []
        for base_combo, comp_combo in comparisons_to_make:
            for metric in metrics:
                base_scores = all_scores.get((model_name, base_combo), {}).get(
                    metric, []
                )
                comp_scores = all_scores.get((model_name, comp_combo), {}).get(
                    metric, []
                )

                if (
                    not base_scores
                    or not comp_scores
                    or len(base_scores) != len(comp_scores)
                ):
                    continue
                
                # Use paired t-test for significance
                t_stat, p_value = ttest_rel(
                    comp_scores, base_scores, alternative="greater"
                )
                model_tests.append(
                    {
                        "Metric": metric,
                        "Comparison": f"{comp_combo} vs. {base_combo}",
                        "Mean_Baseline_Score": np.mean(base_scores),
                        "Mean_Comparison_Score": np.mean(comp_scores),
                        "P_Value_raw": p_value,
                    }
                )

        # Perform Benjamini/Hochberg correction on all p-values for the current model
        if model_tests:
            raw_p_values = [test["P_Value_raw"] for test in model_tests]
            is_significant_bh, p_corrected, _, _ = multipletests(
                raw_p_values, alpha=0.05, method="fdr_bh"
            )
            for i, test in enumerate(model_tests):
                test.update(
                    {
                        "P_Value_BH_Corrected": p_corrected[i],
                        "Is_Significant_BH": "Yes" if is_significant_bh[i] else "No",
                        "Classifier": model_name,
                    }
                )
                final_results.append(test)

    final_report_df = pd.DataFrame(final_results)

    # Feature importance processing
    avg_importances_df = pd.DataFrame()
    if not all_importances_df.empty:
        avg_importances_df = (
            all_importances_df.groupby(["model", "combo", "feature"])
            .importance.mean()
            .reset_index()
        )
        avg_importances_path = os.path.join(
            output_dir, "report_avg_feature_importances.csv"
        )
        avg_importances_df.to_csv(
            avg_importances_path, index=False, float_format="%.4f"
        )
        print(f"  -> Saved average feature importances to {avg_importances_path}")

    return final_report_df, avg_importances_df


def format_p_value(p):
    """Formats a p-value for clean display on a plot."""
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def generate_qualitative_plots(
    data_df,
    avg_importances_df,
    model_name,
    combo_name,
    k,
    top_n_confident,
    plot_dir,
    FEATURE_CONCEPT_PREFIXES,
    plot_suffix,
):
    """
    NEW HELPER FUNCTION
    Generates and saves a set of violin plots for a given dataframe (either single fold or aggregated).
    """
    if data_df.empty:
        return []

    # Find the "best" feature for each concept based on average importance from CV
    imp_df = avg_importances_df[
        (avg_importances_df["model"] == model_name)
        & (avg_importances_df["combo"] == combo_name)
    ]
    if imp_df.empty:
        return []

    concepts_in_combo = set()
    for f in data_df.columns:
        for concept, prefix in FEATURE_CONCEPT_PREFIXES.items():
            if f.startswith(prefix.replace("_baseline", "")):
                concepts_in_combo.add(concept)

    violin_features = []
    for concept_key in sorted(
        list(concepts_in_combo)
    ):  # Sort for consistent plot order
        concept_prefix = FEATURE_CONCEPT_PREFIXES[concept_key].replace("_baseline", "")
        concept_imp_df = imp_df[imp_df["feature"].str.startswith(concept_prefix)]
        if not concept_imp_df.empty:
            best_feature = concept_imp_df.loc[concept_imp_df["importance"].idxmax()][
                "feature"
            ]
            if best_feature in data_df.columns:
                violin_features.append(best_feature)

    if not violin_features:
        return []

    # Significance test data prep
    significance_results = []
    data_df["label_str"] = data_df["label"].map({0: "Correct", 1: "Hallucinated"})

    # Create the confident subset for this dataframe
    correct_df = data_df[data_df["label"] == 0]
    halluc_df = data_df[data_df["label"] == 1]
    n_samples = min(top_n_confident, len(correct_df), len(halluc_df))
    confident_df = pd.DataFrame()
    if n_samples >= 2:
        confident_df = pd.concat(
            [
                correct_df.sort_values("probability", ascending=True).head(n_samples),
                halluc_df.sort_values("probability", ascending=False).head(n_samples),
            ]
        )

    # Plotting 
    num_viol_cols = 2
    num_viol_rows = math.ceil(len(violin_features) / num_viol_cols)
    fig_viol, axes_viol = plt.subplots(
        num_viol_rows,
        num_viol_cols,
        figsize=(6 * num_viol_cols, 5 * num_viol_rows),
        sharey=False,
        squeeze=False,
    )
    axes_viol = axes_viol.flatten()
    fig_viol.suptitle(
        f"Key Feature Distributions for {model_name} - {combo_name}\n({plot_suffix})",
        fontsize=16,
    )

    for i, feature in enumerate(violin_features):
        ax = axes_viol[i]
        concept = next(
            (
                c.upper()
                for c, p in FEATURE_CONCEPT_PREFIXES.items()
                if feature.startswith(p.replace("_baseline", ""))
            ),
            "Unknown",
        )

        sns.violinplot(
            data=data_df,
            x="label_str",
            y=feature,
            hue="label_str",
            palette={"Correct": "#2390CA", "Hallucinated": "#DE6A05"},
            inner="box",
            ax=ax,
            legend=False,
        )
        ax.set_title(f"Best Feature for {concept}")
        ax.set_xlabel("")
        ax.set_ylabel("Score" if ax.get_subplotspec().is_first_col() else "")

        # Perform and annotate statistical sests 
        group_correct_full = data_df.loc[data_df["label"] == 0, feature]
        group_halluc_full = data_df.loc[data_df["label"] == 1, feature]

        if len(group_correct_full) > 1 and len(group_halluc_full) > 1:
            stat_full, p_value_full = mannwhitneyu(
                group_correct_full, group_halluc_full, alternative="two-sided"
            )
            p_text_full = f"{'p (all)' if plot_suffix == 'All Folds' else 'p (holdout)'}: {format_p_value(p_value_full)}"
            ax.text(
                0.5,
                0.95,
                p_text_full,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                weight="bold",
            )
            significance_results.append(
                {
                    "Test_Data_Scope": f"Full_{plot_suffix.replace(' ', '_')}",
                    "k_percent": k,
                    "Classifier": model_name,
                    "Model_Combination": combo_name,
                    "Mechanistic_Concept": concept,
                    "Most_Important_Feature": feature,
                    "MannWhitneyU_statistic": stat_full,
                    "p_value": p_value_full,
                    "Is_Significant_p<0.05": "Yes" if p_value_full < 0.05 else "No",
                }
            )

        if not confident_df.empty:
            group_correct_confident = confident_df[confident_df["label"] == 0][feature]
            group_halluc_confident = confident_df[confident_df["label"] == 1][feature]
            if len(group_correct_confident) > 1 and len(group_halluc_confident) > 1:
                stat_conf, p_value_conf = mannwhitneyu(
                    group_correct_confident,
                    group_halluc_confident,
                    alternative="two-sided",
                )
                p_text_confident = f"{'p (conf_all)' if plot_suffix == 'All Folds' else 'p (conf)'}: {format_p_value(p_value_conf)}"
                ax.text(
                    0.5,
                    0.88,
                    p_text_confident,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                significance_results.append(
                    {
                        "Test_Data_Scope": f"Confident_{plot_suffix.replace(' ', '_')}",
                        "k_percent": k,
                        "Classifier": model_name,
                        "Model_Combination": combo_name,
                        "Mechanistic_Concept": concept,
                        "Most_Important_Feature": feature,
                        "MannWhitneyU_statistic": stat_conf,
                        "p_value": p_value_conf,
                        "Is_Significant_p<0.05": "Yes" if p_value_conf < 0.05 else "No",
                    }
                )

    for i in range(len(violin_features), len(axes_viol)):
        axes_viol[i].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = os.path.join(
        plot_dir, f"plot_violins_{combo_name}_{plot_suffix.replace(' ', '_')}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close(fig_viol)

    return significance_results


def run_qualitative_analysis(
    all_holdout_data,
    avg_importances_df,
    k,
    top_n_confident,
    output_dir,
    FEATURE_CONCEPT_PREFIXES,
    args,
):
    """
    HEAVILY REFACTORED
    Generates qualitative plots using the aggregated data from all CV folds.
    """
    print(
        f"\n--- Stage 6 (k={k}%): Generating Qualitative Plots and Validations from CV Folds ---"
    )

    all_significance_results = []

    for (model_name, combo_name), fold_data_list in all_holdout_data.items():
        if not fold_data_list:
            continue

        print(f"\n  -> Plotting for Model Combination: {model_name} - {combo_name}")
        model_plot_dir = os.path.join(
            output_dir, "qualitative_plots", model_name.replace(" ", "_")
        )
        os.makedirs(model_plot_dir, exist_ok=True)

        # Single fold analysis (using the first fold)
        single_fold_df = fold_data_list[0]
        single_fold_results = generate_qualitative_plots(
            data_df=single_fold_df,
            avg_importances_df=avg_importances_df,
            model_name=model_name,
            combo_name=combo_name,
            k=k,
            top_n_confident=top_n_confident,
            plot_dir=model_plot_dir,
            FEATURE_CONCEPT_PREFIXES=FEATURE_CONCEPT_PREFIXES,
            plot_suffix="Single_Fold",
        )
        all_significance_results.extend(single_fold_results)

        # Aggregated folds analysis
        aggregated_df = pd.concat(fold_data_list, ignore_index=True)

        # For the aggregated confident plot, we pool confident examples from each fold
        confident_dfs_pooled = []
        for fold_df in fold_data_list:
            correct_df = fold_df[fold_df["label"] == 0]
            halluc_df = fold_df[fold_df["label"] == 1]
            n_samples = min(top_n_confident, len(correct_df), len(halluc_df))
            if n_samples >= 2:
                confident_dfs_pooled.append(
                    correct_df.sort_values("probability", ascending=True).head(
                        n_samples
                    )
                )
                confident_dfs_pooled.append(
                    halluc_df.sort_values("probability", ascending=False).head(
                        n_samples
                    )
                )

        if confident_dfs_pooled:
            aggregated_confident_df = pd.concat(confident_dfs_pooled, ignore_index=True)
            # This logic is now handled inside the helper, but the concept is what matters

        aggregated_fold_results = generate_qualitative_plots(
            data_df=aggregated_df,
            avg_importances_df=avg_importances_df,
            model_name=model_name,
            combo_name=combo_name,
            k=k,
            top_n_confident=top_n_confident,  # top_n is applied per-fold inside helper for pooling
            plot_dir=model_plot_dir,
            FEATURE_CONCEPT_PREFIXES=FEATURE_CONCEPT_PREFIXES,
            plot_suffix="All_Folds",
        )
        all_significance_results.extend(aggregated_fold_results)

    if all_significance_results:
        stats_df = pd.DataFrame(all_significance_results)
        cols_order = [
            "Test_Data_Scope",
            "k_percent",
            "Classifier",
            "Model_Combination",
            "Mechanistic_Concept",
            "Most_Important_Feature",
            "MannWhitneyU_statistic",
            "p_value",
            "Is_Significant_p<0.05",
        ]
        stats_df = stats_df[cols_order]

        save_path = os.path.join(output_dir, "report_feature_significance.csv")
        stats_df.to_csv(save_path, index=False, float_format="%.5f")
        print(f"\n  -> Saved feature significance test results to {save_path}")


def stage7_report_and_visualize_selections(
    all_selections, FEATURE_CONCEPT_PREFIXES, k, output_dir, num_layers, num_heads
):
    """
    Analyzes the components selected during the top-k pruning process across all
    CV folds and creates reports and visualizations of the selection frequencies.
    """
    print(f"\n--- Stage 7 (k={k}%): Analyzing and Visualizing Component Selections ---")
    selection_records = []
    for selection_dict in all_selections:
        for concept, components in selection_dict.items():
            for comp_name in components:
                selection_records.append({"concept": concept, "component": comp_name})
    if not selection_records:
        return

    selection_df = pd.DataFrame(selection_records)
    freq_df = (
        selection_df.groupby(["concept", "component"])
        .size()
        .reset_index(name="selection_count")
    )
    freq_df = freq_df.sort_values(
        ["concept", "selection_count"], ascending=[True, False]
    )
    save_path = os.path.join(output_dir, "report_component_selection_frequency.csv")
    freq_df.to_csv(save_path, index=False)
    print(f"  -> Saved component selection frequency report to {save_path}")

    for concept, prefix in FEATURE_CONCEPT_PREFIXES.items():
        concept_freq = freq_df[freq_df["concept"] == concept]
        if "per_head" in prefix:
            heatmap_data = np.zeros((num_heads, num_layers))
            for _, row in concept_freq.iterrows():
                match = re.search(r"_layer_(\d+)_head_(\d+)", row["component"])
                if match:
                    layer_idx, head_idx = int(match.group(1)), int(match.group(2))
                    if layer_idx < num_layers and head_idx < num_heads:
                        heatmap_data[head_idx, layer_idx] = row["selection_count"]

            plt.figure(figsize=(20, 10))
            sns.heatmap(heatmap_data, cmap="viridis", annot=False)
            plt.title(
                f"Selection Frequency Heatmap for {concept.upper()} (k={k}%)",
                fontsize=16,
            )
            plt.xlabel("Layer")
            plt.ylabel("Head")
            save_path = os.path.join(
                output_dir, f"plot_selection_heatmap_{concept}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved selection heatmap for {concept} to {save_path}")
        else:  # per-layer features
            layer_counts = Counter()
            for _, row in concept_freq.iterrows():
                match = re.search(r"_layer_(\d+)", row["component"])
                if match:
                    layer_counts[int(match.group(1))] += row["selection_count"]

            if not layer_counts:
                continue

            layers = range(num_layers)
            counts = [layer_counts.get(l, 0) for l in layers]

            plt.figure(figsize=(20, 6))
            sns.barplot(x=list(layers), y=counts, color="skyblue")
            plt.title(
                f"Selection Frequency per Layer for {concept.upper()} (k={k}%)",
                fontsize=16,
            )
            plt.xlabel("Layer")
            plt.ylabel("Total Selection Count")
            save_path = os.path.join(
                output_dir, f"plot_selection_barplot_{concept}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved selection bar plot for {concept} to {save_path}")


# ==============================================================================
# --- MAIN SCRIPT ---
# ==============================================================================

if __name__ == "__main__":
    args = config()
    SEED = args.seed
    np.random.seed(SEED)
    random.seed(SEED)

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, random_state=SEED, class_weight="balanced"
                    ),
                ),
            ]
        ),
        "LightGBM": lgb.LGBMClassifier(
            random_state=SEED, class_weight="balanced", verbose=-1
        ),
    }
    if args.run_ebm:
        models["EBM"] = ExplainableBoostingClassifier(random_state=SEED, interactions=0)

    FEATURE_CONCEPT_PREFIXES = {
        "cas": "cas_alt_vs_final_layer_per_head",
        "pfs": "v_ffn_norm_per_layer",
        "bas": "bos_attention_proportion_per_head",
        "pas": "pos_cos_similarity_per_layer",
        "ecs_baseline": "ecs_prompt_final_layer_per_head",
        "pks_baseline": "parameter_knowledge_difference",
    }
    feature_sets_to_build = {
        "Baseline": ["ecs_baseline", "pks_baseline"],
        "Ours 1": ["cas", "pfs"],
        "Ours 2": ["cas", "pfs", "bas", "pas"],
    }

    k = args.k_percent

    print(f"\n{'=' * 80}\nSTARTING FULL ANALYSIS PIPELINE FOR k = {k}%\n{'=' * 80}")
    k_output_dir = os.path.join(args.base_dir, f"ablation_k_{k}")
    os.makedirs(k_output_dir, exist_ok=True)

    # Load data 
    print("Loading all data for the pipeline...")
    full_df = load_full_imbalanced_data(args.results_file, cpu_count(), SEED)
    X_per_layer_full, y_full, NUM_LAYERS, NUM_HEADS = create_per_layer_features(full_df)

    # Create dev split for feature selection
    X_per_layer_dev, _, y_dev, _ = train_test_split(
        X_per_layer_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full
    )
    X_agg_dev = create_aggregate_features(X_per_layer_dev)
    X_agg_full = create_aggregate_features(X_per_layer_full)  # For validation report

    # Stage 1A: Rank all raw components on the dev set
    stage1a_rank_all_components(X_per_layer_dev, y_dev, k_output_dir)

    # Stage 1 & 2: Find the "Dream Team" features using the dev set
    candidate_features = get_candidate_shortlist(
        X_agg_dev, y_dev, models, FEATURE_CONCEPT_PREFIXES, top_n=5
    )
    dream_teams = defaultdict(dict)
    for model_name, model in models.items():
        for combo_name, concepts in feature_sets_to_build.items():
            dream_teams[model_name][combo_name] = find_dream_team_features(
                X_agg_dev, y_dev, model, candidate_features[model_name], concepts
            )
    with open(os.path.join(k_output_dir, "report_dream_teams.json"), "w") as f:
        json.dump(dream_teams, f, indent=4)

    # Stage 3: Validate the selected features on the full dataset
    validate_dream_teams(
        dream_teams, X_agg_full, y_full, k_output_dir, FEATURE_CONCEPT_PREFIXES
    )

    # Stage 4 & 5: Run final CV on the full dataset, get scores, importances, significance, and holdout data
    all_scores, avg_importances_df, all_selections, all_holdout_data = (
        run_final_cv_and_analysis(
            X_per_layer_full,
            y_full,
            models,
            dream_teams,
            k,
            FEATURE_CONCEPT_PREFIXES,
            feature_sets_to_build,
            args,
        )
    )
    final_report_df, avg_importances_df = report_significance_and_importances(
        all_scores, avg_importances_df, models, k_output_dir
    )
    if not final_report_df.empty:
        save_path = os.path.join(k_output_dir, "report_final_significance.csv")
        final_report_df.to_csv(save_path, index=False, float_format="%.4f")
        print("\n--- Final Performance Report ---")
        print(final_report_df.to_string())

    # Stage 6: Qualitative Analysis using the data collected from the CV folds
    run_qualitative_analysis(
        all_holdout_data=all_holdout_data,
        avg_importances_df=avg_importances_df,
        k=k,
        top_n_confident=args.top_n_confident,
        output_dir=k_output_dir,
        FEATURE_CONCEPT_PREFIXES=FEATURE_CONCEPT_PREFIXES,
        args=args,
    )

    # Stage 7: Analyze and visualize component selections
    stage7_report_and_visualize_selections(
        all_selections, FEATURE_CONCEPT_PREFIXES, k, k_output_dir, NUM_LAYERS, NUM_HEADS
    )

    print(f"\n{'=' * 80}\nCOMPLETED FULL ANALYSIS PIPELINE FOR k = {k}%\n{'=' * 80}")
