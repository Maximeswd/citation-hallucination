# ==============================================================================
# --- IMPORTS & SETUP ---
# ==============================================================================
import json
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm
from scipy.stats import ttest_ind
import sys
import re
from scipy.stats import pearsonr
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.utils import resample


print(
    "Definitive analysis script with FULLY REPRODUCIBLE parallel data loading and robust parameters."
)

# ==============================================================================
# --- GLOBAL CONFIGURATION & SEEDING FOR REPRODUCIBILITY ---
# ==============================================================================


def config():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a specific phase of the feature analysis."
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=[
            "12_feature_rank_basic",
            "12_feature_rank_comprehensive",
            "ecs_pks_showdown_comprehensive",
            "cas_pos_bos_showdown_comprehensive",
            "norms_showdown_comprehensive",
            "19_ultimate_showdown",
            "20_final_layer_analysis",
            "21_last_two_layers_analysis",
            "22_sink_and_stability_showdown",
            "23_qualitative_analysis",
            "24_best_feature_planes",
            "25_confident_prediction_planes",
            "26_significance_testing",  # <-- ADDED
            "generic_showdown",
            "all",
        ],
        help="The analysis phase to run.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="hallucination_analysis_final",
        help="Base directory for all output files.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="/exp/mdassen/redeep/ReDEeP-ICLR-main/ReDeEP/log/llama3.2-3b_scale/llama3.2-3b_scale_gpa/not_rel/full/scale_results_full.jsonl",
        help="Path to the full scores file.",
    )
    parser.add_argument(
        "--run_ebm", action="store_true", help="If set, train the EBM model."
    )
    parser.add_argument(
        "--selector",
        type=str,
        default="sfs",
        choices=["sfs", "rfe"],
        help="Feature selection method to use.",
    )
    parser.add_argument(
        "--feature_groups",
        nargs="*",
        default=[],
        help="List of feature group keywords for the generic_showdown phase.",
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="Feature selection method to use."
    )
    return parser.parse_args()


args = config()
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
_CACHED_DATA = {}


# ==============================================================================
# --- REPRODUCIBLE PARALLEL DATA LOADING & FEATURE ENGINEERING  ---
# ==============================================================================
def seed_worker(worker_seed):
    """Initializer function to seed each worker process."""
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parallel_index_worker(args):
    """Worker for the parallel indexing pass"""
    filepath, chunk_start, chunk_end = args
    local_index = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(chunk_start)
        if chunk_start > 0:
            f.readline()
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
    """Worker function: Reads lines for its assigned source_ids and samples them"""
    chunk_tokens = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for source_id in source_ids_chunk:
            all_tokens_for_source = []
            for offset in byte_offset_map.get(source_id, []):
                try:
                    f.seek(offset)
                    line = f.readline()
                    doc = json.loads(line)
                    tokens_from_doc = doc.get("token_data", [])

                    for token_data in tokens_from_doc:
                        token_data["source_id"] = doc.get("source_id")

                    all_tokens_for_source.extend(tokens_from_doc)
                except (json.JSONDecodeError, IndexError):
                    continue

            factual_samples = [t for t in all_tokens_for_source if t.get("label") == 0]
            hallucinated_samples = [
                t for t in all_tokens_for_source if t.get("label") == 1
            ]
            sample_size = min(len(factual_samples), len(hallucinated_samples))
            if sample_size > 0:
                chunk_tokens.extend(random.sample(hallucinated_samples, sample_size))
                chunk_tokens.extend(random.sample(factual_samples, sample_size))
    return chunk_tokens


def deterministic_chunking(data, num_chunks):
    """A deterministic chunking function"""
    k, m = divmod(len(data), num_chunks)
    return (
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)
    )


def load_data_optimized(filepath: str, num_workers: int) -> pd.DataFrame:
    """Loads data using the with reproducible workers."""
    if "raw_df" in _CACHED_DATA:
        return _CACHED_DATA["raw_df"]
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
            pool.imap_unordered(parallel_index_worker, pool_args),
            total=len(pool_args),
            desc="Indexing file chunks",
        ):
            for source_id, offsets in local_index.items():
                byte_offset_map[source_id].extend(offsets)

    unique_source_ids = sorted(list(byte_offset_map.keys()))

    print(
        f"\nPass 2: Processing {len(unique_source_ids)} source_id groups in parallel..."
    )
    source_id_chunks = list(deterministic_chunking(unique_source_ids, num_workers))

    worker_func = partial(
        worker_process_source_group,
        byte_offset_map=byte_offset_map,
        jsonl_path=filepath,
    )
    all_sampled_tokens = []
    with Pool(processes=num_workers, initializer=seed_worker, initargs=(SEED,)) as pool:
        for tokens_from_chunk in tqdm(
            pool.imap(worker_func, source_id_chunks),
            total=len(source_id_chunks),
            desc="Processing data chunks",
        ):
            all_sampled_tokens.extend(tokens_from_chunk)
    if not all_sampled_tokens:
        raise ValueError("No token data was sampled.")
    random.shuffle(all_sampled_tokens)
    df = pd.DataFrame(all_sampled_tokens)
    _CACHED_DATA["raw_df"] = df
    return df


def create_per_layer_features(df: pd.DataFrame):
    """Flattens the raw JSON data into a per-layer feature matrix."""
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    sample_token = df.iloc[0]
    score_keys = [
        k
        for k, v in sample_token.items()
        if isinstance(v, list) and v and isinstance(v[0], (int, float))
    ]
    num_layers = next(
        (len(sample_token[key]) for key in score_keys if "per_layer" in key), 0
    )
    if num_layers == 0:
        raise ValueError("Could not determine number of layers")
    cas_scores = sample_token.get("cas_vs_final_layer_per_head", [])
    num_heads = len(cas_scores) // num_layers if cas_scores and num_layers > 0 else 0
    processed_records = []
    for _, token in tqdm(
        df.iterrows(), total=df.shape[0], desc="Engineering per-layer features"
    ):
        features = {}
        for key in score_keys:
            scores = token.get(key)
            if not isinstance(scores, list) or not scores:
                continue
            is_per_head = num_heads > 0 and len(scores) == num_layers * num_heads
            if is_per_head:
                scores_reshaped = np.array(scores).reshape(num_layers, num_heads)
                for i in range(num_layers):
                    features[f"{key}_mean_layer_{i}"] = np.mean(scores_reshaped[i])
                    features[f"{key}_max_layer_{i}"] = np.max(scores_reshaped[i])
                    features[f"{key}_std_layer_{i}"] = np.std(scores_reshaped[i])
            elif len(scores) == num_layers:
                for i in range(num_layers):
                    features[f"{key}_layer_{i}"] = scores[i]
        processed_records.append(features)
    return pd.DataFrame(processed_records).fillna(0), df["label"]


def create_basic_aggregate_features(df_per_layer: pd.DataFrame):
    """Creates basic aggregate features (mean and max)"""
    all_feature_dfs = []
    prefixes = set(
        re.match(r"^(.*_layer_)", col).group(1)
        for col in df_per_layer.columns
        if re.match(r"^(.*_layer_)", col)
    )
    for prefix in tqdm(prefixes, desc="Engineering basic features"):
        metric_cols = sorted(
            [c for c in df_per_layer.columns if c.startswith(prefix)],
            key=lambda x: int(re.search(r"_(\d+)$", x).group(1)),
        )
        if metric_cols:
            all_feature_dfs.append(
                pd.DataFrame(
                    {
                        prefix + "mean": df_per_layer[metric_cols].mean(axis=1),
                        prefix + "max": df_per_layer[metric_cols].max(axis=1),
                    }
                )
            )
    return pd.concat(all_feature_dfs, axis=1).fillna(0)


def create_aggregate_features(df_per_layer: pd.DataFrame):
    """Create set of features including slope and Fourier features"""
    all_feature_dfs = []
    prefixes = set(
        re.match(r"^(.*_layer_)", col).group(1)
        for col in df_per_layer.columns
        if re.match(r"^(.*_layer_)", col)
    )
    for prefix in tqdm(prefixes, desc="Engineering aggregate features"):
        metric_cols = sorted(
            [c for c in df_per_layer.columns if c.startswith(prefix)],
            key=lambda x: int(re.search(r"_(\d+)$", x).group(1)),
        )
        if len(metric_cols) < 4:
            continue
        metric_data = df_per_layer[metric_cols]
        x = np.arange(metric_data.shape[1])
        prefix_features = {
            prefix + "mean": metric_data.mean(axis=1),
            prefix + "std": metric_data.std(axis=1),
            prefix + "max": metric_data.max(axis=1),
            prefix + "sum": metric_data.sum(axis=1),
            prefix + "slope_full": np.polyfit(x, metric_data.T, 1)[0],
        }
        deltas = metric_data.diff(axis=1).iloc[:, 1:]
        prefix_features.update(
            {
                prefix + "delta_mean": deltas.mean(axis=1),
                prefix + "delta_max": deltas.max(axis=1),
                prefix + "delta_sum": deltas.sum(axis=1),
                prefix + "delta_std": deltas.std(axis=1),
            }
        )
        fft_coeffs = np.fft.rfft(metric_data, axis=1)
        for i in range(1, 7):
            if fft_coeffs.shape[1] > i:
                prefix_features[f"{prefix}fft_mag_{i}"] = np.abs(fft_coeffs[:, i])
        all_feature_dfs.append(pd.DataFrame(prefix_features))
    return pd.concat(all_feature_dfs, axis=1).fillna(0)


def create_final_layer_features(df: pd.DataFrame):
    """Engineers features using only data from the final layer"""
    print("Starting FINAL LAYER ONLY feature engineering...")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    sample_token = df.iloc[0]
    score_keys = [
        k
        for k, v in sample_token.items()
        if isinstance(v, list) and v and isinstance(v[0], (int, float))
    ]
    num_layers = next(
        (len(sample_token[key]) for key in score_keys if "per_layer" in key), 0
    )
    if num_layers == 0:
        raise ValueError("Could not determine number of layers.")
    cas_scores = sample_token.get("cas_vs_final_layer_per_head", [])
    num_heads = len(cas_scores) // num_layers if cas_scores and num_layers > 0 else 0
    processed_records = []
    for _, token in tqdm(
        df.iterrows(), total=df.shape[0], desc="Engineering final-layer features"
    ):
        features = {}
        for key in score_keys:
            scores = token.get(key)
            if not isinstance(scores, list) or not scores:
                continue
            is_per_head = num_heads > 0 and len(scores) == num_layers * num_heads
            if is_per_head:
                final_layer_scores = np.array(scores).reshape(num_layers, num_heads)[
                    -1, :
                ]
                features[f"{key}_FINAL_LAYER_mean_across_heads"] = np.mean(
                    final_layer_scores
                )
                features[f"{key}_FINAL_LAYER_max_across_heads"] = np.max(
                    final_layer_scores
                )
                features[f"{key}_FINAL_LAYER_std_across_heads"] = np.std(
                    final_layer_scores
                )
            elif len(scores) == num_layers:
                features[f"{key}_FINAL_LAYER"] = scores[-1]
        processed_records.append(features)
    return pd.DataFrame(processed_records).fillna(0), df["label"]


def create_last_n_layers_features(df: pd.DataFrame, n: int):
    """Engineers features using data from the last 'n' layers"""
    print(f"Starting LAST {n} LAYERS feature engineering...")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    sample_token = df.iloc[0]
    score_keys = [
        k
        for k, v in sample_token.items()
        if isinstance(v, list) and v and isinstance(v[0], (int, float))
    ]
    num_layers = next(
        (len(sample_token[key]) for key in score_keys if "per_layer" in key), 0
    )
    if num_layers == 0:
        raise ValueError("Could not determine number of layers.")
    cas_scores = sample_token.get("cas_vs_final_layer_per_head", [])
    num_heads = len(cas_scores) // num_layers if cas_scores and num_layers > 0 else 0
    processed_records = []
    for _, token in tqdm(
        df.iterrows(), total=df.shape[0], desc=f"Engineering last {n} layers"
    ):
        features = {}
        for key in score_keys:
            scores = token.get(key)
            if not isinstance(scores, list) or not scores:
                continue
            is_per_head = num_heads > 0 and len(scores) == num_layers * num_heads
            if is_per_head:
                last_n_layers_scores = np.array(scores).reshape(num_layers, num_heads)[
                    -n:, :
                ]
                for i in range(n):
                    actual_layer_idx = num_layers - n + i
                    layer_scores = last_n_layers_scores[i, :]
                    features[f"{key}_layer_{actual_layer_idx}_mean_across_heads"] = (
                        np.mean(layer_scores)
                    )
                    features[f"{key}_layer_{actual_layer_idx}_max_across_heads"] = (
                        np.max(layer_scores)
                    )
                    features[f"{key}_layer_{actual_layer_idx}_std_across_heads"] = (
                        np.std(layer_scores)
                    )
            elif len(scores) == num_layers:
                for i in range(n):
                    actual_layer_idx = num_layers - n + i
                    features[f"{key}_layer_{actual_layer_idx}"] = scores[-n:][i]
        processed_records.append(features)
    return pd.DataFrame(processed_records).fillna(0), df["label"]


def get_engineered_data(args, feature_mode="comprehensive"):
    """Central data pipeline that caches results for different feature modes."""
    cache_key = f"data_{feature_mode}"
    if cache_key in _CACHED_DATA:
        return _CACHED_DATA[cache_key]
    raw_df = load_data_optimized(args.results_file, cpu_count())
    X_per_layer, y = create_per_layer_features(raw_df)
    X_aggregate = (
        create_basic_aggregate_features(X_per_layer)
        if feature_mode == "basic"
        else create_aggregate_features(X_per_layer)
    )
    _CACHED_DATA[cache_key] = (X_aggregate, y)
    return X_aggregate, y


# ==============================================================================
# --- MODELING, EVALUATION & PLOTTING ---
# ==============================================================================
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """Trains and evaluates a set of models, returning results with keys"""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_probas = model.predict_proba(X_test)[:, 1]
        preds = (pred_probas > 0.5).astype(int)
        results[name] = {
            "model": model,
            "auc": roc_auc_score(y_test, pred_probas),
            "pcc": pearsonr(y_test, pred_probas)[0],
            "accuracy": accuracy_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "y_probas": pred_probas,
        }
    return results


def get_bootstrap_ci(
    y_true,
    scores_model_A,
    scores_model_B,
    metric_func,
    metric_type="proba",
    n_bootstraps=1000,
):
    """
    Calculates the 95% CI for the difference between two models' performance using bootstrapping.

    Args:
        y_true (np.array): True labels.
        scores_model_A (np.array): Predictions or probabilities from model A.
        scores_model_B (np.array): Predictions or probabilities from model B.
        metric_func (callable): The metric function (e.g., roc_auc_score, f1_score).
        metric_type (str): Type of score. 'proba' for probabilities, 'pred' for binary predictions.
        n_bootstraps (int): Number of bootstrap iterations.

    Returns:
        tuple: Lower and upper bounds of the 95% confidence interval.
    """
    n_samples = len(y_true)
    differences = []

    for _ in range(n_bootstraps):
        indices = resample(np.arange(n_samples))
        if len(np.unique(y_true[indices])) < 2:
            continue

        y_sample = y_true[indices]
        scores_A_sample = scores_model_A[indices]
        scores_B_sample = scores_model_B[indices]

        if metric_type == "pred":
            metric_A = metric_func(y_sample, scores_A_sample)
            metric_B = metric_func(y_sample, scores_B_sample)
        else:  # 'proba'
            metric_A = metric_func(y_sample, scores_A_sample)
            metric_B = metric_func(y_sample, scores_B_sample)

        differences.append(metric_B - metric_A)

    lower = np.percentile(differences, 2.5)
    upper = np.percentile(differences, 97.5)
    return (lower, upper)


def plot_roc_curves(results, y_test, save_path, plot_title):
    """
    Plots and saves ROC curves for a dictionary of model results
    """
    custom_colors = [
        "#FF7F0E",
        "#359c35",
        "#905bc1",
        "#2589d1",
        "#D32B2B",
        "#359c35",
        "#2589d1",
        "#9768c3",
        "#FF7F0E",
        "#D32B2B",
        "#e377c2",
        "#905bc1",
        "#f02727",
        "#fa8e2f",
        "#2589d1",
        "#9768c3",
        "#20c9dc",
        "#3ede3e#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#2589d1",
        "#D62728",
        "#FF7F0E",
        "#FFC300",
    ]
    markers = ["o", "X", "s", "^", "P", "*", "D", "v", "<", ">"]

    title_fontsize = 43
    axis_label_fontsize = 40
    legend_fontsize = 40
    tick_fontsize = 40
    line_thickness = 10
    marker_size = 22

    plt.figure(figsize=(24, 22))
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=2,
        label="Random (AUC = 0.500)",
    )

    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result["y_probas"])

        color = custom_colors[i % len(custom_colors)]
        marker = markers[i % len(markers)]

        plt.plot(
            fpr,
            tpr,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=line_thickness,
            markersize=marker_size,
            label=f"{name} (AUC={result['auc']:.3f})",
        )

    plt.xlabel("False Positive Rate", fontsize=axis_label_fontsize)
    plt.ylabel("True Positive Rate", fontsize=axis_label_fontsize)
    plt.title(plot_title, fontsize=title_fontsize, pad=20)
    plt.legend(fontsize=legend_fontsize, loc="lower right")
    plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_best_feature_plane(
    args, models, feature_pairs, base_rank_phase="12_feature_rank_comprehensive"
):
    """
    Creates 2D density plots for the best performing engineered features for specified pairs of raw score types (e.g., v_ffn vs. pos_score).
    """
    phase_name = "24_best_feature_planes"
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Phase 24: Best Feature Plane Plotting ---\n"
        + "=" * 80
    )

    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")
    df_for_plotting = X_aggregate.copy()
    df_for_plotting["label"] = y

    for model_name in models.keys():
        print(f"\n--- Generating planes for model: {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir, base_rank_phase, f"summary_single_features_{safe_name}.csv"
        )

        try:
            ranked_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(f"ERROR: Prerequisite summary for {model_name} not found. Skipping.")
            continue

        for (x_keyword, y_keyword), (x_label, y_label) in feature_pairs.items():
            x_df = ranked_df[ranked_df.Feature.str.contains(x_keyword)]
            if x_df.empty:
                continue
            best_x_feature = x_df.iloc[0]["Feature"]

            y_df = ranked_df[ranked_df.Feature.str.contains(y_keyword)]
            if y_df.empty:
                continue
            best_y_feature = y_df.iloc[0]["Feature"]

            print(f"  -> Plotting Plane: '{x_label}' vs. '{y_label}'")

            plt.figure(figsize=(11, 9))

            correct_data = df_for_plotting[df_for_plotting["label"] == 0]
            halluc_data = df_for_plotting[df_for_plotting["label"] == 1]

            sns.kdeplot(
                data=correct_data,
                x=best_x_feature,
                y=best_y_feature,
                cmap="Blues",
                fill=True,
                thresh=0.1,
            )
            sns.kdeplot(
                data=halluc_data,
                x=best_x_feature,
                y=best_y_feature,
                cmap="Oranges",
                fill=True,
                thresh=0.1,
            )

            plt.title(
                f"{x_label} vs. {y_label}\n(Best Features for {model_name})",
                fontsize=22,
                pad=20,
            )
            plt.xlabel(f"Best '{x_label}' Feature", fontsize=18)
            plt.ylabel(f"Best '{y_label}' Feature", fontsize=18)
            plt.grid(True, linestyle="--", alpha=0.6)

            legend_handles = [
                Patch(facecolor=sns.color_palette("Blues")[3], label="Correct"),
                Patch(facecolor=sns.color_palette("Oranges")[3], label="Hallucinated"),
            ]

            plt.legend(handles=legend_handles, fontsize=16)
            plt.tight_layout()
            plot_filename = f"plane_{x_keyword}_vs_{y_keyword}_{safe_name}.png"
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
            plt.close()


def run_standard_analysis_protocol(X, y, models, output_dir, phase_name):
    """A universal function that performs complete standardized analysis"""
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Standard Analysis for: {phase_name} ---\n"
        + "=" * 80
    )
    if X.empty:
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"--- Part 1: Evaluating {len(X.columns)} Combined Features ---")
    combined_results = train_and_evaluate(models, X_train, y_train, X_test, y_test)
    plot_roc_curves(
        combined_results,
        y_test,
        os.path.join(output_dir, "roc_curves_combined.png"),
        f"ROC - {phase_name} (Combined)",
    )

    print(
        f"\n--- Part 2: Evaluating {len(X.columns)} Single Features Individually for Each Model ---"
    )
    for model_name, model in models.items():
        single_feature_performance = []
        for feature in tqdm(X.columns, desc=f"Analyzing with {model_name}"):
            try:
                results = train_and_evaluate(
                    {model_name: clone(model)},
                    X_train[[feature]],
                    y_train,
                    X_test[[feature]],
                    y_test,
                )
                if model_name in results:
                    performance_dict = {
                        "Feature": feature,
                        **{
                            k: v
                            for k, v in results[model_name].items()
                            if k not in ["model", "y_probas"]
                        },
                    }
                    single_feature_performance.append(performance_dict)
            except Exception as e:
                print(f"Failed on feature {feature}: {e}")

        if single_feature_performance:
            safe_name = model_name.replace(" ", "_")
            df = pd.DataFrame(single_feature_performance).sort_values(
                by=["auc", "Feature"], ascending=[False, True]
            )
            df.to_csv(
                os.path.join(output_dir, f"summary_single_features_{safe_name}.csv"),
                index=False,
            )


def run_feature_showdown(
    args, models, phase_name, feature_groups, feature_mode, base_rank_phase
):
    """Runner for testing feature combinations that now saves detailed CSV reports."""
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Feature Showdown: {phase_name} ({feature_mode} features) ---\n"
        + "=" * 80
    )
    X_aggregate, y = get_engineered_data(args, feature_mode=feature_mode)
    X_train, X_test, y_train, y_test = train_test_split(
        X_aggregate, y, test_size=0.2, random_state=SEED, stratify=y
    )

    for model_name, model in models.items():
        print(f"\n--- Building and Testing Combinations for {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir, base_rank_phase, f"summary_single_features_{safe_name}.csv"
        )
        try:
            summary_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(f"ERROR: Prerequisite summary for {model_name} not found.")
            continue

        best_features = {}
        for group, keywords in feature_groups.items():
            filtered_df = summary_df[
                summary_df["Feature"].str.contains("|".join(keywords), regex=True)
            ]
            if not filtered_df.empty:
                best_features[group] = filtered_df.iloc[0]["Feature"]

        if not best_features:
            continue

        all_results_for_model, summary_rows_for_model = {}, []

        for group_name, feature_name in best_features.items():
            results = train_and_evaluate(
                {model_name: clone(model)},
                X_train[[feature_name]],
                y_train,
                X_test[[feature_name]],
                y_test,
            )
            if model_name in results:
                all_results_for_model[group_name] = results[model_name]
                summary_rows_for_model.append(
                    {
                        "Combination": group_name,
                        "Features": feature_name,
                        **{
                            k: v
                            for k, v in results[model_name].items()
                            if k not in ["model", "y_probas"]
                        },
                    }
                )

        for i in range(2, len(best_features) + 1):
            for combo in combinations(best_features.values(), i):
                combo_key = " & ".join(
                    sorted([k for k, v in best_features.items() if v in combo])
                )
                results = train_and_evaluate(
                    {model_name: clone(model)},
                    X_train[list(combo)],
                    y_train,
                    X_test[list(combo)],
                    y_test,
                )
                if model_name in results:
                    all_results_for_model[combo_key] = results[model_name]
                    summary_rows_for_model.append(
                        {
                            "Combination": combo_key,
                            "Features": str(list(combo)),
                            **{
                                k: v
                                for k, v in results[model_name].items()
                                if k not in ["model", "y_probas"]
                            },
                        }
                    )

        plot_roc_curves(
            all_results_for_model,
            y_test,
            os.path.join(output_dir, f"roc_showdown_{safe_name}.png"),
            f"Showdown - {phase_name} - {model_name}",
        )
        pd.DataFrame(summary_rows_for_model).sort_values(
            by="auc", ascending=False
        ).to_csv(
            os.path.join(output_dir, f"summary_showdown_{safe_name}.csv"), index=False
        )
        print(f"  -> Saved ROC plot and detailed CSV summary for {model_name}.")


# ==============================================================================
# --- UNIFIED & SELECTABLE FEATURE SELECTION ---
# ==============================================================================


def _find_optimal_sfs(estimator, X, y, max_features, is_pipeline):
    """Helper for Sequential Forward Selection (SFS): basically a forward feature selection method"""
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sfs = SFS(
        estimator=estimator,
        k_features=(3, max_features),
        forward=True,
        floating=False,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
    )
    if is_pipeline:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        sfs.fit(X_scaled, y)
    else:
        sfs.fit(X.values, y)
    best_feature_indices = list(sfs.k_feature_idx_)
    return X.columns[best_feature_indices].tolist()


def _find_optimal_rfe(estimator, X, y, n_features_to_select):
    """Helper for Recursive Feature Elimination (RFE): basically a backward feature selection method"""
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    return X.columns[rfe.support_].tolist()


def find_optimal_feature_set(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    candidate_features,
    method="sfs",
    n_features=5,
):
    """Finds the best feature set using a selectable method (SFS or RFE)."""
    print(
        f"\n--- Finding optimal {n_features}-feature set for {model_name} using {method.upper()} ---"
    )

    if model_name == "EBM":
        print("  -> EBM uses its internal importance scores for selection.")
        temp_model = clone(model)
        temp_model.fit(X_train[candidate_features], y_train)
        ebm_global = temp_model.explain_global()
        best_features = (
            pd.DataFrame(
                {
                    "Feature": ebm_global.data()["names"],
                    "Importance": ebm_global.data()["scores"],
                }
            )
            .sort_values(by="Importance", ascending=False)["Feature"]
            .head(n_features)
            .tolist()
        )
    else:
        estimator = (
            model.named_steps["classifier"] if isinstance(model, Pipeline) else model
        )
        is_pipeline = isinstance(model, Pipeline)

        if method == "sfs":
            best_features = _find_optimal_sfs(
                clone(estimator),
                X_train[candidate_features],
                y_train,
                max_features=n_features,
                is_pipeline=is_pipeline,
            )
        elif method == "rfe":
            best_features = _find_optimal_rfe(
                clone(estimator),
                X_train[candidate_features],
                y_train,
                n_features_to_select=n_features,
            )
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    final_results = train_and_evaluate(
        {model_name: clone(model)},
        X_train[best_features],
        y_train,
        X_test[best_features],
        y_test,
    )
    print(f"  -> Optimal set found: {best_features}")
    print(
        f"  -> Final Test AUC on selected features: {final_results[model_name]['auc']:.4f}"
    )
    return best_features, final_results[model_name]


def create_and_save_distribution_plot(
    df, feature_name, friendly_name, output_dir, model_name_for_filename
):
    """
    Creates and saves a violin plot for a given feature's distribution.
    """
    if friendly_name == "POS":
        y_label = "Pathway Orthogonality Score (Higher = Less Conflict)"
    elif friendly_name == "CAS":
        y_label = "Contextual Alignment Score (Higher = More Aligned)"
    elif friendly_name == "BOS_Attention":
        y_label = "BOS Attention (Higher = Less Context Focus)"
    elif friendly_name == "V_ffn_Norm":
        y_label = "FFN Update Norm (Higher = More Parametric Force)"
    else:
        y_label = friendly_name

    plt.figure(figsize=(8, 6))

    sns.violinplot(
        x="label",
        y=feature_name,
        data=df,
        inner="box",
        hue="label",
        palette={0: "#4C72B0", 1: "#DD8452"},
        legend=False,
    )

    sns.stripplot(
        x="label",
        y=feature_name,
        data=df,
        jitter=True,
        color="black",
        size=2,
        alpha=0.3,
    )

    plt.title(
        f"Distribution of {friendly_name}\n({model_name_for_filename})",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Token Label", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(ticks=[0, 1], labels=["Correct Citation", "Hallucination"])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    safe_model_name = model_name_for_filename.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(
        output_dir, f"distribution_{friendly_name}_{safe_model_name}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Distribution plot saved to: {save_path}")


def create_combined_distribution_plot(df, feature_mapping, output_dir, model_name):
    """
    Creates a plot showing the distributions for all PACT scores
    """

    id_vars = ["label"]
    value_vars = list(feature_mapping.keys())
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="PACT_Probe",
        value_name="Score",
    )
    palette = {0: "#2390CA", 1: "#DE6A05"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)

    for i, probe_name in enumerate(value_vars):
        ax = axes[i]
        subset_df = df_long[df_long["PACT_Probe"] == probe_name]

        sns.violinplot(
            data=subset_df,
            x="label",
            y="Score",
            hue="label",
            palette=palette,
            inner="box",
            legend=False,
            ax=ax,
        )

        sns.stripplot(
            data=subset_df,
            x="label",
            y="Score",
            jitter=True,
            color="black",
            size=2,
            alpha=0.15,
            ax=ax,
        )

        ax.set_title(probe_name, fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Hallucination"], fontsize=12)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        if i == 0:
            ax.set_ylabel("Probe Score", fontsize=12)
        else:
            ax.set_ylabel("")

    fig.suptitle(
        f"PACT Probe Distributions for High-Confidence Predictions ({model_name})",
        fontsize=18,
    )
    fig.supxlabel("Token Label", fontsize=14, y=0.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f"combined_distribution_{safe_model_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Advanced combined distribution plot saved to: {save_path}")


def plot_confident_prediction_planes(
    args,
    models,
    feature_pairs,
    base_rank_phase="12_feature_rank_comprehensive",
    top_percent=30,
):
    """
    Creates 2D density plots for the best performing features, using ONLY the most 
    confident predictions (top X percent) made by each model.
    """
    phase_name = f"25_paper_confident_prediction_planes_{top_percent}percent"
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Phase 25: Confident Prediction Plane Plotting (Top {top_percent}%) ---\n"
        + "=" * 80
    )

    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")

    for model_name, model in models.items():
        print(f"\n--- Generating confident planes for model: {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir, base_rank_phase, f"summary_single_features_{safe_name}.csv"
        )

        try:
            ranked_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(f"ERROR: Prerequisite summary for {model_name} not found. Skipping.")
            continue

        feature_keywords = [
            "ecs_final",
            "ecs_prompt_final",
            "parameter_knowledge",
            "v_ffn_norm",
            "bos_attention",
            "pos_score",
        ]
        best_features = [
            ranked_df[ranked_df["Feature"].str.contains(kw, regex=False)].iloc[0][
                "Feature"
            ]
            for kw in feature_keywords
            if not ranked_df[ranked_df["Feature"].str.contains(kw, regex=False)].empty
        ]
        if len(best_features) != len(feature_keywords):
            print(
                "Could not find all required features for probability model. Aborting for this model."
            )
            continue

        # Train on all data to get a probability score for every sample for this analysis
        final_model = clone(model)
        final_model.fit(X_aggregate[best_features], y)
        all_probas = final_model.predict_proba(X_aggregate[best_features])[:, 1]

        temp_df_for_selection = X_aggregate.copy()
        temp_df_for_selection["label"] = y
        temp_df_for_selection["prediction_probability"] = all_probas

        # Separate by class to calculate percentages correctly
        correct_df = temp_df_for_selection[temp_df_for_selection["label"] == 0]
        halluc_df = temp_df_for_selection[temp_df_for_selection["label"] == 1]

        # Calculate the number of samples to take for each class
        n_correct = int(len(correct_df) * (top_percent / 100.0))
        n_halluc = int(len(halluc_df) * (top_percent / 100.0))

        # Select the most confident predictions based on the calculated number
        confident_hallucinations = halluc_df.sort_values(
            by="prediction_probability", ascending=False
        ).head(n_halluc)
        
        confident_correct = correct_df.sort_values(
            by="prediction_probability", ascending=True
        ).head(n_correct)

        confident_df = pd.concat([confident_hallucinations, confident_correct])

        for (x_keyword, y_keyword), (x_label, y_label) in feature_pairs.items():
            x_df = ranked_df[ranked_df.Feature.str.contains(x_keyword)]
            if x_df.empty:
                continue
            best_x_feature = x_df.iloc[0]["Feature"]

            y_df = ranked_df[ranked_df.Feature.str.contains(y_keyword)]
            if y_df.empty:
                continue
            best_y_feature = y_df.iloc[0]["Feature"]

            print(f"  -> Plotting Confident Plane: '{x_label}' vs. '{y_label}'")

            title_fontsize = 28
            subtitle_fontsize = 20
            axis_label_fontsize = 24
            legend_fontsize = 20
            tick_fontsize = 16

            fig, ax = plt.subplots(figsize=(12, 10))

            correct_data = confident_df[confident_df["label"] == 0]
            halluc_data = confident_df[confident_df["label"] == 1]

            sns.kdeplot(
                data=correct_data,
                x=best_x_feature,
                y=best_y_feature,
                cmap="Blues",
                fill=True,
                thresh=0.1,
                ax=ax,
            )
            sns.kdeplot(
                data=halluc_data,
                x=best_x_feature,
                y=best_y_feature,
                cmap="Oranges",
                fill=True,
                thresh=0.1,
                ax=ax,
            )

            ax.set_title(
                f"{x_label} vs. {y_label} \n Model: {model_name}",
                fontsize=subtitle_fontsize + 4,
                pad=20,
            )
            ax.set_xlabel(x_label, fontsize=axis_label_fontsize, labelpad=15)
            ax.set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=15)
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
            ax.grid(True, linestyle="--", alpha=0.6)
            legend_handles = [
                Patch(facecolor=sns.color_palette("Blues")[3], label="Correct"),
                Patch(facecolor=sns.color_palette("Oranges")[3], label="Hallucinated"),
            ]
            ax.legend(handles=legend_handles, fontsize=legend_fontsize)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            plot_filename = (
                f"confident_plane_{x_keyword}_vs_{y_keyword}_{safe_name}.png"
            )
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
            plt.close(fig)


# ==============================================================================
# --- PHASE-SPECIFIC RUNNERS ---
# ==============================================================================


def run_specific_combinations_plot(
    args, models, phase_name, base_rank_phase="12_feature_rank_comprehensive"
):
    """
    Runs an analysis on a predefined, specific set of feature combinations and generates a single ROC-AUC plot per model comparing them
    """
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Specific Combination Showdown: {phase_name} ---\n"
        + "=" * 80
    )

    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")
    X_train, X_test, y_train, y_test = train_test_split(
        X_aggregate, y, test_size=0.2, random_state=SEED, stratify=y
    )

    for model_name, model in models.items():
        print(f"\n--- Processing model: {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir, base_rank_phase, f"summary_single_features_{safe_name}.csv"
        )

        try:
            summary_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(
                f"ERROR: Prerequisite summary for {model_name} not found at {source_summary_path}. Skipping model."
            )
            continue

        def get_best_feature(keyword):
            filtered_df = summary_df[
                summary_df["Feature"].str.contains(keyword, regex=False)
            ]
            if not filtered_df.empty:
                return filtered_df.iloc[0]["Feature"]
            else:
                print(
                    f"WARNING: No feature found for keyword '{keyword}' for model {model_name}."
                )
                return None

        # Find best features for all components
        best_cas_feature = get_best_feature("ecs_final")
        best_vffn_feature = get_best_feature("v_ffn_norm")
        best_bos_feature = get_best_feature("bos_attention")
        best_pos_feature = get_best_feature("pos_score")
        best_ecs_feature = get_best_feature("ecs_prompt_final")
        best_pks_feature = get_best_feature("parameter_knowledge")

        print("  -> Best features identified:")
        print(f"     - CAS (ecs_final): {best_cas_feature}")
        print(f"     - V_ffn: {best_vffn_feature}")
        print(f"     - BOS: {best_bos_feature}")
        print(f"     - POS: {best_pos_feature}")
        print(f"     - ECS (ecs_prompt_final): {best_ecs_feature}")
        print(f"     - PKS (parameter_knowledge): {best_pks_feature}")

        combinations_to_test = [
            ("Baseline", [best_ecs_feature, best_pks_feature]),  # ECS + PKS
            ("Ours 1", [best_cas_feature, best_vffn_feature]),  # CAS + V_ffn
            (
                "Ours 2",
                [
                    best_bos_feature,
                    best_pos_feature,
                    best_cas_feature,
                    best_vffn_feature,
                ],
            ),  # BOS + POS + CAS + V_ffn
        ]
        all_results_for_model = {}
        summary_rows = []

        for combo_name, feature_list in tqdm(
            combinations_to_test, desc=f"Evaluating combinations for {model_name}"
        ):
            feature_list = [f for f in feature_list if f is not None]
            if not feature_list:
                print(f"Skipping '{combo_name}' as its features could not be found.")
                continue

            results = train_and_evaluate(
                {model_name: clone(model)},
                X_train[feature_list],
                y_train,
                X_test[feature_list],
                y_test,
            )

            if model_name in results:
                all_results_for_model[combo_name] = results[model_name]
                summary_data = {
                    "Combination": combo_name,
                    "Features": str(feature_list),
                }
                summary_data.update(
                    {
                        k: v
                        for k, v in results[model_name].items()
                        if k not in ["model", "y_probas"]
                    }
                )
                summary_rows.append(summary_data)

        # Generate the plot and summary CSV for current model
        if all_results_for_model:
            plot_roc_curves(
                all_results_for_model,
                y_test,
                os.path.join(output_dir, f"roc_specific_combinations_{safe_name}.png"),
                f"ROC-AUC our combinations vs. baseline \n Model: {model_name}",
            )
            pd.DataFrame(summary_rows).sort_values(by="auc", ascending=False).to_csv(
                os.path.join(
                    output_dir, f"summary_specific_combinations_{safe_name}.csv"
                ),
                index=False,
            )
            print(
                f"  -> Successfully generated ROC plot and summary CSV for {model_name}."
            )
        else:
            print(f"  -> No results were generated for {model_name}, skipping plot.")


def run_significance_testing(
    args, models, base_rank_phase="12_feature_rank_comprehensive"
):
    """
    Performs bootstrap significance testing to compare model combinations.
    """
    phase_name = "26_significance_testing"
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 80 + f"\n--- Starting Phase 26: Significance Testing ---\n" + "=" * 80)

    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")
    X_train, X_test, y_train, y_test = train_test_split(
        X_aggregate, y, test_size=0.2, random_state=SEED, stratify=y
    )
    y_test_np = y_test.to_numpy() # For bootstrap function

    all_significance_results = []

    for model_name, model in models.items():
        print(f"\n--- Processing Classifier: {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir, base_rank_phase, f"summary_single_features_{safe_name}.csv"
        )
        try:
            summary_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(f"ERROR: Prerequisite summary for {model_name} not found. Skipping.")
            continue

        def get_best_feature(keyword):
            df = summary_df[summary_df["Feature"].str.contains(keyword, regex=False)]
            return df.iloc[0]["Feature"] if not df.empty else None

        best_features_map = {
            "cas": get_best_feature("ecs_final"),
            "vffn": get_best_feature("v_ffn_norm"),
            "bos": get_best_feature("bos_attention"),
            "pos": get_best_feature("pos_score"),
            "ecs": get_best_feature("ecs_prompt_final"),
            "pks": get_best_feature("parameter_knowledge"),
        }

        feature_sets = {
            "Baseline": [best_features_map["ecs"], best_features_map["pks"]],
            "Ours 1": [best_features_map["cas"], best_features_map["vffn"]],
            "Ours 2": [
                best_features_map["bos"],
                best_features_map["pos"],
                best_features_map["cas"],
                best_features_map["vffn"],
            ],
        }

        # Train all models and store their results (including predictions)
        trained_models_results = {}
        for combo_name, feature_list in feature_sets.items():
            feature_list = [f for f in feature_list if f]
            if not feature_list:
                print(f"Skipping '{combo_name}' for {model_name}: missing features.")
                continue

            print(f"  -> Training {combo_name} combination...")
            results = train_and_evaluate(
                {model_name: clone(model)},
                X_train[feature_list], y_train,
                X_test[feature_list], y_test,
            )
            trained_models_results[combo_name] = results[model_name]

        if "Baseline" not in trained_models_results:
            print(f"Cannot perform significance tests for {model_name} without baseline results.")
            continue
        
        baseline_probas = trained_models_results["Baseline"]["y_probas"]
        baseline_preds = (baseline_probas > 0.5).astype(int)

        # Perform comparisons
        for combo_to_compare in ["Ours 1", "Ours 2"]:
            if combo_to_compare not in trained_models_results:
                continue

            print(f"  -> Comparing '{combo_to_compare}' vs. 'Baseline' for {model_name}")
            
            comparison_results = trained_models_results[combo_to_compare]
            comparison_probas = comparison_results["y_probas"]
            comparison_preds = (comparison_probas > 0.5).astype(int)

            # AUC
            ci_auc = get_bootstrap_ci(y_test_np, baseline_probas, comparison_probas, roc_auc_score, metric_type='proba')
            
            # F1 Score
            ci_f1 = get_bootstrap_ci(y_test_np, baseline_preds, comparison_preds, f1_score, metric_type='pred')
            
            # Recall
            ci_recall = get_bootstrap_ci(y_test_np, baseline_preds, comparison_preds, recall_score, metric_type='pred')
            
            # Pearson CC
            def pcc_func(y_true, y_pred): return pearsonr(y_true, y_pred)[0]
            ci_pcc = get_bootstrap_ci(y_test_np, baseline_probas, comparison_probas, pcc_func, metric_type='proba')
            
            all_significance_results.append({
                "Classifier": model_name,
                "Comparison": f"{combo_to_compare} vs. Baseline",
                "Metric": "AUC",
                "Baseline_Score": trained_models_results["Baseline"]["auc"],
                "Comparison_Score": comparison_results["auc"],
                "Difference": comparison_results["auc"] - trained_models_results["Baseline"]["auc"],
                "95%_CI_Lower": ci_auc[0],
                "95%_CI_Upper": ci_auc[1],
                "Is_Significant": "Yes" if ci_auc[0] > 0 or ci_auc[1] < 0 else "No",
            })
            all_significance_results.append({
                "Classifier": model_name,
                "Comparison": f"{combo_to_compare} vs. Baseline",
                "Metric": "F1",
                "Baseline_Score": trained_models_results["Baseline"]["f1"],
                "Comparison_Score": comparison_results["f1"],
                "Difference": comparison_results["f1"] - trained_models_results["Baseline"]["f1"],
                "95%_CI_Lower": ci_f1[0],
                "95%_CI_Upper": ci_f1[1],
                "Is_Significant": "Yes" if ci_f1[0] > 0 or ci_f1[1] < 0 else "No",
            })
            all_significance_results.append({
                "Classifier": model_name,
                "Comparison": f"{combo_to_compare} vs. Baseline",
                "Metric": "Recall",
                "Baseline_Score": trained_models_results["Baseline"]["recall"],
                "Comparison_Score": comparison_results["recall"],
                "Difference": comparison_results["recall"] - trained_models_results["Baseline"]["recall"],
                "95%_CI_Lower": ci_recall[0],
                "95%_CI_Upper": ci_recall[1],
                "Is_Significant": "Yes" if ci_recall[0] > 0 or ci_recall[1] < 0 else "No",
            })
            all_significance_results.append({
                "Classifier": model_name,
                "Comparison": f"{combo_to_compare} vs. Baseline",
                "Metric": "PCC",
                "Baseline_Score": trained_models_results["Baseline"]["pcc"],
                "Comparison_Score": comparison_results["pcc"],
                "Difference": comparison_results["pcc"] - trained_models_results["Baseline"]["pcc"],
                "95%_CI_Lower": ci_pcc[0],
                "95%_CI_Upper": ci_pcc[1],
                "Is_Significant": "Yes" if ci_pcc[0] > 0 or ci_pcc[1] < 0 else "No",
            })

    # Save final results to a CSV file
    if all_significance_results:
        results_df = pd.DataFrame(all_significance_results)
        save_path = os.path.join(output_dir, "significance_test_results_8020.csv")
        results_df.to_csv(save_path, index=False, float_format="%.4f")
        print(f"\nSignificance test results saved to: {save_path}")
        print("\n" + results_df.to_string())
    else:
        print("\nNo significance tests were run. Check for errors.")


def run_ultimate_showdown(args, models):
    """Finds the best 5-feature combo with anchor feature"""
    phase_name = f"19_ultimate_showdown_{args.selector}"
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Ultimate Showdown: {phase_name} ---\n"
        + "=" * 80
    )

    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")
    X_train, X_test, y_train, y_test = train_test_split(
        X_aggregate, y, test_size=0.2, random_state=SEED, stratify=y
    )

    final_roc_results, summary_data = {}, []
    for model_name, model in models.items():
        safe_name = model_name.replace(" ", "_")
        source_summary = os.path.join(
            args.base_dir,
            "12_feature_rank_comprehensive",
            f"summary_single_features_{safe_name}.csv",
        )
        try:
            ranked_df = pd.read_csv(source_summary)
        except FileNotFoundError:
            print(f"WARNING: Prerequisite ranking for {model_name} not found.")
            continue

        best_ecs = ranked_df[ranked_df.Feature.str.contains("ecs_prompt_final")].iloc[
            0
        ]["Feature"]
        best_cas = ranked_df[ranked_df.Feature.str.contains("ecs_final")].iloc[0][
            "Feature"
        ]
        companions = (
            ranked_df[~ranked_df.Feature.str.contains("ecs|cas")]
            .head(40)["Feature"]
            .tolist()
        )

        for anchor_name, anchor_feature in [("ECS", best_ecs), ("CAS", best_cas)]:
            pool = [anchor_feature] + companions
            best_features, final_metrics = find_optimal_feature_set(
                model,
                model_name,
                X_train,
                y_train,
                X_test,
                y_test,
                pool,
                method=args.selector,
                n_features=5,
            )
            final_roc_results[f"{model_name} ({anchor_name} Anchor)"] = final_metrics
            summary_data.append(
                {
                    "Model": model_name,
                    "Anchor": anchor_name,
                    **{
                        k: v
                        for k, v in final_metrics.items()
                        if k not in ["model", "y_probas"]
                    },
                    "Features": str(best_features),
                }
            )

    plot_roc_curves(
        final_roc_results,
        y_test,
        os.path.join(output_dir, "roc_curves_ultimate_showdown.png"),
        f"Ultimate Showdown ({args.selector.upper()})",
    )
    pd.DataFrame(summary_data).sort_values(
        by=["auc", "Model"], ascending=[False, True]
    ).to_csv(os.path.join(output_dir, "summary.csv"), index=False)


def run_phase_20_final_layer_analysis(args, models):
    """Runs experiments based only on features from the final layer."""
    phase_name = "20_final_layer_analysis"
    phase_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Running Phase 20: Final Layer Showdown ---\n"
        + "=" * 80
    )
    raw_df = load_data_optimized(args.results_file, cpu_count())
    X_final, y = create_final_layer_features(raw_df)
    run_standard_analysis_protocol(
        X_final,
        y,
        models,
        os.path.join(phase_dir, "final_layer_all_features"),
        "Final Layer (All)",
    )
    layer_only_cols = [
        c for c in X_final.columns if not any(k in c for k in ["cas", "ecs", "bos"])
    ]
    run_standard_analysis_protocol(
        X_final[layer_only_cols],
        y,
        models,
        os.path.join(phase_dir, "final_layer_layer_only"),
        "Final Layer (Layer-Based Only)",
    )


def run_phase_21_last_two_layers_analysis(args, models):
    """Runs experiments based only on features from the LAST TWO layers."""
    phase_name = "21_last_two_layers_analysis"
    phase_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Running Phase 21: Last Two Layers Showdown ---\n"
        + "=" * 80
    )
    raw_df = load_data_optimized(args.results_file, cpu_count())
    X_last_two, y = create_last_n_layers_features(raw_df, n=2)
    run_standard_analysis_protocol(
        X_last_two,
        y,
        models,
        os.path.join(phase_dir, "last_two_layers_all_features"),
        "Last Two Layers (All)",
    )
    layer_only_cols = [
        c for c in X_last_two.columns if not any(k in c for k in ["cas", "ecs", "bos"])
    ]
    run_standard_analysis_protocol(
        X_last_two[layer_only_cols],
        y,
        models,
        os.path.join(phase_dir, "last_two_layers_layer_only"),
        "Last Two Layers (Layer-Based Only)",
    )


def run_generic_ultimate_showdown(args, models):
    """
    A generic showdown that finds the best combination of features from specified groups of features.
    Runs default showdowns if no groups are specified via the command line.
    """
    feature_groups_to_run = args.feature_groups

    # If no groups are passed via command line, use the predefined defaults
    if not feature_groups_to_run:
        print(">>> No feature groups specified. Running default showdowns. <<<")
        # Loop through defaults and call the logic function for each
        for name, groups in DEFAULT_SHOWDOWNS.items():
            _run_single_showdown_logic(args, models, groups, name)
    else:
        # Run a single showdown with the specified feature groups
        phase_name = f"generic_showdown_{'_vs_'.join(sorted(feature_groups_to_run))}"
        _run_single_showdown_logic(args, models, feature_groups_to_run, phase_name)


def _run_single_showdown_logic(args, models, feature_groups, phase_name):
    """Run a single showdown instance"""
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Generic Showdown: {phase_name} ---\n"
        + "=" * 80
    )

    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")
    X_train, X_test, y_train, y_test = train_test_split(
        X_aggregate, y, test_size=0.2, random_state=SEED, stratify=y
    )

    for model_name, model in models.items():
        print(f"\n--- Finding best combinations for {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir,
            "12_feature_rank_comprehensive",
            f"summary_single_features_{safe_name}.csv",
        )
        try:
            ranked_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(f"WARNING: Prerequisite ranking for {model_name} not found.")
            continue

        best_features_per_group = {}
        for group_keyword in feature_groups:
            group_df = ranked_df[ranked_df.Feature.str.contains(group_keyword)]
            if not group_df.empty:
                best_feature = group_df.iloc[0]["Feature"]
                best_features_per_group[group_keyword] = best_feature
                print(f"  -> Best feature for group '{group_keyword}': {best_feature}")
            else:
                print(
                    f"  -> WARNING: No features found for group keyword '{group_keyword}'."
                )

        if len(best_features_per_group) < len(feature_groups):
            print(
                "  -> Could not find a best feature for all specified groups. Aborting showdown for this model."
            )
            continue

        candidate_pool = list(
            set(best_features_per_group.values())
        )  # Use set to handle duplicate keywords
        n_features = len(candidate_pool)

        best_combo_features, final_metrics = find_optimal_feature_set(
            model,
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            candidate_pool,
            method=args.selector,
            n_features=n_features,
        )

        summary_data = [
            {
                "Model": model_name,
                "Selector": args.selector,
                "Best_Combination": str(best_combo_features),
                **{
                    k: v
                    for k, v in final_metrics.items()
                    if k not in ["model", "y_probas"]
                },
            }
        ]
        pd.DataFrame(summary_data).to_csv(
            os.path.join(output_dir, f"summary_{safe_name}.csv"), index=False
        )
        print(f"  -> Saved best combination summary for {model_name}.")


def run_qualitative_case_study_analysis(
    args, models, base_rank_phase="12_feature_rank_comprehensive", top_percent=30
):
    """
    Performs a model-centric case study. For each model, it identifies the most confident
    predictions (top X percent), runs significance tests, and plots their distributions.
    """
    phase_name = f"23_paper_qualitative_analysis_{top_percent}percent"
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)
    print(
        "\n"
        + "=" * 80
        + f"\n--- Starting Phase 23: Model-Centric Qualitative Analysis (Top {top_percent}%) ---\n"
        + "=" * 80
    )

    # Get base data
    X_aggregate, y = get_engineered_data(args, feature_mode="comprehensive")
    if "raw_df" not in _CACHED_DATA:
        print("Error: Raw data not found in cache.")
        return
    raw_df = _CACHED_DATA["raw_df"]
    # Ensure indices are aligned before concatenation
    X_aggregate = X_aggregate.loc[raw_df.index].reset_index(drop=True)
    y = y.loc[raw_df.index].reset_index(drop=True)
    raw_df = raw_df.reset_index(drop=True)


    for model_name, model in models.items():
        print(f"\n--- Processing model: {model_name} ---")
        safe_name = model_name.replace(" ", "_")
        source_summary_path = os.path.join(
            args.base_dir, base_rank_phase, f"summary_single_features_{safe_name}.csv"
        )

        # Find best feature for each PACT component
        try:
            ranked_df = pd.read_csv(source_summary_path)
        except FileNotFoundError:
            print(f"ERROR: Prerequisite summary for {model_name} not found. Skipping.")
            continue

        feature_keywords = {
            "CAS": "ecs_final",
            "V_ffn_Norm": "v_ffn_norm",
            "BOS": "bos_attention",
            "POS": "pos_score",
        }
        best_features = {
            name: ranked_df[ranked_df["Feature"].str.contains(kw, regex=False)].iloc[0][
                "Feature"
            ]
            for name, kw in feature_keywords.items()
            if not ranked_df[ranked_df["Feature"].str.contains(kw, regex=False)].empty
        }
        if len(best_features) != len(feature_keywords):
            print("Could not find all required features. Aborting for this model.")
            continue
        print("  -> Best diagnostic features identified:", best_features)

        # Train definitive model and get probabilities ON THE ENTIRE DATASET
        # This is important for getting a probability for every single sample
        best_feature_list = list(best_features.values())
        
        # We need to train on all data to get probabilities for all data
        # Note: This is for analysis, not for reporting generalization performance.
        final_model = clone(model)
        final_model.fit(X_aggregate[best_feature_list], y)
        all_probas = final_model.predict_proba(X_aggregate[best_feature_list])[:, 1]

        # Build the diagnostic df
        analysis_df = raw_df[["source_id", "token_str", "label"]].copy()
        analysis_df["prediction_probability"] = all_probas
        analysis_df["label"] = pd.to_numeric(analysis_df["label"])
        for friendly_name, feature_col in best_features.items():
            analysis_df[friendly_name] = X_aggregate[feature_col].values

        # Separate correct and hallucinated predictions
        correct_df = analysis_df[analysis_df["label"] == 0]
        halluc_df = analysis_df[analysis_df["label"] == 1]
        
        # Calculate the number of samples to take for each class (top 30%)
        n_correct = int(len(correct_df) * (top_percent / 100.0))
        n_halluc = int(len(halluc_df) * (top_percent / 100.0))

        # Select the most CONFIDENT examples per model based on the percentage
        confident_correct = correct_df.sort_values(
            by="prediction_probability", ascending=True
        ).head(n_correct)
        
        confident_hallucinations = halluc_df.sort_values(
            by="prediction_probability", ascending=False
        ).head(n_halluc)

        confident_df = pd.concat([confident_hallucinations, confident_correct])

        # Save the best predictions and their scores for each feature
        halluc_path = os.path.join(
            output_dir, f"case_study_candidates_HALLUCINATIONS_{safe_name}_{top_percent}percent.csv"
        )
        correct_path = os.path.join(
            output_dir, f"case_study_candidates_CORRECT_{safe_name}_{top_percent}percent.csv"
        )
        confident_hallucinations.to_csv(halluc_path, index=False)
        confident_correct.to_csv(correct_path, index=False)
        print(
            f"\n  -> Saved top {top_percent}% confident candidate examples for {model_name}."
        )

        # Perform and save statistical significance tests on for confident subset
        print(
            f"\n  --- Significance Tests on Top {top_percent}% Confident Predictions for {model_name} ---"
        )
        stats_results = []
        for friendly_name in best_features.keys():
            stat, p_value = ttest_ind(
                confident_hallucinations[friendly_name],
                confident_correct[friendly_name],
                equal_var=False,
                nan_policy="omit",
            )
            stats_results.append(
                {
                    "PACT_Probe": friendly_name,
                    "T-statistic": stat,
                    "P-value": p_value,
                    "Significance": "p < 0.001"
                    if p_value < 0.001
                    else (
                        "p < 0.01"
                        if p_value < 0.01
                        else ("p < 0.05" if p_value < 0.05 else "Not Significant")
                    ),
                }
            )

        stats_df = pd.DataFrame(stats_results)
        stats_path = os.path.join(
            output_dir, f"significance_tests_confident_{safe_name}_{top_percent}percent.csv"
        )
        stats_df.to_csv(stats_path, index=False)
        print(f"  -> Confident prediction significance results saved to: {stats_path}")
        print(stats_df)

        # Generate and save a combined distribution plot per model
        print(f"\n  --- Generating Combined Distribution Plot for {model_name} ---")
        create_combined_distribution_plot(
            df=confident_df,
            feature_mapping=best_features,
            output_dir=output_dir,
            model_name=f"{model_name} (Top {top_percent}% Confident Predictions)",
        )


def run_phase_create_master_dataset(args, models):
    """
    Runs the full data engineering pipeline and saves the output to a
    master feature file for fast reuse in all other phases.
    """
    phase_name = "create_master_dataset"
    output_dir = os.path.join(args.base_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"--- Running Phase: {phase_name} ---")
    print("This will create a reusable master feature file.")
    print("=" * 80)

    # Load the raw data from JSONL
    raw_df = load_data_optimized(args.results_file, cpu_count())

    # Engineer per-layer features
    X_per_layer, y = create_per_layer_features(raw_df)

    # Engineer comprehensive aggregate features
    X_aggregate = create_aggregate_features(X_per_layer)

    # Combine everything into one master DataFrame
    y.name = "label"
    master_df = pd.concat(
        [
            raw_df[["source_id", "token_str"]].reset_index(drop=True),
            y.reset_index(drop=True),
            X_aggregate.reset_index(drop=True),
        ],
        axis=1,
    )

    # Save to the specified Parquet file path
    master_path = args.master_feature_path
    master_df.to_parquet(master_path, index=False)

    print("\n" + "*" * 80)
    print("--- MASTER FEATURE DATASET CREATION COMPLETE! ---")
    print(f"Saved {len(master_df)} records with {len(master_df.columns)} features to:")
    print(f"  -> {master_path}")
    print("You can now run all other analysis phases much faster.")
    print("*" * 80)


# ==============================================================================
# --- MAIN EXECUTION LOGIC ---
# ==============================================================================
def main():
    args = config()

    global DEFAULT_SHOWDOWNS
    DEFAULT_SHOWDOWNS = {
        "showdown_bos_pos_norm": ["bos", "pos", "norm"],
        # "showdown_cas_vs_pos": ["cas_alt_vs_final", "pos_score"],
        # "showdown_cas_vs_pos_vs_bos": ["cas_alt_vs_final", "pos_score", "bos_attention"],
    }

    print(
        ">>> Using model parameters optimized for small datasets to prevent overfitting. <<<"
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, random_state=SEED, class_weight="balanced", C=0.1
                    ),
                ),
            ]
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_estimators=75,
            learning_rate=0.05,
            num_leaves=8,
            max_depth=4,
            reg_alpha=0.5,
            reg_lambda=0.5,
            colsample_bytree=0.8,
            subsample=0.8,
        ),
    }
    if args.run_ebm:
        models["EBM"] = ExplainableBoostingClassifier(random_state=SEED, interactions=0)

    BEST_FEATURE_PLANE_PAIRS = {
        ("ecs_final", "v_ffn_norm"): (
            "Contextual Alignment (CAS)",
            "FFN Force (V_ffn)",
        ),
        ("ecs_prompt_final", "parameter_knowledge"): (
            "External Context (ECS)",
            "Parametric Knowledge (PKS)",
        ),
        ("ecs_final", "pos_score"): (
            "Contextual Alignment (CAS)",
            "Pathway Orthogonality (POS)",
        ),
        ("v_ffn_norm", "pos_score"): (
            "FFN Force (V_ffn)",
            "Pathway Orthogonality (POS)",
        ),
        ("v_attn_norm", "v_ffn_norm"): (
            "Attention Force (V_attn)",
            "FFN Force (V_ffn)",
        ),
        ("bos_attention", "ecs_final"): (
            "BOS Attention (BOS)",
            "Contextual Alignment (CAS)",
        ),
    }

    phase_runners = {
        "12_feature_rank_comprehensive": lambda a, m: run_standard_analysis_protocol(
            get_engineered_data(a, "comprehensive")[0],
            get_engineered_data(a, "comprehensive")[1],
            m,
            os.path.join(a.base_dir, "12_feature_rank_comprehensive"),
            "12_feature_rank_comprehensive",
        ),
        # "ecs_pks_showdown_comprehensive": lambda a, m: run_feature_showdown(
        #     a,
        #     m,
        #     "ecs_pks_showdown_comprehensive",
        #     {
        #         "ecs_prompt_final": ["ecs_prompt_final"],
        #         "parameter_knowledge": ["parameter_knowledge"],
        #     },
        #     "comprehensive",
        #     "12_feature_rank_comprehensive",
        # ),
        # "casf_vffn_pos_bos_showdown_comprehensive": lambda a, m: run_feature_showdown(
        #     a,
        #     m,
        #     "casf_vffn_pos_bos_showdown_comprehensive",
        #     {
        #         "CAS": ["ecs_final"],
        #         "V_ffn": ["v_ffn_norm"],
        #         "POS": ["pos_score"],
        #         "<bos>": ["bos_attention"],
        #     },
        #     "comprehensive",
        #     "12_feature_rank_comprehensive",
        # ),
        # "19_ultimate_showdown": run_ultimate_showdown,
        # "20_final_layer_analysis": run_phase_20_final_layer_analysis,
        # "21_last_two_layers_analysis": run_phase_21_last_two_layers_analysis,
        # "generic_showdown": run_generic_ultimate_showdown,
        "poster_plots_min": lambda a, m: run_specific_combinations_plot(a, m, "poster_plots_min"),
        "23_qualitative_analysis": lambda a, m: run_qualitative_case_study_analysis(
            a, m, top_percent=30
        ),
        "25_confident_prediction_planes": lambda a, m: plot_confident_prediction_planes(
            a, m, BEST_FEATURE_PLANE_PAIRS, top_percent=30
        ),
        # "26_significance_testing": run_significance_testing, 
    }

    if args.phase == "all":
        all_phases_to_run = phase_runners.copy()
        # all_phases_to_run["generic_showdown"] = (
        #     lambda a, m: run_generic_ultimate_showdown(a, m)
        # )
        # all_phases_to_run["24_best_feature_planes"] = (
        #     lambda a, m: plot_best_feature_plane(a, m, BEST_FEATURE_PLANE_PAIRS)
        # )
        for name, phase_func in all_phases_to_run.items():
            print(f"\n{'='*30} RUNNING PHASE: {name.upper()} {'='*30}")
            phase_func(args, models)
            print(f"\n{'='*30} COMPLETED PHASE: {name.upper()} {'='*30}")


    elif args.phase in phase_runners:
        phase_runners[args.phase](args, models)
    else:
        print(f"Error: Phase '{args.phase}' is not defined in the active runners.")
        sys.exit(1)


if __name__ == "__main__":
    main()