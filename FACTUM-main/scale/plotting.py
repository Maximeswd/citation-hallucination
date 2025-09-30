# ==============================================================================
# --- IMPORTS & SETUP ---
# ==============================================================================
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import random
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu, pearsonr
import os
from multiprocessing import Pool, cpu_count
from functools import partial


def config():
    parser = argparse.ArgumentParser(
        description="EXTREMELY FAST and reproducible script for analyzing and plotting results."
    )
    parser.add_argument(
        "--jsonl_path", required=True, help="Path to the detailed results JSONL file."
    )
    parser.add_argument(
        "--out_dir",
        default="unified_analysis_plots",
        help="Directory to save plots and reports.",
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="Number of layers in the model."
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of attention heads in the model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU cores to use. Defaults to all available.",
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="Feature selection method to use."
    )
    return parser.parse_args()


# ==============================================================================
# --- REPRODUCIBLE DATA LOADING (WITH ALL FIXES) ---
# ==============================================================================
args = config()
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)

def seed_worker(worker_seed):
    """Initializer function to seed each worker process for reproducibility"""
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


def worker_process_source_group(source_ids, byte_offset_map, jsonl_path):
    """Worker function for the data processing pass"""
    processed_tokens_for_chunk = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for source_id in source_ids:
            all_tokens_for_source = []
            for offset in byte_offset_map[source_id]:
                try:
                    f.seek(offset)
                    line = f.readline()
                    doc = json.loads(line)
                    for token in doc.get("token_data", []):
                        token["source_id"] = source_id
                    all_tokens_for_source.extend(doc.get("token_data", []))
                except (json.JSONDecodeError, IndexError):
                    continue

            true_samples = [
                tok for tok in all_tokens_for_source if tok.get("label") == 0
            ]
            hall_samples = [
                tok for tok in all_tokens_for_source if tok.get("label") == 1
            ]
            if not true_samples or not hall_samples:
                continue

            n_samples = min(len(true_samples), len(hall_samples))
            random.shuffle(hall_samples)
            random.shuffle(true_samples)
            balanced_tokens = true_samples[:n_samples] + hall_samples[:n_samples]

            for token in balanced_tokens:
                attn_data = token.get("top_p_attended_context_positions_per_head", [])
                if attn_data:
                    token["attn_dispersion_per_head"] = [
                        np.std(h.get("indices", []))
                        if len(h.get("indices", [])) > 1
                        else 0
                        for h in attn_data
                    ]
            processed_tokens_for_chunk.extend(balanced_tokens)
    return processed_tokens_for_chunk


def deterministic_chunking(data, num_chunks):
    """A deterministic chunking function"""
    k, m = divmod(len(data), num_chunks)
    return (
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)
    )


def load_and_prepare_data_optimized(jsonl_path, num_workers, num_layers, num_heads):
    """Loads data using a fully reproducible parallel strategy"""
    print("Pass 1: Creating byte-offset index in parallel...")
    file_size = os.path.getsize(jsonl_path)
    num_workers = min(num_workers, os.cpu_count())
    chunk_size = file_size // num_workers
    pool_args = [
        (jsonl_path, i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)
    ]
    pool_args[-1] = (jsonl_path, (num_workers - 1) * chunk_size, file_size)

    byte_offset_map = defaultdict(list)
    with Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(parallel_index_worker, pool_args)
        for local_index in tqdm(
            results_iterator, total=len(pool_args), desc="Indexing file chunks"
        ):
            for source_id, offsets in local_index.items():
                byte_offset_map[source_id].extend(offsets)

    unique_source_ids = sorted(list(byte_offset_map.keys()))
    print(
        f"Parallel indexing complete. Found {len(unique_source_ids)} unique source_ids."
    )
    source_id_chunks = list(deterministic_chunking(unique_source_ids, num_workers))
    print(
        f"\nPass 2: Processing {len(unique_source_ids)} source_id groups in parallel (REPRODUCIBLE)..."
    )

    worker_func = partial(
        worker_process_source_group,
        byte_offset_map=byte_offset_map,
        jsonl_path=jsonl_path,
    )

    final_balanced_tokens = []
    with Pool(processes=num_workers, initializer=seed_worker, initargs=(SEED,)) as pool:
        results_iterator = pool.imap(worker_func, source_id_chunks)
        for tokens_from_chunk in tqdm(
            results_iterator, total=len(source_id_chunks), desc="Processing data chunks"
        ):
            final_balanced_tokens.extend(tokens_from_chunk)

    print(
        f"\nParallel processing complete. Assembled dataset of {len(final_balanced_tokens)} tokens."
    )

    random.shuffle(final_balanced_tokens)
    true_samples = [tok for tok in final_balanced_tokens if tok.get("label") == 0]
    hall_samples = [tok for tok in final_balanced_tokens if tok.get("label") == 1]
    return true_samples, hall_samples


# ==============================================================================
# --- ALL PLOTTING AND REPORTING FUNCTIONS ---
# ==============================================================================

def _check_and_print_data_slice(samples, required_keys, plot_name):
    if not samples:
        print(f"\n[INFO] Skipping '{plot_name}': The sample list is empty.")
        return False
    if not isinstance(required_keys, list):
        required_keys = [required_keys]
    first_sample = samples[0]
    missing_keys = [
        key
        for key in required_keys
        if key not in first_sample or not first_sample.get(key)
    ]
    if missing_keys:
        print(
            f"\n[INFO] Skipping '{plot_name}': Required data key(s) not found or empty: {missing_keys}"
        )
        return False
    return True


def generate_descriptive_stats(
    true_samples, hall_samples, out_dir, num_layers, num_heads, source_id_counts
):
    """
    Generate descriptive statistics report.txt file
    """
    print("-> Generating Descriptive Statistics Report...")
    if not _check_and_print_data_slice(true_samples, [], "Descriptive Stats"):
        return

    report_lines = []

    # Sample distribution section
    report_lines.append("=" * 115)
    report_lines.append("SAMPLE DISTRIBUTION REPORT")
    report_lines.append("=" * 115)
    if source_id_counts:

        # Calculate average samples per group
        avg_samples = np.mean(list(source_id_counts.values()))
        report_lines.append(
            f"\nAverage balanced samples per source_id group: {avg_samples:.2f}"
        )
        report_lines.append(
            f"Total source_id groups with samples: {len(source_id_counts)}"
        )
        report_lines.append(
            f"Total tokens in final dataset: {len(true_samples) + len(hall_samples)}"
        )
        report_lines.append("\nDetailed counts per source_id (Top 30 shown):")

        # Sort counts by the number of samples 
        sorted_counts = sorted(
            source_id_counts.items(), key=lambda item: item[1], reverse=True
        )
        for sid, count in sorted_counts[:30]:
            report_lines.append(f"  - {sid}: {count} samples")
        if len(sorted_counts) > 30:
            report_lines.append("  ...")
    else:
        report_lines.append("\nNo source_id counts were provided.")

    report_lines.append("\n\n" + "=" * 115)
    report_lines.append("DESCRIPTIVE STATISTICS AND SIGNIFICANCE REPORT")
    report_lines.append("=" * 115)
    report_lines.append(
        f"\nBased on a balanced sample of {len(true_samples)} tokens per class."
    )

    sample_token = true_samples[0]
    per_layer_keys = [
        k
        for k, v in sample_token.items()
        if isinstance(v, list) and len(v) == num_layers
    ]
    per_head_keys = [
        k
        for k, v in sample_token.items()
        if isinstance(v, list)
        and len(v) == num_layers * num_heads
        and isinstance(v[0], (int, float))
    ]

    for s in true_samples + hall_samples:
        for key in per_head_keys:
            if key in s:
                data_reshaped = np.array(s[key]).reshape(num_layers, num_heads)
                s[f"{key}_mean_per_layer"] = np.mean(data_reshaped, axis=1).tolist()

    metrics_to_analyze = sorted(
        list(set(per_layer_keys + [f"{k}_mean_per_layer" for k in per_head_keys]))
    )
    report = [
        f"{'Metric':<50}{'Factual Mean':>15}{'Factual Std':>15}{'Hall. Mean':>15}{'Hall. Std':>15}{'p-value (M-W U)':>20}"
    ]
    report.append("-" * len(report[0]))

    for key in metrics_to_analyze:
        true_data = np.array(
            [s[key] for s in true_samples if key in s and s[key]]
        ).flatten()
        hall_data = np.array(
            [s[key] for s in hall_samples if key in s and s[key]]
        ).flatten()
        if true_data.size == 0 or hall_data.size == 0:
            continue
        t_mean, t_std, h_mean, h_std = (
            np.mean(true_data),
            np.std(true_data),
            np.mean(hall_data),
            np.std(hall_data),
        )
        try:
            _, p_value = mannwhitneyu(hall_data, true_data, alternative="two-sided")
            p_value_str = f"{p_value:.3e}{'*' * int(p_value < 0.05) + '*' * int(p_value < 0.01) + '*' * int(p_value < 0.001)}"
        except ValueError:
            p_value_str = "N/A"
        report.append(
            f"{key:<50}{t_mean:>15.4f}{t_std:>15.4f}{h_mean:>15.4f}{h_std:>15.4f}{p_value_str:>20}"
        )

    full_report = [
        "=" * 115,
        "DESCRIPTIVE STATISTICS AND SIGNIFICANCE REPORT",
        "=" * 115,
        f"\nBased on a balanced sample of {len(true_samples)} tokens per class.\n",
    ] + report
    with open(os.path.join(out_dir, "report_descriptive_statistics.txt"), "w") as f:
        f.write("\n".join(full_report))
    print("  - Saved report_descriptive_statistics.txt")


# Helper function for significance level stars
def get_significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def analyze_and_report_correlations(true_samples, hall_samples, out_dir):
    print("-> Generating Aggregate Correlation Report...")
    all_samples = true_samples + hall_samples
    if not all_samples:
        return

    # Total number of samples used for correlation
    N = len(all_samples)

    df_data = [
        {
            "label": s["label"],
            **{
                f"{k}_mean": np.mean(v)
                for k, v in s.items()
                if isinstance(v, list) and v and isinstance(v[0], (int, float))
            },
        }
        for s in all_samples
    ]
    df = pd.DataFrame(df_data).fillna(0)
    if len(df.columns) <= 1:
        return  

    correlations_with_p_values = []
    for key in df.columns:
        if key != "label" and df[key].nunique() > 1 and np.std(df[key]) > 0:
            # pearsonr returns (correlation_coefficient, p_value)
            r_val, p_val = pearsonr(df["label"], df[key])
            correlations_with_p_values.append((key, r_val, p_val))

    # Sort by absolute correlation coefficient descending order
    correlations_with_p_values.sort(key=lambda x: abs(x[1]), reverse=True)

    report = []
    report.append("=" * 105)  
    report.append("Point-Biserial Correlation with Hallucination Label")
    report.append(f"Sample Size (N): {N}")  
    report.append("=" * 105)
    report.append(
        f"{'Metric':<55} {'Correlation (r)':>15} {'p-value':>15} {'Significance':>12}"
    )  
    report.append("-" * 105)  

    for name, r_val, p_val in correlations_with_p_values:
        p_val_str = f"{p_val:.4e}"  # Format p-value to scientific notation
        significance_stars = get_significance_stars(p_val)
        report.append(
            f"{name:<55} {r_val:>15.4f} {p_val_str:>15} {significance_stars:<12}"
        )

    report.append("=" * 105)  
    with open(os.path.join(out_dir, "report_correlations.txt"), "w") as f:
        f.write("\n".join(report))
    print("  - Saved report_correlations.txt")


def plot_generic_heatmap(
    true_samples,
    hall_samples,
    out_dir,
    num_layers,
    num_heads,
    metric_key,
    plot_title_prefix,
    filename_prefix,
):
    """
    Generates and saves a set of three heatmaps:
    1. Factual Tokens
    2. Hallucinated Tokens
    3. Difference (Factual - Hallucinated)
    """
    if not _check_and_print_data_slice(true_samples, [metric_key], plot_title_prefix):
        return
    print(f"-> Generating {plot_title_prefix} heatmaps with updated styling...")

    # Calculate means
    true_mean = (
        np.mean([s[metric_key] for s in true_samples], axis=0)
        .reshape(num_layers, num_heads)
        .T
    )
    hall_mean = (
        np.mean([s[metric_key] for s in hall_samples], axis=0)
        .reshape(num_layers, num_heads)
        .T
    )
    diff = true_mean - hall_mean

    fig, axes = plt.subplots(
        1, 3, figsize=(40, 12), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )

    fig.suptitle(f"{plot_title_prefix} per Head/Layer", fontsize=30)

    vmin, vmax = (
        min(np.nanmin(true_mean), np.nanmin(hall_mean)),
        max(np.nanmax(true_mean), np.nanmax(hall_mean)),
    )

    sns.heatmap(true_mean, cmap="viridis", vmin=vmin, vmax=vmax, ax=axes[0], cbar=False)
    axes[0].set_title("Factual Tokens", fontsize=24)
    axes[0].set_xlabel("Layer", fontsize=20)
    axes[0].set_ylabel("Head", fontsize=20)

    sns.heatmap(hall_mean, cmap="viridis", vmin=vmin, vmax=vmax, ax=axes[1], cbar=False)
    axes[1].set_title("Hallucinated Tokens", fontsize=24)
    axes[1].set_xlabel("Layer", fontsize=20)
    axes[1].set_ylabel("")  

    diff_max = np.nanmax(np.abs(diff))
    cbar_kws = {"label": "Difference (Factual - Hallucinated)"}
    sns.heatmap(
        diff,
        cmap="coolwarm",
        center=0,
        vmin=-diff_max,
        vmax=diff_max,
        ax=axes[2],
        cbar_kws=cbar_kws,
    )
    axes[2].set_title("Difference Heatmap (F - H)", fontsize=24)
    axes[2].set_xlabel("Layer", fontsize=20)
    axes[2].set_ylabel("")  

    cbar = axes[2].collections[0].colorbar
    cbar.ax.get_yaxis().label.set_size(20)
    cbar.ax.tick_params(labelsize=16)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, f"{filename_prefix}_heatmaps.png"), dpi=300)
    plt.close()


def plot_all_per_layer_metrics(
    true_samples, hall_samples, out_dir, num_layers, num_heads
):
    """
    Generates per-layer metric trend and difference plots 
    """
    print(
        "-> Generating all Per-Layer Metric Trend and Difference Plots with updated styling..."
    )
    metrics = {
        "POS Score": "pos_score_per_layer",
        "V_attn Norm": "v_attn_norm_per_layer",
        "V_ffn Norm": "v_ffn_norm_per_layer",
    }
    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    # Define font sizes
    suptitle_fontsize = 30
    title_fontsize = 24
    label_fontsize = 20
    tick_fontsize = 16
    legend_fontsize = 16

    # Trend plots
    rows = (n_metrics + 1) // 2
    fig_trends, axes_trends = plt.subplots(rows, 2, figsize=(24, 8 * rows), sharex=True)
    fig_trends.suptitle(
        "Mean Trend of Metrics per Layer", fontsize=suptitle_fontsize, y=0.98
    )
    axes_trends = axes_trends.flatten()

    # Difference plots 
    fig_diffs, axes_diffs = plt.subplots(rows, 2, figsize=(24, 8 * rows), sharex=True)
    fig_diffs.suptitle(
        "Difference in Metrics per Layer (Hallucinated - Factual)",
        fontsize=suptitle_fontsize,
        y=0.98,
    )
    axes_diffs = axes_diffs.flatten()

    for i, (title, key) in enumerate(metrics.items()):
        if not _check_and_print_data_slice(true_samples, [key], title):
            axes_trends[i].axis("off")
            axes_diffs[i].axis("off")
            continue

        true_mean, true_std = (
            np.mean([s[key] for s in true_samples], axis=0),
            np.std([s[key] for s in true_samples], axis=0),
        )
        hall_mean, hall_std = (
            np.mean([s[key] for s in hall_samples], axis=0),
            np.std([s[key] for s in hall_samples], axis=0),
        )
        layers = np.arange(len(true_mean))

        ax_trend = axes_trends[i]
        ax_trend.plot(layers, true_mean, color="royalblue", label="Factual")
        ax_trend.fill_between(
            layers,
            true_mean - true_std,
            true_mean + true_std,
            color="royalblue",
            alpha=0.2,
        )
        ax_trend.plot(
            layers, hall_mean, color="orangered", ls="--", label="Hallucinated"
        )
        ax_trend.fill_between(
            layers,
            hall_mean - hall_std,
            hall_mean + hall_std,
            color="orangered",
            alpha=0.2,
        )
        ax_trend.set_title(title, fontsize=title_fontsize)
        ax_trend.set_ylabel("Score Value", fontsize=label_fontsize)
        ax_trend.grid(True, ls="--", alpha=0.7)
        ax_trend.legend(fontsize=legend_fontsize)
        ax_trend.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        if i // 2 == rows - 1:  
            ax_trend.set_xlabel("Layer", fontsize=label_fontsize)

        ax_diff = axes_diffs[i]
        diff = hall_mean - true_mean

        ax_diff.bar(
            layers,
            diff,
            color=["#FF7F0E" if x > 0 else "#2589d1" for x in diff],
            alpha=0.8,
            edgecolor="black",
        )
        ax_diff.axhline(0, color="k", lw=0.8)
        ax_diff.set_title(title, fontsize=title_fontsize)
        ax_diff.set_ylabel("Difference (H - F)", fontsize=label_fontsize)
        ax_diff.grid(True, linestyle="--")
        ax_diff.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        if i // 2 == rows - 1: 
            ax_diff.set_xlabel("Layer", fontsize=label_fontsize)

    for fig, filename in [
        (fig_trends, "per_layer_metric_trends.png"),
        (fig_diffs, "per_layer_metric_differences.png"),
    ]:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close(fig)


def plot_single_metric_diff_bar(
    true_samples, hall_samples, out_dir, metric_key, title, filename
):
    """
    Generates a single bar plot for metric differences
    """
    if not _check_and_print_data_slice(true_samples, [metric_key], title):
        return
    print(f"-> Generating {title} plot with updated styling...")

    diff = np.mean([s[metric_key] for s in hall_samples], axis=0) - np.mean(
        [s[metric_key] for s in true_samples], axis=0
    )
    avg_diff = diff.mean()

    plt.figure(figsize=(16, 9))
    layers = np.arange(len(diff))

    plt.bar(
        layers,
        diff,
        color=["#FF7F0E" if x > 0 else "#2589d1" for x in diff],
        edgecolor="k",
        alpha=0.8,
    )
    plt.axhline(
        avg_diff, color="k", ls="--", lw=2, label=f"Average Difference ({avg_diff:.4f})"
    )

    plt.xlabel("Layer", fontsize=20)
    plt.ylabel("Difference (H - F)", fontsize=20)
    plt.title(f"{title} (Hallucination − Factual)", fontsize=24)
    plt.xticks(layers[:: (2 if len(layers) > 20 else 1)], fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", ls="--", alpha=0.7)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()


def plot_correlation_barplots(true_samples, hall_samples, out_dir):
    """
    Generates correlation bar plots
    """
    print("-> Generating Correlation Bar Plots with updated styling...")
    all_samples = true_samples + hall_samples
    if not all_samples:
        return

    label = np.array([s["label"] for s in all_samples])
    metrics = {
        "POS_Score": "pos_cos_similarity_per_layer",
        "PKS_Score": "parameter_knowledge_difference",
        "V_ffn": "v_ffn_norm_per_layer",
    }

    for name, key in metrics.items():
        if not _check_and_print_data_slice(all_samples, [key], name):
            continue

        num_metrics = len(all_samples[0][key])
        corrs = [
            pearsonr(label, [s[key][l] for s in all_samples])[0]
            for l in range(num_metrics)
        ]

        plt.figure(figsize=(20, 10))
        plt.bar(
            range(num_metrics),
            corrs,
            color=["#FF7F0E" if c > 0 else "#2589d1" for c in corrs],
            edgecolor="k",
        )

        plt.title(f"Per-Layer Correlation: {name} vs. Hallucination", fontsize=26)
        plt.xlabel("Layer", fontsize=22)
        plt.ylabel("Correlation (r)", fontsize=22)
        plt.tick_params(axis="both", which="major", labelsize=18)
        plt.grid(True, axis="y", ls="--")
        plt.axhline(0, color="k", lw=0.8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"corr_barplot_{name.lower()}_vs_hallucination.png"),
            dpi=300,
        )
        plt.close()


def plot_correlation_heatmaps(
    true_samples, hall_samples, out_dir, num_layers, num_heads
):
    """
    Generates and saves a correlation heatmap between feature and hallucination label
    """
    print("-> Generating Correlation Heatmaps with updated styling...")
    all_samples = true_samples + hall_samples
    if not all_samples:
        return

    label = np.array([s["label"] for s in all_samples])
    metrics = {
        "cas_alt_vs_final": "cas_alt_vs_final_layer_per_head",
        "ecs_prompt_final": "ecs_prompt_final_layer_per_head",
        "attn_dispersion": "attn_dispersion_per_head",
        "bos_attention": "bos_attention",
    }

    for name, key in metrics.items():
        if not _check_and_print_data_slice(all_samples, [key], name):
            continue

        corr_matrix = np.zeros((num_heads, num_layers))
        for l in range(num_layers):
            for h in range(num_heads):
                idx = l * num_heads + h
                series = [s[key][idx] for s in all_samples]
                # Ensure there's variance before calculating correlation
                if np.std(series) > 0:
                    corr_matrix[h, l] = pearsonr(label, series)[0]

        plt.figure(figsize=(24, 12))
        cbar_kws = {"label": "Correlation with Hallucination (r)"}
        ax = sns.heatmap(corr_matrix, cmap="coolwarm", center=0, cbar_kws=cbar_kws)
        ax.set_title(f"Per-Head Correlation: {name} vs. Hallucination", fontsize=26)
        ax.set_xlabel("Layer", fontsize=20)
        ax.set_ylabel("Head", fontsize=20)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
        cbar = ax.collections[0].colorbar
        cbar.ax.get_yaxis().label.set_size(20)
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()

        # Save the figure
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, f"corr_heatmap_{name}.png"), dpi=300)
        plt.close()


def plot_attention_distributions(
    true_samples, hall_samples, out_dir, key, title_suffix, filename_suffix
):
    """
    Generates attention distribution plots.
    """
    if not _check_and_print_data_slice(
        true_samples, [key], f"Attention Distributions for {title_suffix}"
    ):
        return
    print(
        f"-> Generating Detailed Attention Distribution plots for {title_suffix} with updated styling..."
    )

    # Collect all unique token types that exist in the data
    all_present_types = set()
    all_samples = true_samples + hall_samples
    for s in all_samples:
        for h_data in s.get(key, []):
            if isinstance(h_data, dict):
                all_present_types.update(h_data.get("types", []))

    # If no types are found, return aka exit 
    if not all_present_types:
        print(f"  [WARNING] No token 'types' found in data key '{key}'. Skipping plot.")
        return

    # Use types for plotting
    types_to_plot = sorted(list(all_present_types))

    indices = {"factual": defaultdict(list), "hallucinated": defaultdict(list)}
    for label, samples in [("factual", true_samples), ("hallucinated", hall_samples)]:
        for s in samples:
            for h_data in s.get(key, []):
                if isinstance(h_data, dict):
                    for i, type_str in enumerate(h_data.get("types", [])):
                        if i < len(h_data.get("indices", [])):
                            indices[label][type_str].append(h_data["indices"][i])

    # Adjust figure height based on the number of types found
    fig, axes = plt.subplots(
        len(types_to_plot), 1, figsize=(20, 7 * len(types_to_plot)), squeeze=False
    )
    axes = axes.flatten() 
    fig.suptitle(
        f"Distribution of Top-P Attended Positions ({title_suffix})", fontsize=30
    )

    for i, token_type in enumerate(types_to_plot):
        ax = axes[i]
        fact_indices = indices["factual"].get(token_type, [])
        hall_indices = indices["hallucinated"].get(token_type, [])

        all_type_indices = fact_indices + hall_indices

        if all_type_indices:
            min_pos, max_pos = min(all_type_indices), max(all_type_indices)
            current_range = max_pos - min_pos
            buffer = current_range * 0.1 if current_range > 0 else 5
            ax.set_xlim(max(0, min_pos - buffer), max_pos + buffer)

        if fact_indices:
            sns.kdeplot(
                fact_indices,
                ax=ax,
                color="royalblue",
                fill=True,
                label="Factual",
                alpha=0.6,
                bw_adjust=0.5,
                cut=0,
            )
        if hall_indices:
            sns.kdeplot(
                hall_indices,
                ax=ax,
                color="orangered",
                fill=True,
                label="Hallucinated",
                alpha=0.6,
                bw_adjust=0.5,
                cut=0,
            )

        ax.set_title(f"Attention to '{token_type.title()}' Tokens", fontsize=24)
        ax.set_xlabel("Token Position", fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True, ls="--")
        if fact_indices or hall_indices:  
            ax.legend(fontsize=18)

    # Hide any unused subplots if the figure grid is larger than needed
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        os.path.join(out_dir, f"attn_dist_by_type_{filename_suffix}.png"), dpi=300
    )
    plt.close()


def plot_top_p_token_type_frequency(true_samples, hall_samples, out_dir):
    """
    Generates a token type frequency bar plot
    """
    key = "top_p_attended_positions_per_head"
    if not _check_and_print_data_slice(true_samples, [key], "Token Type Frequency"):
        return
    print("-> Generating Top-P Token Type Frequency plot with updated styling...")

    def count_types(samples):
        counts = Counter()
        for s in samples:
            for h_data in s.get(key, []):
                if isinstance(h_data, dict):
                    counts.update(h_data.get("types", []))
        return counts

    fact_counts, hall_counts = count_types(true_samples), count_types(hall_samples)
    df = pd.DataFrame(
        [
            {
                "Token Type": t,
                "Frequency": fact_counts.get(t, 0) / len(true_samples),
                "Label": "Factual",
            }
            for t in set(fact_counts) | set(hall_counts)
        ]
        + [
            {
                "Token Type": t,
                "Frequency": hall_counts.get(t, 0) / len(hall_samples),
                "Label": "Hallucinated",
            }
            for t in set(fact_counts) | set(hall_counts)
        ]
    )

    plt.figure(figsize=(16, 10))
    ax = sns.barplot(
        data=df,
        x="Token Type",
        y="Frequency",
        hue="Label",
        palette={"Factual": "royalblue", "Hallucinated": "orangered"},
    )

    ax.set_title("Average Frequency of Token Types in Top Attention Set", fontsize=24)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel("Avg. Count per Response Token", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize="18")  
    plt.setp(ax.get_legend().get_title(), fontsize="20")  
    plt.grid(axis="y", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attn_type_frequency_barplot.png"), dpi=300)
    plt.close()


def generate_attention_sink_report(
    true_samples, hall_samples, out_dir, num_layers, num_heads
):
    """
    Analyzes the 'attention sink' by calculating the average number of heads
    and layers that attend to position 0 for both factual and hallucinated tokens. 
    Basically to show that the original ECS score almost always includes the bos in the top-10% most attended to tokens.
    """
    print("-> Generating Detailed Attention Sink Report...")
    key = "top_p_attended_positions_per_head"

    if not _check_and_print_data_slice(
        true_samples + hall_samples, [key], "Attention Sink Report"
    ):
        return

    def _calculate_detailed_sink_stats(samples, n_layers, n_heads):
        """Helper to calculate detailed sink stats for a list of token samples."""
        total_heads_attending_to_sink = 0
        total_layers_with_sink_attention = 0
        total_valid_tokens = 0

        for token in samples:
            attn_data = token.get(key, [])
            if not attn_data or len(attn_data) != n_layers * n_heads:
                continue

            total_valid_tokens += 1

            # Count total heads attending to sink for this token 
            heads_this_token = sum(
                1 for head_data in attn_data if 0 in head_data.get("indices", [])
            )
            total_heads_attending_to_sink += heads_this_token

            # Count layers with at least one head attending to sink for this token 
            layers_this_token = 0
            for l in range(n_layers):
                layer_has_sink = False
                # Check all heads within current layer
                for h in range(n_heads):
                    idx = l * n_heads + h
                    if 0 in attn_data[idx].get("indices", []):
                        layer_has_sink = True
                        break  
                if layer_has_sink:
                    layers_this_token += 1
            total_layers_with_sink_attention += layers_this_token

        if total_valid_tokens == 0:
            return {"avg_heads": 0, "avg_layers": 0, "total_tokens": 0}

        avg_heads = total_heads_attending_to_sink / total_valid_tokens
        avg_layers = total_layers_with_sink_attention / total_valid_tokens

        return {
            "avg_heads": avg_heads,
            "avg_layers": avg_layers,
            "total_tokens": total_valid_tokens,
        }

    fact_stats = _calculate_detailed_sink_stats(true_samples, num_layers, num_heads)
    hall_stats = _calculate_detailed_sink_stats(hall_samples, num_layers, num_heads)

    report_lines = [
        "=" * 80,
        "QUANTITATIVE ATTENTION SINK ANALYSIS (Attention to Position 0)",
        "=" * 80,
        "This report measures the average number of heads and layers per token that include",
        "position 0 in their set of top-attended context tokens.",
        f"Model Configuration: {num_layers} Layers, {num_heads} Heads.",
        "-" * 80,
        f"\n--- Factual Tokens (Analyzed {fact_stats['total_tokens']} tokens) ---",
        f"  - Average HEADS per token attending to sink: {fact_stats['avg_heads']:.2f} (out of {num_heads * num_layers} total heads)",
        f"  - Average LAYERS per token attending to sink: {fact_stats['avg_layers']:.2f} (out of {num_layers} total layers)",
        f"\n--- Hallucinated Tokens (Analyzed {hall_stats['total_tokens']} tokens) ---",
        f"  - Average HEADS per token attending to sink: {hall_stats['avg_heads']:.2f} (out of {num_heads * num_layers} total heads)",
        f"  - Average LAYERS per token attending to sink: {hall_stats['avg_layers']:.2f} (out of {num_layers} total layers)",
        "\n" + "=" * 80,
    ]

    with open(
        os.path.join(out_dir, "report_attention_sink_quantitative.txt"), "w"
    ) as f:
        f.write("\n".join(report_lines))
    print("  - Saved report_attention_sink_quantitative.txt")


def report_inter_metric_correlations(true_samples, hall_samples, out_dir, metric_pairs):
    """
    Calculates and reports the Pearson correlation between specified pairs of features/scores.
    This is for analyzing relationships between different scores, not their correlation with the hallucination label.
    """
    print("-> Generating Inter-Metric Correlation Report...")
    all_samples = true_samples + hall_samples
    if not all_samples:
        return

    report_lines = [
        "=" * 110,
        "CORRELATION REPORT BETWEEN METRICS",
        "=" * 110,
        "This report shows the Pearson correlation between the per-head values of different metrics,",
        "aggregated across all tokens in the balanced dataset.",
        "-" * 110,
        f"{'Metric Pair':<75} {'Correlation (r)':>15} {'p-value':>15}",
        "-" * 110,
    ]

    for key1, key2 in metric_pairs:
        # Check if the keys exist in the first sample to avoid unnecessary processing
        if not _check_and_print_data_slice(
            all_samples, [key1, key2], f"{key1} vs {key2}"
        ):
            continue

        # Flatten the per-head data from all tokens into two long lists
        values1, values2 = [], []
        for s in all_samples:
            # Ensure both keys are present for current token
            if key1 in s and key2 in s and s[key1] and s[key2]:
                # Ensure the lists have the same length (they should if they are per-head metrics)
                if len(s[key1]) == len(s[key2]):
                    values1.extend(s[key1])
                    values2.extend(s[key2])

        if not values1 or not values2:
            print(
                f"  [WARNING] No valid data found for metric pair: {key1} vs {key2}. Skipping."
            )
            continue

        # Calculate Pearson cor
        try:
            r_val, p_val = pearsonr(values1, values2)
            p_val_str = f"{p_val:.4e}"
            significance_stars = get_significance_stars(p_val)
            pair_name = (
                f"{key1.replace('_per_head', '')}  VS  {key2.replace('_per_head', '')}"
            )
            report_lines.append(
                f"{pair_name:<75} {r_val:>15.4f} {p_val_str:>15}  {significance_stars}"
            )
        except ValueError:
            report_lines.append(
                f"{pair_name:<75} {'N/A (check variance)':>15} {'N/A':>15}"
            )

    report_lines.append("=" * 110)

    with open(os.path.join(out_dir, "report_inter_metric_correlations.txt"), "w") as f:
        f.write("\n".join(report_lines))
    print("  - Saved report_inter_metric_correlations.txt")


def plot_alignment_conflict_plane(true_samples, hall_samples, out_dir):
    """
    Generates a 2D density plot visualizing the relationship between the
    Contextual Alignment Score (CAS) and the Pathway Orthogonality Score (POS),
    revealing the 'Alignment-Conflict Plane'.
    """
    print("-> Generating 2D Alignment-Conflict Plane plot...")

    # We use the mean of the per-head CAS scores for each token
    true_cas_scores = [
        np.mean(s["cas_alt_vs_final_layer_per_head"])
        for s in true_samples
        if "cas_alt_vs_final_layer_per_head" in s and s["cas_alt_vs_final_layer_per_head"]
    ]
    hall_cas_scores = [
        np.mean(s["cas_alt_vs_final_layer_per_head"])
        for s in hall_samples
        if "cas_alt_vs_final_layer_per_head" in s and s["cas_alt_vs_final_layer_per_head"]
    ]

    # Pathway Alignment Score (PAS)
    # We use the mean of the per-layer POS scores for each token.
    true_pos_scores = [
        np.mean(s["pos_cos_similarity_per_layer"])
        for s in true_samples
        if "pos_cos_similarity_per_layer" in s and s["pos_cos_similarity_per_layer"]
    ]
    hall_pos_scores = [
        np.mean(s["pos_cos_similarity_per_layer"])
        for s in hall_samples
        if "pos_cos_similarity_per_layer" in s and s["pos_cos_similarity_per_layer"]
    ]

    plt.figure(figsize=(11, 9))

    # Plot the 'Correct' and 'Hallucinated' distributions
    sns.kdeplot(
        x=true_cas_scores,
        y=true_pos_scores,
        cmap="Blues",
        fill=True,
        thresh=0.1,
        label="Correct",
    )
    sns.kdeplot(
        x=hall_cas_scores,
        y=hall_pos_scores,
        cmap="Oranges",
        fill=True,
        thresh=0.1,
        label="Hallucinated",
    )

    plt.title("The Alignment-Conflict Plane", fontsize=24, pad=20)
    plt.xlabel("Contextual Alignment Score (CAS)", fontsize=20)
    plt.ylabel("Pathway Orthogonality Score (POS)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Prediction", fontsize=14, title_fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.axhline(
        y=np.median(true_pos_scores + hall_pos_scores),
        color="grey",
        linestyle="--",
        alpha=0.5,
    )
    plt.axvline(
        x=np.median(true_cas_scores + hall_cas_scores),
        color="grey",
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_conflict_plane.png"), dpi=300)
    plt.close()


def plot_conviction_conflict_plane(true_samples, hall_samples, out_dir):
    """
    Generates a 2D density plot visualizing the relationship between the
    FFN Update Force (V_ffn) and the Pathway Orthogonality Score (POS),
    revealing the 'Conviction-Conflict Plane'.
    """
    print("-> Generating 2D Conviction-Conflict Plane plot...")

    # FFN Update Force (V_ffn Norm)
    true_vffn_scores = [
        np.mean(s["v_ffn_norm_per_layer"])
        for s in true_samples
        if "v_ffn_norm_per_layer" in s and s["v_ffn_norm_per_layer"]
    ]
    hall_vffn_scores = [
        np.mean(s["v_ffn_norm_per_layer"])
        for s in hall_samples
        if "v_ffn_norm_per_layer" in s and s["v_ffn_norm_per_layer"]
    ]

    # Pathway Orthogonality Score (POS)
    true_pos_scores = [
        np.mean(s["pos_score_per_layer"])
        for s in true_samples
        if "pos_score_per_layer" in s and s["pos_score_per_layer"]
    ]
    hall_pos_scores = [
        np.mean(s["pos_score_per_layer"])
        for s in hall_samples
        if "pos_score_per_layer" in s and s["pos_score_per_layer"]
    ]


    plt.figure(figsize=(11, 9))

    # Plot the 'Correct' and 'Hallucinated' distributions
    sns.kdeplot(
        x=true_vffn_scores,
        y=true_pos_scores,
        cmap="Blues",
        fill=True,
        thresh=0.1,
        label="Correct",
    )
    sns.kdeplot(
        x=hall_vffn_scores,
        y=hall_pos_scores,
        cmap="Oranges",
        fill=True,
        thresh=0.1,
        label="Hallucinated",
    )

    plt.title("The Conviction-Conflict Plane", fontsize=24, pad=20)
    plt.xlabel("Parametric Conviction (Mean ||v_ffn||)", fontsize=20)
    plt.ylabel("Pathway Conflict (POS)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Prediction", fontsize=14, title_fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "conviction_conflict_plane.png"), dpi=300)
    plt.close()


def plot_peak_pathway_force_distribution(true_samples, hall_samples, out_dir):
    """
    Generates a translucent 2D density plot visualizing the relationship between the
    PEAK (max) force of the Attention and FFN pathways.
    """
    print("-> Generating 2D Peak Pathway Force Distribution plot...")

    true_attn_force = [
        np.max(s["v_attn_norm_per_layer"])
        for s in true_samples
        if "v_attn_norm_per_layer" in s and s["v_attn_norm_per_layer"]
    ]
    true_ffn_force = [
        np.max(s["v_ffn_norm_per_layer"])
        for s in true_samples
        if "v_ffn_norm_per_layer" in s and s["v_ffn_norm_per_layer"]
    ]
    hall_attn_force = [
        np.max(s["v_attn_norm_per_layer"])
        for s in hall_samples
        if "v_attn_norm_per_layer" in s and s["v_attn_norm_per_layer"]
    ]
    hall_ffn_force = [
        np.max(s["v_ffn_norm_per_layer"])
        for s in hall_samples
        if "v_ffn_norm_per_layer" in s and s["v_ffn_norm_per_layer"]
    ]

    plt.figure(figsize=(11, 9))

    sns.kdeplot(
        x=true_attn_force,
        y=true_ffn_force,
        cmap="Blues",
        fill=True,
        thresh=0.1,
        label="Correct",
        alpha=0.6,
    )
    sns.kdeplot(
        x=hall_attn_force,
        y=hall_ffn_force,
        cmap="Oranges",
        fill=True,
        thresh=0.1,
        label="Hallucinated",
        alpha=0.6,
    )

    plt.title("Distribution of Peak Pathway Forces", fontsize=24, pad=20)
    plt.xlabel("Peak Attention Update Force (max ||v_attn||)", fontsize=20)
    plt.ylabel("Peak FFN Update Force (max ||v_ffn||)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Prediction", fontsize=14, title_fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "peak_pathway_force_distribution.png"), dpi=300)
    plt.close()


def main():
    args = config()
    plt.style.use("seaborn-v0_8-whitegrid")
    os.makedirs(args.out_dir, exist_ok=True)

    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} parallel workers.")

    true_samples, hall_samples = load_and_prepare_data_optimized(
        args.jsonl_path, num_workers, args.num_layers, args.num_heads
    )

    if not true_samples or not hall_samples:
        print("[FATAL] No balanced data could be sampled. Exiting.")
        return

    all_final_samples = true_samples + hall_samples
    source_id_counts = Counter(s["source_id"] for s in all_final_samples)

    print("\n--- Starting Text Report Generation ---")
    generate_descriptive_stats(
        true_samples,
        hall_samples,
        args.out_dir,
        args.num_layers,
        args.num_heads,
        source_id_counts,
    )
    analyze_and_report_correlations(true_samples, hall_samples, args.out_dir)
    generate_attention_sink_report(
        true_samples, hall_samples, args.out_dir, args.num_layers, args.num_heads
    )

    metric_pairs_to_correlate = [
        ('ecs_prompt_final_layer_per_head', 'bos_attention_proportion_per_head'),
        ('cas_alt_vs_final_layer_per_head', 'bos_attention_proportion_per_head')
    ]
    report_inter_metric_correlations(true_samples, hall_samples, args.out_dir, metric_pairs_to_correlate)

    print("\n--- Starting Plot Generation ---")

    heatmap_tasks = {
        'cas_alt_vs_final_layer_per_head': 'CAS-Alt vs Final', 
        'ecs_prompt_final_layer_per_head': 'ECS (Prompt) vs Final', 
        'bos_attention_proportion_per_head': '<bos> Attention',
    }
    for key, title in tqdm(heatmap_tasks.items(), desc="Generating Heatmaps"):
        plot_generic_heatmap(true_samples, hall_samples, args.out_dir, args.num_layers, args.num_heads, key, title, key)

    plot_all_per_layer_metrics(true_samples, hall_samples, args.out_dir, args.num_layers, args.num_heads)
    plot_single_metric_diff_bar(true_samples, hall_samples, args.out_dir, "pos_cos_similarity_per_layer", "Pathway Orthogonality Score Difference", "pos_plot_difference.png")
    plot_single_metric_diff_bar(true_samples, hall_samples, args.out_dir, "parameter_knowledge_difference", "Parametric Knowledge Score Difference", "pks_plot_difference.png")
    plot_single_metric_diff_bar(true_samples, hall_samples, args.out_dir, "v_ffn_norm_per_layer", "V_ffn L2 Norm Score Difference", "vffn_plot_difference.png")
    plot_correlation_heatmaps(true_samples, hall_samples, args.out_dir, args.num_layers, args.num_heads)
    plot_correlation_barplots(true_samples, hall_samples, args.out_dir)
    plot_top_p_token_type_frequency(true_samples, hall_samples, args.out_dir)
    plot_peak_pathway_force_distribution(true_samples, hall_samples, args.out_dir)
    plot_alignment_conflict_plane(true_samples, hall_samples, args.out_dir)
    plot_conviction_conflict_plane(true_samples, hall_samples, args.out_dir)

    print(
        f"\n--- Unified analysis complete. Plots and reports saved to: {args.out_dir} ---"
    )


if __name__ == "__main__":
    main()
