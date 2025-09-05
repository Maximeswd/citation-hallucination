import json
import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import gc
import os
import re
import random
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_curve,
)
from scipy.stats import pearsonr

# Suppress verbose warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckBERTScore
except ImportError:
    print(
        "Error: 'selfcheckgpt' library not found. Please install with 'pip install selfcheckgpt'"
    )
    exit()

# CONFIGURATION
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# DATA PREP
def robust_read_jsonl(file_path: str):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line.strip()}")
    return pd.DataFrame(records)


def parse_prompt_for_sources(prompt_text: str) -> dict:
    source_map = {}
    if pd.isna(prompt_text) or "---" not in prompt_text:
        return {}
    parts = prompt_text.split("---")
    if len(parts) < 2:
        return {}
    context_section = parts[1]
    pattern = re.compile(
        r"Source: (\d+)\nTitle: .*?\nContent: (.*?)(?=\n\s*Source: \d+|\Z)", re.DOTALL
    )
    for match in pattern.finditer(context_section):
        source_num = int(match.group(1))
        content = match.group(2).strip()
        source_map[source_num] = content
    return source_map


def prepare_evaluation_data(df: pd.DataFrame, nlp, tokenizer) -> pd.DataFrame:
    print("Preparing evaluation data by pre-calculating token spans...")
    all_eval_points = []
    SAFE_MAX_LENGTH = 4096
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing labeled spans"):
        if (
            pd.isna(row.get("prompt"))
            or pd.isna(row.get("response"))
            or not isinstance(row.get("labels"), list)
        ):
            continue
        source_map = parse_prompt_for_sources(row["prompt"])
        response_text = row["response"]
        prompt_text = row["prompt"]
        full_text = prompt_text + response_text
        response_char_len = len(response_text)
        encoding = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=SAFE_MAX_LENGTH,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding["offset_mapping"][0]
        prompt_char_len = len(prompt_text)
        for label in row["labels"]:
            if not (
                isinstance(label.get("start"), int)
                and isinstance(label.get("end"), int)
                and 0 <= label["start"] < label["end"] <= response_char_len
            ):
                continue
            label_start_char = prompt_char_len + label["start"]
            label_end_char = prompt_char_len + label["end"]
            start_token, end_token = -1, -1
            for idx, (start, end) in enumerate(offset_mapping):
                if start_token == -1 and start <= label_start_char < end:
                    start_token = idx
                if end_token == -1 and start < label_end_char <= end:
                    end_token = idx
                    break
            if start_token == -1 or end_token == -1:
                continue
            token_span = (start_token, end_token + 1)
            doc = nlp(response_text)
            containing_sentence = ""
            for sent in doc.sents:
                if sent.start_char <= label["start"] and sent.end_char >= label["end"]:
                    containing_sentence = sent.text
                    break
            if not containing_sentence:
                continue
            is_citation_task = str(label.get("text", "")).isdigit()
            if is_citation_task:
                sentence_to_check = re.sub(
                    r"\[Source: \d+\]\.?", "", containing_sentence
                ).strip()
                citation_num = int(label["text"])
                context = source_map.get(citation_num, " ".join(source_map.values()))
            else:
                sentence_to_check = containing_sentence
                context = row.get("source_info", "")
            all_eval_points.append(
                {
                    "source_id": row["source_id"],
                    "prompt": prompt_text,
                    "full_response": response_text,
                    "sentence_to_check": sentence_to_check,
                    "context": context,
                    "token_span": token_span,
                    "label_text": label["text"],
                    "label": 0 if label["label_type"] == "good" else 1,
                    "is_citation_task": is_citation_task,
                }
            )
    return pd.DataFrame(all_eval_points)


def create_reproducible_balanced_sample(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sampled_dfs = []
    for _, group in df.groupby("source_id"):
        factual = group[group["label"] == 0]
        hallucinated = group[group["label"] == 1]
        sample_size = min(len(factual), len(hallucinated))
        if sample_size > 0:
            sampled_dfs.extend(
                [
                    factual.sample(n=sample_size, random_state=SEED),
                    hallucinated.sample(n=sample_size, random_state=SEED),
                ]
            )
    if not sampled_dfs:
        return pd.DataFrame()
    return (
        pd.concat(sampled_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    )


# BASELINE RUNNER
class BaselineRunner:
    def __init__(self, model_path: str):
        print("\nStep 3: Initializing All Baseline Models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self.temp_cache = {}
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
        self.selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)

    def _get_and_cache_internal_states(self, prompt, full_response):
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt + full_response,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            outputs = self.model(
                input_ids=inputs.input_ids.to(self.device), output_hidden_states=True
            )
            self.temp_cache = {
                "input_ids": inputs.input_ids.cpu(),
                "logits": outputs.logits.cpu(),
                "hidden_states": [hs.cpu() for hs in outputs.hidden_states],
            }

    def clear_cache(self):
        self.temp_cache = {}
        gc.collect()
        torch.cuda.empty_cache()

    def _get_tensors_for_span(self, span_tuple, tensor_name):
        full_tensor = self.temp_cache[tensor_name]
        start, end = span_tuple
        span = slice(start, end)
        if tensor_name == "hidden_states":
            return [hs[:, span, :] for hs in full_tensor]
        elif tensor_name == "logits":
            return full_tensor[:, span.start - 1 : span.stop - 1, :]
        elif tensor_name == "input_ids":
            return full_tensor[:, span.start : span.stop]

    def calculate_perplexity(self, row):
        logits = self._get_tensors_for_span(row["token_span"], "logits")
        target_ids = self._get_tensors_for_span(row["token_span"], "input_ids")
        if (
            logits is None
            or target_ids is None
            or logits.shape[1] == 0
            or target_ids.shape[1] == 0
            or logits.shape[1] != target_ids.shape[1]
        ):
            return np.nan
        loss = F.cross_entropy(
            logits.to(self.device).view(-1, self.model.config.vocab_size),
            target_ids.to(self.device).view(-1),
        )
        return torch.exp(loss).item()

    def calculate_energy_score(self, row):
        logits = self._get_tensors_for_span(row["token_span"], "logits")
        if logits is None or logits.shape[1] == 0:
            return np.nan
        return -torch.logsumexp(logits.to(self.device).float(), dim=-1).mean().item()

    def calculate_ptrue_score(self, row):
        span = row["token_span"]
        hidden_states = self._get_tensors_for_span(span, "hidden_states")
        target_ids = self._get_tensors_for_span(span, "input_ids")
        if hidden_states is None or target_ids is None or target_ids.shape[1] == 0:
            return np.nan
        final_hs, early_hs = hidden_states[-1], hidden_states[2]
        final_logits, early_logits = (
            self.model.lm_head(final_hs.to(self.device)),
            self.model.lm_head(early_hs.to(self.device)),
        )
        final_log_probs = (
            F.log_softmax(final_logits, dim=-1)
            .gather(2, target_ids.to(self.device).unsqueeze(-1))
            .squeeze(-1)
        )
        early_log_probs = (
            F.log_softmax(early_logits, dim=-1)
            .gather(2, target_ids.to(self.device).unsqueeze(-1))
            .squeeze(-1)
        )
        return -(final_log_probs - early_log_probs).mean().item()

    def calculate_ln_entropy(self, row):
        logits = self._get_tensors_for_span(row["token_span"], "logits")
        if logits is None or logits.shape[1] == 0:
            return np.nan
        probs = torch.softmax(logits.to(self.device).float(), dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()

    def run_selfcheck_nli(self, row):
        sentence, context = (
            str(row.get(k, "")) for k in ["sentence_to_check", "context"]
        )
        return (
            self.selfcheck_nli.predict(
                sentences=[sentence], sampled_passages=[context]
            )[0]
            if sentence and context
            else 0.0
        )

    def run_selfcheck_bertscore(self, row):
        sentence, context = (
            str(row.get(k, "")) for k in ["sentence_to_check", "context"]
        )
        return (
            self.selfcheck_bertscore.predict(
                sentences=[sentence], sampled_passages=[context]
            )[0]
            if sentence and context
            else 0.5
        )


# EVALUATION
def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if len(thresholds) == 0:
        return 0.5
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    return optimal_threshold if np.isfinite(optimal_threshold) else 0.5


def evaluate_predictions(y_true, y_scores, metric_name=""):
    y_true_clean, y_scores_clean = [
        np.array(v)
        for v in zip(
            *[
                (yt, ys)
                for yt, ys in zip(y_true, y_scores)
                if ys is not None and not np.isnan(ys)
            ]
        )
    ]
    if len(set(y_true_clean)) < 2:
        return {
            "AUC": np.nan,
            "PCC": np.nan,
            "Accuracy": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
        }
    if metric_name in ["SelfCheck-NLI", "SelfCheck-BERTScore"]:
        y_scores_clean = -y_scores_clean
    threshold = find_optimal_threshold(y_true_clean, y_scores_clean)
    y_pred = (y_scores_clean > threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true_clean, y_scores_clean),
        "PCC": pearsonr(y_true_clean, y_scores_clean)[0],
        "Accuracy": accuracy_score(y_true_clean, y_pred),
        "Recall": recall_score(y_true_clean, y_pred),
        "F1": f1_score(y_true_clean, y_pred),
    }


def run_scoring(args):
    """Runs the full scoring process on a single data file and saves the raw scores."""
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    responses_df = robust_read_jsonl(args.response_file)
    source_info_df = robust_read_jsonl(args.source_info_file)
    merged_df = pd.merge(responses_df, source_info_df, on="source_id", how="left")
    eval_df = prepare_evaluation_data(merged_df, nlp, tokenizer)
    if args.dataset_type == "ragtruth":
        eval_df = create_reproducible_balanced_sample(eval_df)
    if eval_df.empty:
        print("CRITICAL: No data to evaluate after preparation. Exiting.")
        pd.DataFrame().to_csv(args.output_file, index=False)
        return
    runner = BaselineRunner(model_path=args.model_path)
    baselines = {
        "SelfCheck-NLI": runner.run_selfcheck_nli,
        "SelfCheck-BERTScore": runner.run_selfcheck_bertscore,
        "Perplexity": runner.calculate_perplexity,
        "LN-Entropy": runner.calculate_ln_entropy,
        "Energy Score": runner.calculate_energy_score,
        "P(True) Score": runner.calculate_ptrue_score,
    }
    text_baselines = {b for b in baselines if "SelfCheck" in b}
    for name, method in baselines.items():
        print(f"\n--- Running Baseline: {name} ---")
        if name in text_baselines:
            eval_df[name] = eval_df.progress_apply(method, axis=1)
        else:
            eval_df[name] = np.nan
            for source_id, group in tqdm(
                eval_df.groupby("source_id"), desc=f"Processing documents for {name}"
            ):
                runner._get_and_cache_internal_states(
                    group.iloc[0]["prompt"], group.iloc[0]["full_response"]
                )
                eval_df.loc[group.index, name] = group.apply(method, axis=1)
                runner.clear_cache()
    print(f"\n--- Raw scores and labels saved to {args.output_file} ---")
    eval_df.to_csv(args.output_file, index=False)


def run_summarization(args):
    """Combines raw score files from an input directory and computes the final, rounded metrics."""
    score_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ]
    if not score_files:
        print(f"CRITICAL: No CSV files found in {args.input_dir}. Exiting.")
        return
    combined_df = pd.concat((pd.read_csv(f) for f in score_files), ignore_index=True)
    print(
        f"--- Combined {len(score_files)} score files into a single dataset of {len(combined_df)} rows. ---"
    )
    baselines = [
        "SelfCheck-NLI",
        "SelfCheck-BERTScore",
        "Perplexity",
        "LN-Entropy",
        "Energy Score",
        "P(True) Score",
    ]
    all_results = []
    y_true = combined_df["label"].tolist()
    for name in baselines:
        if name not in combined_df.columns:
            continue
        metrics = evaluate_predictions(
            y_true, combined_df[name].tolist(), metric_name=name
        )
        metrics["Baseline"] = name
        all_results.append(metrics)
    results_df = pd.DataFrame(all_results)
    numeric_cols = ["AUC", "PCC", "Accuracy", "Recall", "F1"]
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    print(f"\n--- Final Results Summary saved to {args.output_file} ---")
    print(
        results_df[["Baseline", "AUC", "PCC", "Accuracy", "Recall", "F1"]].to_string(
            index=False
        )
    )
    results_df.to_csv(args.output_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run or summarize hallucination detection baselines.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["score", "summarize"],
        help="Choose 'score' to run on a data split, or 'summarize' to combine raw score files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the output file.\n- For 'score' mode, this is the raw scores CSV for a split.\n- For 'summarize' mode, this is the final summary CSV.",
    )
    score_args = parser.add_argument_group('Arguments for "score" mode')
    score_args.add_argument(
        "--response_file", type=str, help="Path to the response data split."
    )
    score_args.add_argument(
        "--source_info_file", type=str, help="Path to the source info file."
    )
    score_args.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Path to the Hugging Face LLM.",
    )
    score_args.add_argument(
        "--dataset_type",
        type=str,
        choices=["neuclir", "ragtruth"],
        help="Type of dataset (for sampling).",
    )
    summarize_args = parser.add_argument_group('Arguments for "summarize" mode')
    summarize_args.add_argument(
        "--input_dir", type=str, help="Directory containing the raw score CSV files."
    )
    args = parser.parse_args()

    if args.mode == "score":
        if not all([args.response_file, args.source_info_file, args.dataset_type]):
            parser.error(
                "--response_file, --source_info_file, and --dataset_type are required for 'score' mode."
            )
        run_scoring(args)
    elif args.mode == "summarize":
        if not args.input_dir:
            parser.error("--input_dir is required for 'summarize' mode.")
        run_summarization(args)


if __name__ == "__main__":
    main()
