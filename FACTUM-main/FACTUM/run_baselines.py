# ==============================================================================
# --- IMPORTS & SETUP ---
# ==============================================================================

import json
import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn.functional as F
import argparse
from tqdm.auto import tqdm
import gc
import os
import re
import random
from collections import defaultdict
from sklearn.utils import resample
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_score,
)
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# --- CONFIGURATION & HELPERS ---
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def robust_read_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")

def is_numeric(s: str) -> bool:
    if not isinstance(s, str) or not s: return False
    s = s.strip()
    if s in ['.', '-', '+']: return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_labeled_token_map_numeric_only(labels, response_text, tokenizer, prefix_len):
    token_map = {}
    if not labels: return token_map
    response_tokens = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = response_tokens["offset_mapping"]

    for label in labels:
        label_text = str(label.get("text", ""))
        if not (isinstance(label.get("start"), int) and is_numeric(label_text)): continue
        span_start_char, span_end_char = label["start"], label["end"]
        start_token_idx, end_token_idx = -1, -1
        for i, (tok_start, tok_end) in enumerate(offsets):
            if start_token_idx == -1 and tok_start <= span_start_char < tok_end: start_token_idx = i
            if start_token_idx != -1 and tok_start < span_end_char <= tok_end:
                end_token_idx = i
                break
        if start_token_idx != -1 and end_token_idx == -1: end_token_idx = start_token_idx

        if start_token_idx != -1 and end_token_idx != -1:
            label_code = 0 if label["label_type"] == "good" else 1
            for i in range(start_token_idx, end_token_idx + 1):
                token_id = response_tokens['input_ids'][i]
                token_text = tokenizer.decode(token_id)
                if is_numeric(token_text):
                    token_map[prefix_len + i] = {"label": label_code, "char_start": label['start']}
    return token_map

# ==============================================================================
# --- EVALUATION FRAMEWORK ---
# ==============================================================================

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if len(thresholds) == 0: return 0.5
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def evaluate_fold(train_df, test_df, baseline_name):
    """Evaluates a single fold of cross-validation with corrected score logic."""
    y_test = test_df["label"].values
    scores_test = test_df[baseline_name].values
    
    valid_test = ~np.isnan(scores_test)
    if not np.any(valid_test) or len(np.unique(y_test[valid_test])) < 2:
        return {"AUC": np.nan, "PCC": np.nan, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan}
    
    y_test_clean = y_test[valid_test]
    scores_test_clean = scores_test[valid_test]
    
    if "P(True)" in baseline_name:
        # P(True): High score = good (0), Low score = bad (1)
        threshold = 0.5
        y_pred = (scores_test_clean < threshold).astype(int)
        roc_scores = 1 - scores_test_clean  # Invert so higher score means "bad" for AUC
        pcc_scores = scores_test_clean
    else:
        # Perplexity, etc: High score = bad (1), Low score = good (0)
        y_train, scores_train = train_df["label"].values, train_df[baseline_name].values
        valid_train = ~np.isnan(scores_train)
        if not np.any(valid_train) or len(np.unique(y_train[valid_train])) < 2:
             return {"AUC": np.nan, "PCC": np.nan, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan}
        
        threshold = find_optimal_threshold(y_train[valid_train], scores_train[valid_train])
        y_pred = (scores_test_clean > threshold).astype(int)
        roc_scores = scores_test_clean  # Higher score already means "bad" for AUC
        pcc_scores = scores_test_clean

    return {
        "AUC": roc_auc_score(y_test_clean, roc_scores),
        "PCC": pearsonr(y_test_clean, pcc_scores)[0] * (-1 if "P(True)" in baseline_name else 1),
        "Accuracy": accuracy_score(y_test_clean, y_pred),
        "Precision": precision_score(y_test_clean, y_pred, zero_division=0),
        "Recall": recall_score(y_test_clean, y_pred, zero_division=0),
        "F1": f1_score(y_test_clean, y_pred, zero_division=0),
    }


# ==============================================================================
# --- MAIN ORCHESTRATOR ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for internal confidence baselines on numeric tokens.")
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--source_info_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for CV.")
    args = parser.parse_args()

    print("--- Loading Model, Tokenizer, and NLP ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto").eval()
    device = model.device
    nlp = spacy.load("en_core_web_sm")

    print("--- Calculating All Internal Baseline Scores on Numeric Tokens Only ---")
    responses_data = list(robust_read_jsonl(args.response_file))
    source_info_dict = {item["source_id"]: item for item in robust_read_jsonl(args.source_info_file)}
    all_token_data = []

    true_token_ids = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok in ["True", " true", " True"]]
    true_token_ids = list(set(true_token_ids))

    for item in tqdm(responses_data, desc="Processing Documents"):
        source_id = item.get("source_id")
        if not source_id or not item.get("labels"): continue
        source_item = source_info_dict.get(source_id)
        if not source_item: continue

        prompt_text, response_text = source_item["prompt"], item["response"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt").input_ids
        prefix_len = prompt_ids.shape[1]
        
        labeled_token_map = get_labeled_token_map_numeric_only(item["labels"], response_text, tokenizer, prefix_len)
        if not labeled_token_map: continue

        response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt").input_ids
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        past_key_values = None
        all_logits_chunks = []

        with torch.no_grad():
            for chunk_start in range(0, full_ids.shape[1], 512):
                chunk_end = min(chunk_start + 512, full_ids.shape[1])
                input_chunk = full_ids[:, chunk_start:chunk_end].to(device)
                
                outputs = model(input_ids=input_chunk, use_cache=True, past_key_values=past_key_values)
                
                all_logits_chunks.append(outputs.logits.cpu())
                past_key_values = outputs.past_key_values
        
        logits = torch.cat(all_logits_chunks, dim=1)
        
        doc = nlp(response_text)
        sents = list(doc.sents)
        
        for token_idx, label_info in labeled_token_map.items():
            if token_idx >= full_ids.shape[1] or token_idx < prefix_len: continue
            
            token_logits = logits[:, token_idx - 1, :]
            target_id = full_ids[:, token_idx]
            
            loss = F.cross_entropy(token_logits.view(-1, model.config.vocab_size), target_id.view(-1))
            perplexity = torch.exp(loss).item()
            probs = torch.softmax(token_logits.float(), dim=-1)
            ln_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
            energy_score = -torch.logsumexp(token_logits.float(), dim=-1).mean().item()

            sentence_to_check = ""
            for i, sent in enumerate(sents):
                if sent.start_char <= label_info['char_start'] < sent.end_char:
                    cleaned_sent = re.sub(r"\[Source: \d+\]\.?", "", sent.text).strip()
                    if not cleaned_sent and i > 0:
                        sentence_to_check = sents[i-1].text.strip()
                    else:
                        sentence_to_check = cleaned_sent
                    break
            
            ptrue_query_score = np.nan
            if sentence_to_check:
                query_prompt = f"Is the statement \"{sentence_to_check}\" true or false? Answer:"
                query_ids = tokenizer(query_prompt, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    query_logits = model(query_ids).logits.cpu()
                
                next_token_logits = query_logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                ptrue_query_score = next_token_probs[true_token_ids].sum().item()
            
            all_token_data.append({
                "source_id": source_id, "label": label_info['label'],
                "Perplexity": perplexity, "LN-Entropy": ln_entropy,
                "Energy Score": energy_score, "P(True) Query": ptrue_query_score
            })
        
        del past_key_values, all_logits_chunks, logits, full_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_df = pd.DataFrame(all_token_data)
    if eval_df.empty:
        print("\nCRITICAL: No scorable numeric data points were found. Exiting.")
        return
        
    print(f"\nTotal numeric tokens scored: {len(eval_df)}")
    print(f"Class distribution of scored tokens:\n{eval_df['label'].value_counts()}")
    
    print(f"\n--- Running 10-Repeat, {args.n_splits}-Fold Group-Stratified Cross-Validation ---")
    baselines = ["Perplexity", "LN-Entropy", "Energy Score", "P(True) Query"]
    all_results = defaultdict(lambda: defaultdict(list))
    
    for i in tqdm(range(10), desc=f"CV Repeats ({args.n_splits}-Fold)"):
        sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED + i)
        
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(eval_df, eval_df["label"], eval_df["source_id"])):
            train_df_imbalanced, test_df = eval_df.iloc[train_idx], eval_df.iloc[test_idx]
            
            minority = train_df_imbalanced[train_df_imbalanced["label"] == 1]
            majority = train_df_imbalanced[train_df_imbalanced["label"] == 0]
            if len(minority) == 0 or len(majority) == 0: continue
            
            majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=SEED)
            train_df_balanced = pd.concat([minority, majority_downsampled])
            
            if i == 0 and fold_idx == 0:
                 print(f"\n--- Example Fold Data Split (approx. {100*(args.n_splits-1)/args.n_splits:.0f}/{100/args.n_splits:.0f}) ---")
                 print(f"Total Train Samples (Balanced for thresholding): {len(train_df_balanced)}")
                 print(f"  - Class distribution: {train_df_balanced['label'].value_counts().to_dict()}")
                 print(f"Total Test Samples: {len(test_df)}")
                 print(f"  - Class distribution: {test_df['label'].value_counts().to_dict()}")
                 print("---------------------------------\n")

            for baseline_name in baselines:
                fold_metrics = evaluate_fold(train_df_balanced, test_df, baseline_name)
                for metric, value in fold_metrics.items(): all_results[baseline_name][metric].append(value)
    
    print("\n--- Final Cross-Validated Results ---")
    final_summary = []
    for baseline_name in baselines:
        if baseline_name in all_results:
            avg_metrics = {"Baseline": baseline_name}
            for metric, values in all_results[baseline_name].items():
                avg_metrics[metric] = np.nanmean(values)
            final_summary.append(avg_metrics)
    
    if not final_summary:
        print("No results were generated. Please check your data and code.")
        return

    results_df = pd.DataFrame(final_summary).round(4)
    print(results_df.to_string(index=False))
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()