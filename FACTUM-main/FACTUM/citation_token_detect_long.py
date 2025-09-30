# ==============================================================================
# --- IMPORTS & SETUP ---
# ==============================================================================

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
import math
import argparse
import os
import platform
import datetime
import gc

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

# Initial environment and system info logging
print(f"Hostname: {platform.node()}  Date: {datetime.datetime.now().strftime('%c')}")
print("------------------------")
print(f"Python interpreter: {sys.executable}")
print(f"Conda prefix: {os.environ.get('CONDA_PREFIX', 'Not set')}")
print(f" sys.executable: {sys.executable}")
try:
    print(f" torch found:  {torch.__version__}")
except Exception:
    print("torch not found.")
print("------------------------\n")

# Arg parsing
parser = argparse.ArgumentParser(
    description="Script for unified pathway analysis (CAS, POS, ECS, PKS) on long-context models."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.2-3b",
    help="Hugging Face model identifier (e.g., 'meta-llama/Meta-Llama-3.2-3B-Instruct').",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="scale",
    help="Dataset identifier to load the correct paths (e.g., 'scale').",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=1024,
    help="Chunk size for processing long context hidden states and prompt KV-cache.",
)
parser.add_argument(
    "--response_chunk_size",
    type=int,
    default=64,
    help="Number of response tokens to process in a single batch to balance speed and memory.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="/exp/mdassen/redeep/ReDEeP-ICLR-main/ReDeEP/log/llama3.2-3b_scale/not_rel/all_scores_scale_numonly.jsonl",
    help="Directory to save the output log files.",
)
parser.add_argument(
    "--response_path",
    type=str,
    default="/exp/mdassen/redeep/ReDEeP-ICLR-main/dataset/scale/NeuCLIR24/citation/numonly_3.2-3b_response.jsonl",
    help="Path to the .jsonl file containing model responses.",
)
parser.add_argument(
    "--source_info_path",
    type=str,
    default="/exp/mdassen/redeep/ReDEeP-ICLR-main/dataset/scale/NeuCLIR24/citation/numonly_3.2-3b_source_info.jsonl",
    help="Path to the .jsonl file containing source info and prompts.",
)
parser.add_argument(
    "--top_p_attention",
    type=float,
    default=0.1,
    help="Top percentage of attended-to context positions/weights to store for attention analysis (e.g., 0.1 for top 10%).",
)
parser.add_argument(
    "--attention_entropy",
    action="store_true",
    help="Calculate attention entropy and top-k attended positions for each response token.",
)

args = parser.parse_args()

print(
    f"[INFO] Starting unified scoring script with model: {args.model_name}, dataset: {args.dataset}"
)

# Data loading and model setup: make sure to change the directories to downloaded model
if args.model_name == "llama3.2-3b":
    model_path = "/exp/mdassen/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.2-3B-Instruct"
elif args.model_name == "llama3.1-8b":
    model_path = "/exp/mdassen/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
elif args.model_name == "llama2-7b":
    model_path = "/exp/mdassen/.cache/huggingface/hub/models--meta-llama--Meta-Llama-2-7B-chat-hf"
else:
    ValueError(
        f"[ERROR] No model was found, adjust the 'model_path' to your downloaded model path"
    )


print(f"[INFO] Loading responses from: {args.response_path}")
responses = [json.loads(line) for line in open(args.response_path, "r")]
print(f"[INFO] Loading source info from: {args.source_info_path}")
source_info_dict = {
    data["source_id"]: data
    for data in (json.loads(line) for line in open(args.source_info_path, "r"))
}


if args.model_name == "llama3.2-3b":
    data_type = "llama-3.2-3b-instruct"
elif args.model_name == "llama3.1-8b":
    data_type = "llama-3.1-8b-instruct"
elif args.model_name == "llama2-7b":
    data_type = "llama-2-7b-chat"
else:
    data_type = os.path.basename(model_path).lower()
    print(
        f"[WARNING] Automatically determined data_type as '{data_type}'. Please verify this is correct."
    )


print(f"[INFO] Loading model and tokenizer from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.float16,
)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = model.device
print(f"[INFO] Model loaded on primary device: {device}")

# Sanity check: Print model weights to see if they are correctly loaded
print("\n--- Model Parameter Sanity Check ---")
for n, p in model.named_parameters():
    if p.data.is_floating_point():
        display_name = n[-80:] if len(n) > 80 else n
        print(f"{display_name:<80} | mean_abs: {p.data.abs().mean().item():.6f}")
print("-------------------------------------\n")


# Get BOS token info
bos_token_id = tokenizer.bos_token_id
if bos_token_id is None:
    # Fallback for models that might not have a bos_token set in the config
    tqdm.write(
        f"[WARNING] bos_token_id not found in config, attempting to find it manually."
    )
    bos_token_id = tokenizer.get_vocab().get("<bos>") or tokenizer.get_vocab().get(
        "<s>"
    )
bos_tensor = torch.tensor([[bos_token_id]], dtype=torch.long)


digit_token_ids = [
    token_id
    for token, token_id in tokenizer.get_vocab().items()
    if token.strip().isdigit()
]
digit_token_ids_tensor = torch.tensor(digit_token_ids, device=device)
print(f"[INFO] Identified {len(digit_token_ids)} digit tokens in the vocabulary.")


def get_labeled_token_map(labels, response_rag, tokenizer, prefix_len):
    """
    Maps character-level spans from labels to token-level indices.
    """
    token_map = {}
    if not labels:
        return token_map
    response_tokens = tokenizer(
        response_rag, return_offsets_mapping=True, add_special_tokens=False
    )
    offsets = response_tokens["offset_mapping"]
    for label in labels:
        span_start_char, span_end_char = label["start"], label["end"] - 1
        start_token_idx, end_token_idx = -1, -1
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start <= span_start_char < tok_end:
                start_token_idx = i
                break
        if start_token_idx != -1:
            for i in range(start_token_idx, len(offsets)):
                tok_start, tok_end = offsets[i]
                if tok_start <= span_end_char < tok_end:
                    end_token_idx = i
                    break
        if start_token_idx != -1 and end_token_idx != -1:
            code = 0 if label["label_type"] == "good" else 1
            for i in range(start_token_idx, end_token_idx + 1):
                token_map[prefix_len + i] = code
    return token_map


# PKS helper function
def calculate_pks_and_dists(pre, post):
    """Calculates PKS score (JSD) and returns probability distributions."""
    s_pre = F.softmax(pre, dim=-1, dtype=torch.float32)
    s_post = F.softmax(post, dim=-1, dtype=torch.float32)
    log_soft_pre = F.log_softmax(pre, dim=-1, dtype=torch.float32)
    log_soft_post = F.log_softmax(post, dim=-1, dtype=torch.float32)
    M = 0.5 * (s_pre + s_post)
    kl_div1 = F.kl_div(log_soft_pre, M, reduction="sum")
    kl_div2 = F.kl_div(log_soft_post, M, reduction="sum")
    pks_score = 0.5 * (kl_div1 + kl_div2).cpu().item() * 10e5
    logit_stats = {
        "pre_mean": pre.mean().item(),
        "pre_std": pre.std().item(),
        "post_mean": post.mean().item(),
        "post_std": post.std().item(),
    }
    return pks_score, s_pre, s_post, logit_stats


# Main processing loop
all_results = []
print(f"[INFO] Starting main loop. Expecting data_type '{data_type}'.")

# Define layers (here you could also specify a subset of layers if desired)
knowledge_layers_for_pks = list(range(model.config.num_hidden_layers))
print(f"[INFO] Scoring with {knowledge_layers_for_pks} number of layers.")

for item in tqdm(responses, desc="Overall Progress"):
    if not (item.get("model") == data_type and item.get("labels")):
        continue
    tqdm.write(f"\n[INFO] Processing item with source_id: {item['source_id']}")
    source_item = source_info_dict[item["source_id"]]
    full_prompt_text = source_item["prompt"]
    source_context = source_item["context"]
    
    # Sanity check print statements

    # tqdm.write("\n" + "#" * 80)
    # tqdm.write(f"[DEBUG] FULL PROMPT FOR SID: {item['source_id']}")
    # tqdm.write(full_prompt_text)
    # tqdm.write("#" * 80 + "\n")

    if not source_context:
        tqdm.write(
            f"[ERROR] Context for source_id {item['source_id']} is empty. Skipping."
        )
        continue

    if source_context not in full_prompt_text:
        tqdm.write(
            f"[ERROR] CRITICAL: Context string not found in prompt string for SID {item['source_id']}. Skipping."
        )
        continue

    prompt_parts = full_prompt_text.split(source_context, 1)
    prompt_before_context, prompt_after_context = prompt_parts

    if len(prompt_parts) != 2:
        tqdm.write(
            f"[WARNING] Context string split prompt into {len(prompt_parts)} parts for SID {item['source_id']}. Skipping."
        )
        continue

    instruction_delimiter = "\n\n**Instructions:**"
    if instruction_delimiter in prompt_after_context:
        query_text, instruct_text = prompt_after_context.split(instruction_delimiter, 1)
        instruct_text = instruction_delimiter + instruct_text
    else:
        query_text = prompt_after_context
        instruct_text = ""
        tqdm.write(
            f"[WARNING] Instruction delimiter not found for SID {item['source_id']}. Treating all post-context as query."
        )

    # More sanity check print statements
    # tqdm.write("\n" + "-" * 80)
    # tqdm.write("[DEBUG] PARSED PROMPT COMPONENTS")
    # tqdm.write(f"--- [STARTING PROMPT] ---\n{prompt_before_context}")
    # tqdm.write(f"--- [CONTEXT] ---\n{source_context[:200]}... (truncated)")
    # tqdm.write(f"--- [QUESTION] ---\n{query_text}")
    # tqdm.write(f"--- [INSTRUCTIONS] ---\n{instruct_text}")
    # tqdm.write("-" * 80 + "\n")

    prompt_before_ids = tokenizer(
        prompt_before_context, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    context_ids = tokenizer(
        source_context, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    query_ids = tokenizer(
        query_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    instruct_ids = tokenizer(
        instruct_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    bos_len = 1
    prompt_len = prompt_before_ids.shape[1]
    context_len = context_ids.shape[1]
    query_len = query_ids.shape[1]
    instruct_len = instruct_ids.shape[1]

    prompt_start_idx = bos_len
    prompt_end_idx = prompt_start_idx + prompt_len
    context_start_idx = prompt_end_idx
    context_end_idx = context_start_idx + context_len
    query_start_idx = context_end_idx
    query_end_idx = query_start_idx + query_len
    instruct_start_idx = query_end_idx
    instruct_end_idx = instruct_start_idx + instruct_len

    prefix_ids = torch.cat(
        [bos_tensor, prompt_before_ids, context_ids, query_ids, instruct_ids], dim=1
    )
    prefix_len = prefix_ids.shape[1]

    tqdm.write("\n" + "-" * 80)
    tqdm.write("[DEBUG] TOKENIZATION & INDEXING INFO")
    tqdm.write(f"BOS len: {bos_len},                  Indices: 0 to {bos_len}")
    tqdm.write(
        f"Prompt len: {prompt_len},            Indices: {prompt_start_idx} to {prompt_end_idx}"
    )
    tqdm.write(
        f"Context len: {context_len},         Indices: {context_start_idx} to {context_end_idx}"
    )
    tqdm.write(
        f"Query len: {query_len},             Indices: {query_start_idx} to {query_end_idx}"
    )
    tqdm.write(
        f"Instruct len: {instruct_len},          Indices: {instruct_start_idx} to {instruct_end_idx}"
    )
    tqdm.write(f"Total Prefix Length: {prefix_len}")
    tqdm.write("-" * 80 + "\n")

    response_rag = item["response"]
    response_token_ids = tokenizer(
        response_rag, return_tensors="pt", add_special_tokens=False
    ).input_ids
    response_len = response_token_ids.shape[1]

    labeled_token_map = get_labeled_token_map(
        item.get("labels", []), response_rag, tokenizer, prefix_len
    )

    if not labeled_token_map:
        tqdm.write(
            f"[INFO] No scorable tokens found for item {item['source_id']}. Skipping."
        )
        continue

    tqdm.write(
        f"[INFO] Total Prefix Length: {prefix_len}, Context Length: {context_len}"
    )
    tqdm.write(
        f"[INFO] Context found at token indices: {context_start_idx} to {context_end_idx}"
    )
    tqdm.write(f"[INFO] Scored Token Map (Position: Label): {labeled_token_map}")
    scored_tokens_summary = {
        pos: tokenizer.decode(response_token_ids[0, pos - prefix_len])
        for pos in sorted(labeled_token_map.keys())
    }
    tqdm.write(f"[INFO] Tokens to be scored: {scored_tokens_summary}")

    item_results = {"source_id": item["source_id"], "token_data": []}

    tqdm.write("[INFO] Pre-calculating context hidden states...")

    unique_labels = set(labeled_token_map.values())
    if len(unique_labels) <= 1:
        tqdm.write(
            f"[INFO] Skipping source_id {item['source_id']} because all of its labels are identical ({list(unique_labels)}). This provides no contrast for analysis."
        )
        continue

    # Pre-calculate hidden states for the entire context: efficient for long-contexts
    context_hs_all_layers = [[] for _ in range(model.config.num_hidden_layers + 1)]
    with torch.no_grad():
        for chunk_start_ctx in range(0, context_len, args.chunk_size):
            end_pos = min(chunk_start_ctx + args.chunk_size, context_len)
            chunk_ids_ctx = context_ids[:, chunk_start_ctx:end_pos]
            chunk_outputs = model(
                input_ids=chunk_ids_ctx.to(model.device),
                output_hidden_states=True,
                return_dict=True,
            )
            for layer_idx in range(len(chunk_outputs.hidden_states)):
                context_hs_all_layers[layer_idx].append(
                    chunk_outputs.hidden_states[layer_idx].cpu()
                )
            del chunk_outputs
            torch.cuda.empty_cache()
    context_hs_per_layer = tuple(
        torch.cat(layer_hs, dim=1) for layer_hs in context_hs_all_layers
    )
    tqdm.write("[INFO] Context hidden states pre-calculated successfully.")

    # Pre-calculate hidden states for the entire prompt: This is needed for long-contexts/prompts
    tqdm.write("[INFO] Pre-calculating prompt hidden states...")
    prompt_hs_all_layers = [[] for _ in range(model.config.num_hidden_layers + 1)]
    with torch.no_grad():
        for chunk_start_hs in range(0, prefix_len, args.chunk_size):
            end_pos = min(chunk_start_hs + args.chunk_size, prefix_len)
            chunk_ids_hs = prefix_ids[:, chunk_start_hs:end_pos]
            chunk_outputs = model(
                input_ids=chunk_ids_hs.to(model.device),
                output_hidden_states=True,
                return_dict=True,
            )
            for layer_idx in range(len(chunk_outputs.hidden_states)):
                prompt_hs_all_layers[layer_idx].append(
                    chunk_outputs.hidden_states[layer_idx].cpu()
                )
            del chunk_outputs
            torch.cuda.empty_cache()
    prompt_hs_per_layer = tuple(
        torch.cat(layer_hs, dim=1) for layer_hs in prompt_hs_all_layers
    )
    tqdm.write("[INFO] Prompt hidden states pre-calculated successfully.")

    past_key_values = None
    with torch.no_grad():
        for chunk_start_kv in tqdm(
            range(0, prefix_len, args.chunk_size), desc="Prefix KV", leave=False
        ):
            chunk_ids = prefix_ids[
                :, chunk_start_kv : min(chunk_start_kv + args.chunk_size, prefix_len)
            ].to(model.device)
            cache_outputs = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = cache_outputs.past_key_values
            del cache_outputs
            torch.cuda.empty_cache()

    # Main response processing loop
    for chunk_start in tqdm(
        range(0, response_len, args.response_chunk_size),
        desc=f"Analyzing Chunks for SID {item['source_id']}",
        leave=False,
    ):
        chunk_end = min(chunk_start + args.response_chunk_size, response_len)
        response_chunk_ids = response_token_ids[:, chunk_start:chunk_end].to(
            model.device
        )
        chunk_len = response_chunk_ids.shape[1]
        num_layers = model.config.num_hidden_layers

        with torch.no_grad():
            custom_outputs_gpu, model_outputs_gpu = model(
                input_ids=response_chunk_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                output_gpa_states=True,
                knowledge_layers=knowledge_layers_for_pks,
            )

            gpa_states_gpu = custom_outputs_gpu["gpa_states"]
            pks_logits_dict_gpu = custom_outputs_gpu["pks_logits"]
            past_key_values = model_outputs_gpu.past_key_values

        # Initialize result lists for all scores for the current chunk
        pos_scores_per_layer_chunk = [[] for _ in range(chunk_len)]
        pos_cos_similarities_per_layer_chunk = [[] for _ in range(chunk_len)]
        v_attn_norms_per_layer_chunk = [[] for _ in range(chunk_len)]
        v_ffn_norms_per_layer_chunk = [[] for _ in range(chunk_len)]
        bos_attention_proportion_per_head_chunk = [[] for _ in range(chunk_len)]
        context_direction_vectors_per_layer_chunk = [[] for _ in range(chunk_len)]
        ecs_prompt_final_per_head_chunk = [[] for _ in range(chunk_len)] 
        cas_alt_vs_final_layer_per_head_chunk = [[] for _ in range(chunk_len)]
        h_final_response = model_outputs_gpu.hidden_states[-1][0]

        for layer_idx in range(num_layers):
            layer_device = model.model.layers[layer_idx].self_attn.v_proj.weight.device

            # POS scores
            gpa_layer_states = gpa_states_gpu[layer_idx]
            x_input_b, x_pre_ffn_b = (
                gpa_layer_states["input"][0],
                gpa_layer_states["pre_ffn"][0],
            )
            v_ffn_b, x_post_ffn_b = (
                gpa_layer_states["ffn_update"][0],
                x_pre_ffn_b + gpa_layer_states["ffn_update"][0],
            )
            v_attn_b = x_pre_ffn_b - x_input_b
            cos_sim_b = F.cosine_similarity(v_attn_b, v_ffn_b, dim=1)
            for i in range(chunk_len):
                pos_cos_similarities_per_layer_chunk[i].append(cos_sim_b[i].item())
                pos_scores_per_layer_chunk[i].append(1.0 - abs(cos_sim_b[i].item()))
                v_attn_norms_per_layer_chunk[i].append(
                    torch.linalg.norm(v_attn_b[i]).item()
                )
                v_ffn_norms_per_layer_chunk[i].append(
                    torch.linalg.norm(v_ffn_b[i]).item()
                )

            # CAS & ECS SCORE
            # ecs_prompt_final: this is the baseline ECS score
            # cas_alt_vs_final: that's ecs adjusted for context tokens only and are either attention weighted 
            attn_layer = model_outputs_gpu.attentions[layer_idx][0].to(layer_device)
            bos_attentions_for_chunk = attn_layer[:, :, 0]
            for i in range(chunk_len):
                bos_attention_proportion_per_head_chunk[i].extend(
                    bos_attentions_for_chunk[:, i].tolist()
                )

            h_final_for_layer = h_final_response.to(layer_device)
            h_current_layer_output = model_outputs_gpu.hidden_states[layer_idx + 1][0]
            h_current_for_layer = h_current_layer_output.to(layer_device)
            attn_to_context = attn_layer[:, :, context_start_idx:context_end_idx]

            if attn_to_context.shape[-1] > 0 and context_hs_per_layer is not None:
                hs_context_layer = context_hs_per_layer[layer_idx][0].to(layer_device)
                hs_context_final_layer_for_cas = context_hs_per_layer[-1][0].to(
                    layer_device
                )

                if attn_to_context.shape[-1] == hs_context_layer.shape[0]:
                    clean_attentions = attn_to_context / (
                        attn_to_context.sum(dim=-1, keepdim=True) + 1e-9
                    )
                    
                    # Alternative CAS: Use final layer's HS for CDV
                    cdv_alt_per_head = torch.matmul(
                        clean_attentions, hs_context_final_layer_for_cas
                    )
                    cas_alt_vs_final = F.cosine_similarity(
                        cdv_alt_per_head, h_final_for_layer.unsqueeze(0), dim=-1
                    )
                else:  # Handle mismatch
                    cas_alt_vs_final = cas_alt_vs_current = torch.zeros(
                        (attn_layer.shape[1], chunk_len), device=layer_device
                    )

            else:  # Handle empty context slice
                cas_alt_vs_final = cas_alt_vs_current = torch.zeros(
                    (attn_layer.shape[1], chunk_len), device=layer_device
                )

            for i in range(chunk_len):
                cas_alt_vs_final_layer_per_head_chunk[i].extend(
                    cas_alt_vs_final[:, i].tolist()
                )

            # ECS_prompt CALCULATION (all prompt tokens) == baseline from ReDeEP
            k_attn_ecs_prompt = max(1, math.ceil(prefix_len * 0.10))
            attn_to_prompt = attn_layer[:, :, :prefix_len]

            if attn_to_prompt.shape[-1] > 0 and prompt_hs_per_layer is not None:
                topk_indices_prompt = torch.topk(
                    attn_to_prompt, k_attn_ecs_prompt, dim=-1
                ).indices

                hs_prompt_final_layer = prompt_hs_per_layer[-1][0].to(layer_device)
                hs_prompt_specific_layer = prompt_hs_per_layer[layer_idx + 1][0].to(
                    layer_device
                )

                num_heads, hidden_size = (
                    model.config.num_attention_heads,
                    model.config.hidden_size,
                )

                for i in range(chunk_len):
                    topk_indices_token = topk_indices_prompt[:, i, :]
                    expanded_indices = topk_indices_token.unsqueeze(-1).expand(
                        -1, -1, hidden_size
                    )

                    # ECS_prompt vs Final
                    attended_hs_final = (
                        hs_prompt_final_layer.unsqueeze(0)
                        .expand(num_heads, -1, -1)
                        .gather(1, expanded_indices)
                    )
                    ecs_prompt_final_token = F.cosine_similarity(
                        attended_hs_final.mean(dim=1),
                        h_final_for_layer[i].unsqueeze(0),
                        dim=1,
                    )
                    ecs_prompt_final_per_head_chunk[i].extend(
                        ecs_prompt_final_token.tolist()
                    )

                    # ECS_prompt vs Specific
                    attended_hs_specific = (
                        hs_prompt_specific_layer.unsqueeze(0)
                        .expand(num_heads, -1, -1)
                        .gather(1, expanded_indices)
                    )
                    ecs_prompt_specific_token = F.cosine_similarity(
                        attended_hs_specific.mean(dim=1),
                        h_current_for_layer[i].unsqueeze(0),
                        dim=1,
                    )

                del hs_prompt_final_layer, hs_prompt_specific_layer
            else:
                zeros = [0.0] * model.config.num_attention_heads
                for i in range(chunk_len):
                    ecs_prompt_final_per_head_chunk[i].extend(zeros)


            del attn_layer, h_final_for_layer, cas_alt_vs_final

            h_current_layer_for_new_metrics = model_outputs_gpu.hidden_states[
                layer_idx + 1
            ][0].to(layer_device)
            attn_layer_for_new_metrics = model_outputs_gpu.attentions[layer_idx][0].to(
                layer_device
            )

        # PKS caclulation
        pks_scores_chunk = [[] for _ in range(chunk_len)]
        pks_dists_chunk = [{} for _ in range(chunk_len)]

        # Determine k for top-k, ensuring it's not more than the number of available digit tokens
        k_pks_digits = min(20, len(digit_token_ids))

        for layer_idx_str, (pre_logits, post_logits) in pks_logits_dict_gpu.items():
            layer_idx = int(layer_idx_str)
            for i in range(chunk_len):
                pre_logits_token, post_logits_token = (
                    pre_logits[0, i, :],
                    post_logits[0, i, :],
                )
                pks_score, s_pre, s_post, logit_stats = calculate_pks_and_dists(
                    pre_logits_token, post_logits_token
                )
                pks_scores_chunk[i].append(pks_score)

                # Filter probabilities to only include digit tokens
                pre_digit_probs = s_pre[digit_token_ids_tensor]
                post_digit_probs = s_post[digit_token_ids_tensor]

                # Get the top k from the filtered digit probabilities
                top_probs_pre, top_indices_pre = torch.topk(
                    pre_digit_probs, k_pks_digits
                )
                top_probs_post, top_indices_post = torch.topk(
                    post_digit_probs, k_pks_digits
                )

                # Map the indices back to the original token IDs from the vocabulary
                original_token_ids_pre = digit_token_ids_tensor[top_indices_pre]
                original_token_ids_post = digit_token_ids_tensor[top_indices_post]

                pks_dists_chunk[i][layer_idx] = {
                    "logit_stats": logit_stats,
                    "pre_dist_top_20_digits": list(
                        zip(
                            tokenizer.convert_ids_to_tokens(
                                original_token_ids_pre.cpu()
                            ),
                            top_probs_pre.cpu().tolist(),
                        )
                    ),
                    "post_dist_top_20_digits": list(
                        zip(
                            tokenizer.convert_ids_to_tokens(
                                original_token_ids_post.cpu()
                            ),
                            top_probs_post.cpu().tolist(),
                        )
                    ),
                }

        del gpa_states_gpu, pks_logits_dict_gpu, model_outputs_gpu
        gc.collect()
        torch.cuda.empty_cache()

        # Assemble final results for tokens in this chunk
        for i in range(chunk_len):
            current_global_pos = prefix_len + chunk_start + i
            if current_global_pos not in labeled_token_map:
                continue

            # Print full history for the current scored token for sanity checks
            token_to_be_scored_str = tokenizer.decode(
                response_token_ids[0, chunk_start + i]
            )
            history_ids = torch.cat(
                [prefix_ids, response_token_ids[:, : chunk_start + i]], dim=1
            )
            history_text = tokenizer.decode(history_ids[0], skip_special_tokens=True)

            # tqdm.write("\n" + "=" * 80)
            # tqdm.write(
            #     f"[DEBUG] HISTORY FOR TOKEN #{current_global_pos} ('{token_to_be_scored_str}')"
            # )
            # tqdm.write(
            #     f"[DEBUG] This token is the {chunk_start + i + 1}-th token in the response."
            # )
            # tqdm.write(
            #     f"[DEBUG] Full Input History (Prefix + Previous Response Tokens):\n--- START HISTORY ---\n{history_text}\n--- END HISTORY ---"
            # )
            # tqdm.write("=" * 80 + "\n")

            final_token_data = {
                "token_index": current_global_pos,
                "token_str": tokenizer.decode(response_token_ids[0, chunk_start + i]),
                "label": labeled_token_map[current_global_pos],
                # CAS/POS metrics
                "pos_score_per_layer": pos_scores_per_layer_chunk[i],
                "pos_cos_similarity_per_layer": pos_cos_similarities_per_layer_chunk[i],
                "v_attn_norm_per_layer": v_attn_norms_per_layer_chunk[i],
                "v_ffn_norm_per_layer": v_ffn_norms_per_layer_chunk[i],
                "cas_alt_vs_final_layer_per_head": cas_alt_vs_final_layer_per_head_chunk[
                    i
                ],
                "bos_attention_proportion_per_head": bos_attention_proportion_per_head_chunk[
                    i
                ],
                # ReDeEP scores
                "ecs_prompt_final_layer_per_head": ecs_prompt_final_per_head_chunk[i],
                "parameter_knowledge_difference": pks_scores_chunk[i],
            }
            item_results["token_data"].append(final_token_data)

    all_results.append(item_results)

    tqdm.write(f"\n[INFO] Finished item {item['source_id']}. Cleaning up memory...")

    if "past_key_values" in locals():
        del past_key_values
    if "context_hs_per_layer" in locals():
        del context_hs_per_layer
    if "prompt_hs_per_layer" in locals():
        del prompt_hs_per_layer
    if "gpa_states_gpu" in locals():
        del gpa_states_gpu
    if "model_outputs_gpu" in locals():
        del model_outputs_gpu

    gc.collect()
    torch.cuda.empty_cache()

    tqdm.write(f"[INFO] Memory cleanup complete for item {item['source_id']}.")


# Save final output
save_dir = os.path.dirname(args.save_dir)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)

with open(args.save_dir, "w") as f:
    for res in all_results:
        f.write(json.dumps(res) + "\n")
print(f"\n[INFO] Unified scoring complete. Detailed results saved to: {args.save_dir}")
