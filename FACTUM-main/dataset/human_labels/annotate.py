import json
import random
import os
import pandas as pd
import re
import time


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


FILE_PAIRS = [
    {
        "name": "llama3.1-8b",
        "resp": "citation-hallucination/FACTUM-main/dataset/NeuCLIR24/llama3.1-8b_neuclir24_response.jsonl",
        "src": "citation-hallucination/FACTUM-main/dataset/NeuCLIR24/llama3.1-8b_neuclir24_source_info.jsonl",
    },
    {
        "name": "llama3.2-3b",
        "resp": "citation-hallucination/FACTUM-main/dataset/NeuCLIR24/llama3.2-3b_neuclir24_response.jsonl",
        "src": "citation-hallucination/FACTUM-main/dataset/NeuCLIR24/llama3.2-3b_neuclir24_source_info.jsonl",
    },
]

SAMPLES_PER_MODEL = 50
TOTAL_TARGET = SAMPLES_PER_MODEL * len(FILE_PAIRS)
SEED = 42
OUTPUT_FILE = "human_annotations.csv"

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "of",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "which",
    "who",
    "whom",
    "whose",
    "as",
    "until",
    "while",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
}


def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_full_sentence(text, start, end):
    before = text[:start]
    start_match = list(re.finditer(r"(^|(?<=[.!?])\s+|\n)", before))
    s_start = start_match[-1].end() if start_match else 0
    after = text[end:]
    end_match = re.search(r"([.!?](\s+|$))|\n|$", after)
    s_end = end + end_match.end() if end_match else len(text)
    return text[s_start:s_end].strip()


def get_keywords(sentence):
    clean_text = re.sub(r"\033\[[0-9;]*m", "", sentence)
    words = re.findall(r"\b[A-Za-z]{4,}\b", clean_text.lower())
    return set([w for w in words if w not in STOP_WORDS])


def highlight_text(text, keywords, color_code):
    if not keywords:
        return text
    sorted_kws = sorted(list(keywords), key=len, reverse=True)
    pattern = re.compile(
        rf"\b({'|'.join(map(re.escape, sorted_kws))})\b", re.IGNORECASE
    )
    parts = re.split(r"(\033\[[0-9;]*m)", text)
    res = []
    for p in parts:
        if p.startswith("\033"):
            res.append(p)
        else:
            res.append(pattern.sub(lambda m: f"{color_code}{m.group(0)}{Color.END}", p))
    return "".join(res)


def run_annotation():
    os.system("cls" if os.name == "nt" else "clear")
    print(
        f"{Color.BOLD}{Color.PURPLE}=== NEUCLIR24 CITATION ANNOTATION TOOL ==={Color.END}"
    )
    user_name = input(f"{Color.BOLD}Enter your name:{Color.END} ").strip().lower()

    if os.path.exists(OUTPUT_FILE):
        df_history = pd.read_csv(OUTPUT_FILE)
    else:
        df_history = pd.DataFrame(
            columns=["uid", "model", "rater", "label", "llm_label"]
        )

    all_potential_samples = []
    random.seed(SEED)

    for pair in FILE_PAIRS:
        sources_data = load_jsonl(pair["src"])
        sources = {item["source_id"]: item["passages"] for item in sources_data}
        responses = load_jsonl(pair["resp"])

        model_pool = []
        for item in responses:
            sid = item["source_id"]
            passages = sources.get(sid, [])
            for i, label in enumerate(item["labels"]):
                model_pool.append(
                    {
                        "uid": f"{pair['name']}_{sid}_{i}",
                        "model": pair["name"],
                        "sid": sid,
                        "full_text": item["response"],
                        "cite_token": label["text"],
                        "start": label["start"],
                        "end": label["end"],
                        "llm_label": 1 if label["label_type"] == "good" else 0,
                        "passages": passages,
                    }
                )
        random.shuffle(model_pool)
        all_potential_samples.extend(model_pool)

    user_done_uids = set(df_history[df_history["rater"] == user_name]["uid"].tolist())
    counts = {
        p["name"]: len(
            df_history[
                (df_history["rater"] == user_name) & (df_history["model"] == p["name"])
            ]
        )
        for p in FILE_PAIRS
    }

    for cite in all_potential_samples:
        if cite["uid"] in user_done_uids:
            continue
        if counts[cite["model"]] >= SAMPLES_PER_MODEL:
            continue

        # skip empty sources
        try:
            p_num = int(re.sub(r"\D", "", cite["cite_token"]))
            total_passages = len(cite["passages"])
            if p_num < 1 or p_num > total_passages:
                # print(f"{Color.RED}Skipping {cite['uid']}: LLM cited Source {p_num} but only {total_passages} exist.{Color.END}")
                continue
            passage_text = cite["passages"][p_num - 1]["passage"]
        except Exception:
            print(
                f"{Color.RED}Skipping {cite['uid']}: Could not parse citation index.{Color.END}"
            )
            continue

        os.system("cls" if os.name == "nt" else "clear")
        claim_sentence = get_full_sentence(
            cite["full_text"], cite["start"], cite["end"]
        )
        keywords = get_keywords(claim_sentence)

        # UI header
        current_total_done = sum(counts.values())
        print(
            f"{Color.CYAN}Sample {current_total_done + 1}/{TOTAL_TARGET} | Model: {cite['model']}{Color.END}"
        )
        print(
            f"{Color.BOLD}Source ID: {cite['sid']} | Progress for {cite['model']}: {counts[cite['model']]}/{SAMPLES_PER_MODEL}{Color.END}"
        )
        print(f"{Color.DARKCYAN}{'=' * 80}{Color.END}")

        h_claim = highlight_text(claim_sentence, keywords, Color.CYAN)
        h_claim = h_claim.replace(
            cite["cite_token"],
            f"{Color.RED}{Color.BOLD}{Color.UNDERLINE}{cite['cite_token']}{Color.END}{Color.YELLOW}",
        )
        print(f'{Color.BOLD}CLAIM:{Color.END} {Color.YELLOW}"{h_claim}"{Color.END}\n')

        print(f"{Color.BOLD}{Color.GREEN}SOURCE PASSAGE {p_num}:{Color.END}")
        for s_idx, s_text in enumerate(re.split(r"(?<=[.!?])\s+", passage_text)):
            print(
                f"  {Color.BLUE}{s_idx + 1}.{Color.END} {highlight_text(s_text.strip(), keywords, Color.CYAN + Color.BOLD)}"
            )

        print(f"\n{Color.DARKCYAN}{'=' * 80}{Color.END}")

        ans = ""
        while ans not in ["y", "n", "q", "s"]:
            ans = input(f"Attested? (y/n) | 's' skip | 'q' quit: ").lower()

        if ans == "q":
            break

        if ans == "s":
            print(
                f"\n{Color.YELLOW}Skipped instance: {cite['uid']}. Counter not incremented.{Color.END}"
            )
            time.sleep(1)
            continue

        # Save result
        new_row = {
            "uid": cite["uid"],
            "model": cite["model"],
            "rater": user_name,
            "label": 1 if ans == "y" else 0,
            "llm_label": cite["llm_label"],
        }

        # Append to CSV file
        pd.DataFrame([new_row]).to_csv(
            OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False
        )
        counts[cite["model"]] += 1

    print(f"\n{Color.BOLD}Annotation session finished.{Color.END}")


if __name__ == "__main__":
    run_annotation()
