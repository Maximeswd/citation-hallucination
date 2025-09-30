# ==============================================================================
# --- IMPORTS & SETUP ---
# ==============================================================================

import argparse
import json
import re
import os


def config():
    parser = argparse.ArgumentParser(
        description="Extract responses and source info from GPT-Researcher outputs."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input .jsonl file (data)."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to input .jsonl.log file (logs).",
    )
    parser.add_argument(
        "--response_output_path",
        type=str,
        required=True,
        help="Path to write responses .jsonl.",
    )
    parser.add_argument(
        "--source_output_path",
        type=str,
        required=True,
        help="Path to write source info .jsonl.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama-70b",
        help="Model name to use in output.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature to use in output."
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split to use in output."
    )
    parser.add_argument(
        "--quality", type=str, default="good", help="Quality to use in output."
    )
    parser.add_argument(
        "--citation",
        action="store_true",
        help="Flag for if we want to use the citation or not.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Flag for if we want to use the baseline or not.",
    )
    args = parser.parse_args()
    return args


def check_path(path_name):
    if not os.path.exists(os.path.dirname(path_name)):
        os.makedirs(os.path.dirname(path_name))
        print(f"Path did not exist yet, so it has been made {path_name}")
    else:
        pass


def replace_docids_with_source(full_response: str, doc_dict: dict) -> str:
    # invert mapping so uuid → key (source number as string)
    id_to_key = {uuid: key for key, uuid in doc_dict.items()}

    # match [uuid] or [uuid, uuid, ...]
    pattern = re.compile(r"\[([0-9A-Fa-f\-]{36}(?:\s*,\s*[0-9A-Fa-f\-]{36})*)\]")

    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        # split on commas (allow optional spaces)
        uuids = re.split(r"\s*,\s*", inner)
        sources = []
        for u in uuids:
            key = id_to_key.get(u)
            if key is not None:
                sources.append(f"Source: {key}")
            else:
                # leave unknown IDs untouched
                sources.append(u)
        return "[" + ", ".join(sources) + "]"

    return pattern.sub(_repl, full_response)


def replace_single_docid_with_source(full_response: str, doc_dict: dict) -> str:
    # first invert mapping so we can look up the key by doc_id
    id_to_key = {v: k for k, v in doc_dict.items()}

    # match any 36-char hex UUID in brackets
    pattern = re.compile(r"\[([0-9a-fA-F\-]{36})\]")

    def _repl(m):
        doc_id = m.group(1)
        key = id_to_key.get(doc_id)
        if key is not None:
            return f"[Source: {key}]"
        else:
            # if we don’t know this UUID, leave it unchanged
            return m.group(0)

    return pattern.sub(_repl, full_response)


def citation(
    data_path,
    log_path,
    response_output_path,
    source_output_path,
    model_name="llama3.2-3b",
    temperature=0.6,
    split="test",
    quality="good",
):
    id_counter = 0

    # make sure directory exists
    check_path(response_output_path), check_path(source_output_path)

    with (
        open(data_path, "r", encoding="utf-8") as data_f,
        open(log_path, "r", encoding="utf-8") as log_f,
        open(response_output_path, "w", encoding="utf-8") as r_out_f,
        open(source_output_path, "w", encoding="utf-8") as s_out_f,
    ):
        for data_line, log_line in zip(data_f, log_f):
            data = json.loads(data_line)
            log = json.loads(log_line)

            source_id = log["request_id"]
            full_response = log["report"]
            doc_dict = log["doc_dict"]
            full_new = replace_docids_with_source(full_response, doc_dict)
            labels = []

            for resp in data["responses"]:
                if not resp.get("citations"):
                    continue

                key, value = resp.get("citations").popitem()
                new_key = replace_single_docid_with_source(f"[{key}]", doc_dict)

                # Determine label type first
                label_type = ""
                if value == 1.0 or value == 0.75:
                    label_type = "good"
                elif value == 0.0 or value == 0.25:
                    label_type = "bad"
                else:
                    continue  

                # Construct the snippet to find its location in full response
                snippet = f"{resp['text'][:-1]} {new_key}."
                find_in_full = full_new.find(snippet)

                if find_in_full != -1:
                    source_in_snippet = snippet.find(new_key)
                    if source_in_snippet != -1:

                        # Find the absolute start position of the citation string (e.g., "[Source: 17]")
                        citation_start_pos = find_in_full + source_in_snippet

                        # Use regex to find the number within the citation string
                        match = re.search(r"(\d+)", new_key)

                        if match:
                            # The number itself (e.g., "17")
                            number_str = match.group(1)

                            # The start position of the number relative to the citation string's beginning
                            number_start_in_citation = match.start(1)

                            # Calculate the final, absolute start and end positions
                            label_start = citation_start_pos + number_start_in_citation
                            label_end = label_start + len(number_str)

                            labels.append(
                                {
                                    "start": label_start,
                                    "end": label_end,
                                    "text": number_str,
                                    "meta": "none",
                                    "label_type": label_type,  
                                    "implicit_true": "maybe",
                                    "due_to_null": True,
                                }
                            )

            # Assemble the final data
            response = {
                "id": id_counter,
                "source_id": source_id,
                "model": model_name,
                "temperature": temperature,
                "labels": labels,
                "split": split,
                "quality": quality,
                "response": full_new,
            }

            source = {
                "source_id": source_id,
                "task_type": "none",
                "source": log["collection_ids"],
                "source_info": "none",
                "background": log["background"],
                "limit": log["limit"],
                "title": log["title"],
                "passages": log["passages"],
                "logs": log["logs"],
                "report": log["report"],
                "report_new": full_new,
                "context": log["context"],
                "prompt": log["prompt"],
            }

            id_counter += 1

            r_out_f.write(json.dumps(response, ensure_ascii=False) + "\n")
            s_out_f.write(json.dumps(source, ensure_ascii=False) + "\n")
        print(f"response data stored here: {response_output_path}")
        print(f"source info data stored here: {source_output_path}")


def main():
    args = config()
    if args.citation:
        citation(
            data_path=args.data_path,
            log_path=args.log_path,
            response_output_path=args.response_output_path,
            source_output_path=args.source_output_path,
            model_name=args.model_name,
            temperature=args.temperature,
            split=args.split,
            quality=args.quality,
        )


if __name__ == "__main__":
    main()
