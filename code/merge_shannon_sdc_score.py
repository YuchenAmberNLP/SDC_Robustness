import json
import glob

shannon_output_file = "../util_data/scores/merged_shannon_scores.jsonl"
sdc_output_file = "../util_data/scores/merged_sdc_scores.jsonl"

shannon_jsonl_files = glob.glob("../util_data/shannon/score_shannon_output_*.jsonl")
sdc_jsonl_files = glob.glob("../util_data/sdc_exp/score_sdc_output_*.jsonl")

with open(shannon_output_file, "w", encoding="utf-8") as outfile:
    for file in shannon_jsonl_files:
        summary_type = file.replace("../util_data/shannon/score_shannon_output_", "").replace(".jsonl", "")

        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    filtered_data = {
                        "summary_type": summary_type,
                        "doc_id": data.get("doc_id"),
                        "shannon_score": data.get("shannon_score"),
                        "shannon_star": data.get("shannon_star")

                    }
                    outfile.write(json.dumps(filtered_data) + "\n")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file}: {line.strip()}")

with open(sdc_output_file, "w", encoding="utf-8") as outfile:
    for file in sdc_jsonl_files:

        summary_type = file.replace("../util_data/sdc/score_sdc_output_", "").replace(".jsonl", "")

        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())

                    filtered_data = {
                        "summary_type": summary_type,
                        "doc_id": data.get("doc_id"),
                        "sdc": data.get("sdc"),
                        "sdc_star": data.get("sdc_star")
                    }

                    outfile.write(json.dumps(filtered_data) + "\n")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file}: {line.strip()}")