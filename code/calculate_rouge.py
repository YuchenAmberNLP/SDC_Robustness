import json
import os
from rouge_score import rouge_scorer
from tqdm import tqdm

SUMMARY_KEYS = [
    "decoded", "reordered", "synonym_substituted_2", "antonym_substituted_2",
    "decoded_add_lead3", "decoded_add_rand3", "decoded_2x_expanded", "decoded_3x_expanded"
]
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def compute_rouge(system_summary, reference_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = []
    for ref in reference_summaries:
        score = scorer.score(ref, system_summary)
        scores.append(score)

    avg_scores = {
        'rouge1': sum(s['rouge1'].fmeasure for s in scores) / len(scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in scores) / len(scores),
        'rougeL': sum(s['rougeL'].fmeasure for s in scores) / len(scores),
    }
    return avg_scores

def process_data(input_file, output_file):
    data = load_jsonl(input_file)
    results = []

    for item in tqdm(data, desc="Processing ROUGE"):
        doc_id = item["id"]
        references = item.get("references", [])

        rouge_scores = {}
        for key in SUMMARY_KEYS:
            if key in item:
                system_summary = item[key]
                rouge_scores[key] = compute_rouge(system_summary, references)


        results.append({
            "doc_id": doc_id,
            "rouge_scores": rouge_scores
        })


    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument("--output_file", type=str, required=True)
    # args = parser.parse_args()
    input_file = "../util_data/merged_adversarial_m22.jsonl"
    output_file = "../results/rouge_score.jsonl"
    process_data(input_file, output_file)
