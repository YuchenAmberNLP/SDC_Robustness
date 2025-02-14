import pandas as pd
import json
import scipy.stats as stats

def compute_correlations(score_column, df_merged):
    correlation_results = {}
    metrics = ["coherence", "consistency", "fluency", "relevance"]

    for metric in metrics:
        pearson_corr, _ = stats.pearsonr(df_merged[score_column], df_merged[metric])
        spearman_corr, _ = stats.spearmanr(df_merged[score_column], df_merged[metric])
        kendall_corr, _ = stats.kendalltau(df_merged[score_column], df_merged[metric])

        correlation_results[metric] = {
            "Pearson's γ": pearson_corr,
            "Spearman's ρ": spearman_corr,
            "Kendall's τ": kendall_corr
        }

    return correlation_results


def shannon_correlate(human_annotation_file, shannon_scores_file):
    df_human = pd.read_csv(human_annotation_file, delimiter=",")

    shannon_data = []

    with open(shannon_scores_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            shannon_data.append(data)

    df_shannon = pd.DataFrame(shannon_data)

    df_merged = df_human.merge(df_shannon, left_on=["id", "summary_type"], right_on=["doc_id", "summary_type"], how="inner")


    df_merged = df_merged[["id", "summary_type", "shannon_score", "shannon_star", "coherence", "consistency", "fluency", "relevance"]]
    correlation_shannon = compute_correlations("shannon_score", df_merged)
    correlation_shannon_star = compute_correlations("shannon_star", df_merged)
    jsonl_shannon_path = "../util_data/scores/correlation_shannon_results.jsonl"
    jsonl_shannon_star_path = "../util_data/scores/correlation_shannon_star_results.jsonl"

    with open(jsonl_shannon_path, "w", encoding="utf-8") as jsonl_sdc_file:
        jsonl_sdc_file.write(json.dumps({"shannon_score": correlation_shannon}, indent=4) + "\n")

    with open(jsonl_shannon_star_path, "w", encoding="utf-8") as jsonl_sdc_star_file:
        jsonl_sdc_star_file.write(json.dumps({"shannon_star": correlation_shannon_star}, indent=4) + "\n")

    print(f"JSONL files saved: {jsonl_shannon_path}, {jsonl_shannon_star_path}")



def sdc_correlate(human_annotation_file, sdc_scores_file):
    df_human = pd.read_csv(human_annotation_file, delimiter=",")  # 以 tab 為分隔符
    sdc_data = []

    with open(sdc_scores_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            sdc_data.append(data)

    df_sdc = pd.DataFrame(sdc_data)


    df_merged = df_human.merge(df_sdc, left_on=["id", "summary_type"], right_on=["doc_id", "summary_type"], how="inner")

    df_merged = df_merged[["id", "summary_type", "sdc", "sdc_star", "coherence", "consistency", "fluency", "relevance"]]

    correlation_sdc = compute_correlations("sdc", df_merged)
    correlation_sdc_star = compute_correlations("sdc_star", df_merged)

    df_correlation_sdc = pd.DataFrame.from_dict(correlation_sdc, orient="index")
    df_correlation_sdc_star = pd.DataFrame.from_dict(correlation_sdc_star, orient="index")

    # df_correlation_sdc.to_csv("../util_data/scores/correlation_sdc_results.csv", index=True)
    # df_correlation_sdc_star.to_csv("../util_data/scores/correlation_sdc_star_results.csv", index=True)
    # print("correlation_sdc_results.csv and correlation_sdc_star_results.csv saved")
    jsonl_sdc_path = "../util_data/scores/correlation_sdc_results.jsonl"
    jsonl_sdc_star_path = "../util_data/scores/correlation_sdc_star_results.jsonl"

    with open(jsonl_sdc_path, "w", encoding="utf-8") as jsonl_sdc_file:
        jsonl_sdc_file.write(json.dumps({"sdc": correlation_sdc}, indent=4) + "\n")

    with open(jsonl_sdc_star_path, "w", encoding="utf-8") as jsonl_sdc_star_file:
        jsonl_sdc_star_file.write(json.dumps({"sdc_star": correlation_sdc_star}, indent=4) + "\n")

    print(f"JSONL files saved: {jsonl_sdc_path}, {jsonl_sdc_star_path}")

def rouge_correlate(human_annotation_file, rouge_scores_file):
    df_human = pd.read_csv(human_annotation_file, delimiter=",")  # 以 tab 為分隔符
    rouge_data = []

    with open(rouge_scores_file, "r", encoding="utf-8") as file:
        for line in file:
            rouge_data.append(json.loads(line.strip()))

    # 轉換為 DataFrame
    rouge_list = []
    for entry in rouge_data:
        doc_id = entry["doc_id"]
        for summary_type, scores in entry["rouge_scores"].items():
            rouge_list.append({
                "doc_id": doc_id,
                "summary_type": summary_type,
                "rouge1": scores["rouge1"],
                "rouge2": scores["rouge2"],
                "rougeL": scores["rougeL"]
            })
    df_rouge = pd.DataFrame(rouge_list)
    df_merged = df_human.merge(df_rouge, left_on=["id", "summary_type"], right_on=["doc_id", "summary_type"], how="inner")

    df_merged = df_merged[["id", "summary_type", "rouge1", "rouge2", "rougeL", "coherence", "consistency", "fluency", "relevance"]]


    correlation_rouge1 = compute_correlations("rouge1", df_merged)
    correlation_rouge2 = compute_correlations("rouge2", df_merged)
    correlation_rougeL = compute_correlations("rougeL", df_merged)

    correlation_results = {
        "rouge1": correlation_rouge1,
        "rouge2": correlation_rouge2,
        "rougeL": correlation_rougeL
    }

    output_file = "../util_data/scores/correlation_rouge_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(correlation_results, indent=4) + "\n")

    print(f"{output_file} saved")

if __name__ == "__main__":
    human_annotation_file = "../util_data/adversarial_m22_annotation.csv"
    shannon_scores_file = "../restults/merged_shannon_scores.jsonl"
    sdc_scores_file = "../results/merged_sdc_scores.jsonl"
    rouge_scores_file = "../results/rouge_score.jsonl"
    shannon_correlate(human_annotation_file, shannon_scores_file)
    sdc_correlate(human_annotation_file, sdc_scores_file)
    rouge_correlate(human_annotation_file, rouge_scores_file)


