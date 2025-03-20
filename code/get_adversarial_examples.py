import json
from datasets import load_dataset
import random
import nltk
from transformers import pipeline

def extract_original_texts(input_file, output_file):
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]

    article_lookup = {item["id"]: item["article"] for item in ds}

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            id_value = data["id"].split("-")[-1]

            article = article_lookup.get(id_value, "")
            data["doc"] = article
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"{output_file} saved.")


def add_lead3(input_file, output_file):
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]

    article_lookup = {item["id"]: item["article"] for item in ds}

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            id_value = data["id"].split("-")[-1]
            article = article_lookup.get(id_value, "")
            lead3 = " ".join(article.split(". ")[:3]) + "." if article else ""
            # add lead3
            data["decoded_add_lead3"] = lead3 + " " + data["decoded"]

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f" {output_file} saved.")

def add_rand3(input_file, output_file):
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]

    article_lookup = {item["id"]: item["article"] for item in ds}

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            id_value = data["id"].split("-")[-1]

            article = article_lookup.get(id_value, "")
            sentences = article.split(". ")

            if len(sentences) < 3:
                rand3_sentences = sentences
            else:
                rand3_sentences = random.sample(sentences, min(3, len(sentences)))  # 無視位置，全篇隨機選 3 句

            summary_sentences = data["decoded"].split(". ")
            num_summary_sentences = len(summary_sentences)

            if num_summary_sentences > 1:

                insert_positions = sorted(random.sample(range(num_summary_sentences + 1), len(rand3_sentences)))

                for insert_pos, rand_sentence in zip(insert_positions, rand3_sentences):
                    summary_sentences.insert(insert_pos, rand_sentence)
            else:
                if random.random() > 0.5:
                    summary_sentences = rand3_sentences + summary_sentences
                else:
                    summary_sentences += rand3_sentences

            data["decoded_add_rand3"] = ". ".join(summary_sentences) + "."
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"{output_file} saved.")


def sent_reorder(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    for entry in data:
        original_text = entry["decoded"]
        sentences = nltk.sent_tokenize(original_text)
        random.shuffle(sentences)
        entry["reordered"] = " ".join(sentences)

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f" {output_file} saved.")


def generate_overlong_summaries(input_file, output_file, expansion_factors=[2, 3]):
    paraphraser = pipeline("text2text-generation", model="t5-small")

    def paraphrase_text(text, num_variants=2):
        max_len = min(150, len(text.split()) * 2)
        return [paraphraser(text, max_length=max_len, do_sample=True, temperature=0.8)[0]['generated_text'] for _ in range(num_variants)]
    data = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    for entry in data:
        original_text = entry["decoded"]
        summaries = {}

        for factor in expansion_factors:
            variants = paraphrase_text(original_text, num_variants=factor - 1)
            expanded_summary = " ".join([original_text] + variants)
            summaries[f"decoded_{factor}x_expanded"] = expanded_summary

        entry.update(summaries)

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f" {output_file} saved with expanded summaries!")

def merge_jsonl(original, lead3, rand3, substitution, overlong, reorder, output_path):
    num_lines = 20
    def read_jsonl(file_path, max_lines=num_lines):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                data.append(json.loads(line))
        return data
    original_data = read_jsonl(original)
    lead3_data = read_jsonl(lead3)
    rand3_data = read_jsonl(rand3)
    substitution_data = read_jsonl(substitution)
    overlong_data = read_jsonl(overlong)
    reorder_data = read_jsonl(reorder)

    merged_data = []
    for i in range(num_lines):
        merged_entry = {
            "id": lead3_data[i].get("id"),
            "doc": original_data[i].get("doc"),
            "decoded": lead3_data[i].get("decoded"),
            "expert_annotations": lead3_data[i].get("expert_annotations"),
            "turker_annotations": lead3_data[i].get("turker_annotations"),
            "references": lead3_data[i].get("references"),
            "model_id": lead3_data[i].get("model_id"),
            "filepath": lead3_data[i].get("filepath"),
            "reordered": reorder_data[i].get("reordered"),
            "synonym_substituted_2": substitution_data[i].get("synonym_substituted_2"),
            "antonym_substituted_2": substitution_data[i].get("antonym_substituted_2"),
            "decoded_add_lead3": lead3_data[i].get("decoded_add_lead3"),
            "decoded_add_rand3": rand3_data[i].get("decoded_add_rand3"),
            "decoded_2x_expanded": overlong_data[i].get("decoded_2x_expanded"),
            "decoded_3x_expanded": overlong_data[i].get("decoded_3x_expanded"),
        }
        merged_data.append(merged_entry)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{output_path} saved.")


if __name__=="__main__":
    input_filename = "data/m22_model_annotations_20.aligned.jsonl"
    output_lead3_filename = "data/m22_lead3.jsonl"
    output_reorder = "data/m22_reorder.jsonl"
    output_rand3_filename = "data/m22_rand3.jsonl"
    output_overlong_filename = "data/m22_overlong.jsonl"
    substitution_filename = "data/m22_substitution_20.jsonl"
    merged_file = "util_data/merged_adversarial_m22.jsonl"
    original_doc = "data/original_doc.jsonl"
    extract_original_texts(input_filename, original_doc)
    # add_rand3(input_filename, output_rand3_filename)
    # add_lead3(input_filename, output_lead3_filename)
    # sent_reorder(input_filename, output_reorder)
    # generate_overlong_summaries(input_filename, output_overlong_filename)
    merge_jsonl(original_doc, output_lead3_filename, output_rand3_filename, substitution_filename, output_overlong_filename, output_reorder, merged_file)







