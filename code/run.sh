# to calculate rouge scores:
python calculate_rouge.py

# to calculate sdc and sdc*:
python SDC.py --input_file ../util_data/merged_adversarial_m22.jsonl --summ_keys decoded reordered synonym_substituted_2 antonym_substituted_2 decoded_add_lead3 decoded_add_rand3 decoded_2x_expande decoded_3x_expanded

# to calculate shannon and shannon*:
python shannon.py
python summeval_score.py

# to merge sdc scores and shannon scores:
python merge_shannon_sdc_score.py

# to calculate correlation:
python correlation_calculate.py