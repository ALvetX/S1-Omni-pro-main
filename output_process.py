import json

file_path = "/data/home/zdhs0092/Code/S1-Omni-pro/output_protein_ppi_binding_site_mlp_test_esm2-3b-weight20.jsonl"

all_data = []
threshold=0.9
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        probabilities = item.get("probabilities", [])
        predicted_labels = [i+1 if probabilities[i] >= threshold else 0 for i in range(len(probabilities))]
        item["positive_indices"] = [label for label in predicted_labels if label != 0]
        all_data.append(item)

save_path = "/data/home/zdhs0092/Code/S1-Omni-pro/output_protein_ppi_binding_site_mlp_test_esm2-3b-weight20_with_predictions.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    for item in all_data:
        json.dump(item, f)
        f.write("\n")