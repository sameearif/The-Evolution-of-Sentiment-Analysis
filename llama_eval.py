import json
import glob
import os
import re

def load_predictions(json_path):
    """
    Load the list of examples from one of your output files.
    Each example is a dict with keys: "text", "label", "pred" (the raw model output).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_sentiment(pred_str):
    """
    Robustly pull out the integer after "sentiment": even if inner quotes
    aren't escaped properly. Returns 0 or 1.
    """
    # look for "sentiment": <digit>
    m = re.search(r'"sentiment"\s*:\s*([01])', pred_str)
    if not m:
        return -1
    return int(m.group(1))

def compute_accuracy(examples):
    correct = 0
    for ex in examples:
        true_label = ex['label']
        pred_label = extract_sentiment(ex['pred'])
        if pred_label == true_label:
            correct += 1
        else:
            print(40*"=")
            print(ex["text"])
            print(40*"=")
    return correct / len(examples)

if __name__ == '__main__':
    pattern = os.path.join('english', 'no-reasoning', '*.json')
    all_files = glob.glob(pattern)
    if not all_files:
        print(f"No files found matching {pattern}")
        exit(1)

    total_correct = 0
    total_count = 0

    for file_path in sorted(all_files):
        examples = load_predictions(file_path)
        acc = compute_accuracy(examples)
        n = len(examples)
        # print(f"{os.path.basename(file_path)}: {acc*100:.2f}% ({n} samples)")
        total_correct += acc * n
        total_count += n

    # overall_acc = total_correct / total_count
    # print(f"\nOverall accuracy across {total_count} samples: {overall_acc*100:.2f}%")
