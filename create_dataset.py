import argparse
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from utils import load_imdb
from translator import SeamlessM4T

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to translate")
parser.add_argument("--model_name", type=str, required=True, help="Model for translation")
parser.add_argument("--device", type=str, required=True, help="Cuda/MPS/CPU")

args = parser.parse_args()

def translate(model, dataset):
    new_dataset = {"train": {"text": [], "label": []}, "validation": {"text": [], "label": []}, "test": {"text": [], "label": []}}
    for split in ["train", "validation", "test"]:
        for item in tqdm(dataset[split]):
            new_dataset[split]["text"].append(model.forward(item["text"]))
            new_dataset[split]["label"].append(item["label"])

    new_dataset["train"] = Dataset.from_dict({"text": new_dataset["train"]["text"], "label": new_dataset["train"]["label"]})
    new_dataset["validation"] = Dataset.from_dict({"text": new_dataset["validation"]["text"], "label": new_dataset["validation"]["label"]})
    new_dataset["test"] = Dataset.from_dict({"text": new_dataset["test"]["text"], "label": new_dataset["test"]["label"]})
    new_dataset = DatasetDict(new_dataset)
    new_dataset.push_to_hub("sameearif/imbd-urdu")

def main():
    model = SeamlessM4T(args.model_name, args.device + f":{args.cuda}")
    dataset = load_imdb(args.dataset_name)
    translate(model, dataset)
    
if __name__ == "__main__":
    main()