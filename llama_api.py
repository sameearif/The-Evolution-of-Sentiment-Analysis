import argparse
from tqdm import tqdm
import json
from datasets import load_dataset

from models import API
from utils import SYSTEM_PROMPT_RESONING

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)


args = parser.parse_args()

dataset = load_dataset("sameearif/imdb-english")["test"]
range = range(args.n * 500, (args.n * 500) + 500)
api = API("meta-llama/Meta-Llama-3.1-8B-Instruct", SYSTEM_PROMPT_RESONING)

outputs = []
for i in tqdm(range):
    out = api.generate(dataset[i]["text"])
    outputs.append({"text": dataset[i]["text"], "label": dataset[i]["label"], "pred": out})
    with open(f"english/reasoning/{args.n + 1}.json", "w") as f:
        json.dump(outputs, f, indent=2)
