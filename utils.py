import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict
import re

from models import *
from embeddings import *

SYSTEM_PROMPT_NO_RESONING = """What is the sentiment of the given text? Positive or Negative?
0: Negative
1: Positive
Answer in the following JSON format ONLY:
{
    "sentiment": 0 or 1
}
"""

SYSTEM_PROMPT_RESONING = """What is the sentiment of the given text? Positive or Negative?
0: Negative
1: Positive
Answer in the following JSON format ONLY:
{
    "reasoning": ...add short reasoning here...
    "sentiment": 0 or 1
}
"""

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = torch.tensor(text_to_ids(self.texts[idx], self.vocab, self.max_len))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": input_ids, "label": label}

def clean_text(text):
    english_punct = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    urdu_punct = r"؟،؛۔"

    all_punct = english_punct + urdu_punct
    punct_pattern = f"[{re.escape(all_punct)}]"
    return re.sub(punct_pattern, "", text)

def word_tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=2):
    word_freq = defaultdict(int)
    for text in texts:
        for word in word_tokenize(text):
            word_freq[word] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def text_to_ids(text, vocab, max_len=256):
    tokens = word_tokenize(text)[:max_len]
    ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(ids))
    return ids

def prepare_dataloader(dataset_name, batch_size=64):
    dataset = load_dataset(dataset_name)

    train_texts = [clean_text(t) for t in dataset["train"]["text"]]
    val_texts = [clean_text(t) for t in dataset["validation"]["text"]]
    test_texts = [clean_text(t) for t in dataset["test"]["text"]]
    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]
    test_labels = dataset["test"]["label"]

    vocab = build_vocab(train_texts)

    train_dataset = IMDBDataset(train_texts, train_labels, vocab)
    val_dataset = IMDBDataset(val_texts, val_labels, vocab)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab

def prepare_text(dataset_name):
    dataset = load_dataset(dataset_name)

    train_texts = [clean_text(t) for t in dataset["train"]["text"]]
    val_texts = [clean_text(t) for t in dataset["validation"]["text"]]
    test_texts = [clean_text(t) for t in dataset["test"]["text"]]
    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]
    test_labels = dataset["test"]["label"]

    vocab = build_vocab(train_texts)

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab


