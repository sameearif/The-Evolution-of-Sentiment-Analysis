import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict
import re
import json
from embeddings import load_fasttext_model, create_embedding_matrix_from_fasttext

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
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

def load_imdb(dataset_name):
    return load_dataset(dataset_name)

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

def text_to_ids(text, vocab, max_len=128):
    tokens = word_tokenize(text)[:max_len]
    ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(ids))
    return ids



def read_config(config_path, vocab, language):
    with open(config_path, 'r') as f:
        config = json.load(f)

    type_ = config.get("type")
    embedding_type = config.get("embeddings", None)
    freeze_embeddings = config.get("freeze_embeddings", False)
    network_args = config.get("network_args", {})
    training_args = config.get("training_args", {})

    embedding_weights = None

    if embedding_type and embedding_type.lower() == "fasttext":
        embedding_dim = network_args.get("embed_dim", 300)
        ft_model = load_fasttext_model(language)
        embedding_weights = create_embedding_matrix_from_fasttext(vocab, ft_model, embedding_dim)

    return {
        "type": type_,
        "embedding_weights": embedding_weights,
        "freeze_embeddings": freeze_embeddings,
        "network_args": network_args,
        "training_args": training_args
    }
