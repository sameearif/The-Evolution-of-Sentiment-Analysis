import torch
import torch.nn as nn
from openai import OpenAI
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from utils import word_tokenize

class NaiveBayesTextClassifier:
    def __init__(self, vocab):
        self.vocab = vocab
        self.model = MultinomialNB()

    def text_to_bow(self, text):
        tokens = word_tokenize(text)
        bow = np.zeros(len(self.vocab))
        for token in tokens:
            idx = self.vocab.get(token, self.vocab["<UNK>"])
            bow[idx] += 1
        return bow

    def prepare_features(self, texts):
        return np.array([self.text_to_bow(text) for text in texts])
    
    def train(self, train_texts, train_labels):
        X_train = self.prepare_features(train_texts)
        y_train = np.array(train_labels)
        self.model.fit(X_train, y_train)

    def evaluate(self, texts, labels):
        X = self.prepare_features(texts)
        y = np.array(labels)
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds) * 100
        return acc

class LogisticRegression(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes,
                 embedding_weights=None, freeze_embeddings=False):
        super(LogisticRegression, self).__init__()

        if embedding_weights is not None:
            embedding_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        logits = self.linear(pooled)
        return logits

class MLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden_dims, dropout_rate=0.1,
                 embedding_weights=None, freeze_embeddings=False):
        super(MLP, self).__init__()

        if embedding_weights is not None:
            embedding_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        input_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        logits = self.network(pooled)
        return logits

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden_dim=128, num_layers=1,
                 embedding_weights=None, freeze_embeddings=False):
        super(RNN, self).__init__()

        if embedding_weights is not None:
            embedding_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, hidden = self.rnn(embedded)
        logits = self.fc(torch.cat(( output[:, -1, :], hidden[-1]), dim=1))
        return logits

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden_dim=128, num_layers=1,
                 embedding_weights=None, freeze_embeddings=False):
        super(LSTM, self).__init__()

        if embedding_weights is not None:
            embedding_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, _) = self.lstm(embedded)
        logits = self.fc(torch.cat((output[:, -1, :], hidden[-1]), dim=1))
        return logits

class API:
    def __init__(self, model_name, system_prompt):
        self.client = OpenAI(
            api_key="",
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
    def generate(self, text):
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.6,
            top_p=0.9,
            messages=self.messages,
        )
        output = chat_completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": output})
        return output