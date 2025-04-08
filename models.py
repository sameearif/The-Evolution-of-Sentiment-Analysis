import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from utils import word_tokenize
from tqdm import tqdm

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
                 lr=1e-3, device='cpu',
                 embedding_weights=None, freeze_embeddings=False):
        super(LogisticRegression, self).__init__()
        self.device = device

        if embedding_weights is not None:
            embedding_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.linear = nn.Linear(embed_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        logits = self.linear(pooled)
        return logits

    def backward(self, input_ids, labels):
        self.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.forward(input_ids)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(self, dataloader, validation_loader=None, epochs=5, save_path="best_model_logr.pt"):
        best_val_acc = 0.0
        for epoch in tqdm(range(epochs)):
            self.train()
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['label']
                loss = self.backward(input_ids, labels)
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

            if validation_loader:
                val_acc = self.evaluate(validation_loader)
                print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.state_dict(), save_path)
                    print(f"New best model saved with val acc: {val_acc:.2f}%")
    
        if validation_loader:
            self.load_state_dict(torch.load(save_path))
            print(f"Loaded best model with val acc: {best_val_acc:.2f}%")


    def evaluate(self, dataloader):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.forward(input_ids)
                _, preds = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        return acc


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden_dims, lr=1e-3, device='cpu',
                 embedding_weights=None, freeze_embeddings=False):
        super(NeuralNetwork, self).__init__()
        self.device = device

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
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        logits = self.network(pooled)
        return logits

    def backward(self, input_ids, labels):
        self.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.forward(input_ids)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(self, dataloader, validation_loader=None, epochs=5,  save_path="best_model_mlp.pt"):
        best_val_acc = 0.0
        for epoch in tqdm(range(epochs)):
            self.train()
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['label']
                loss = self.backward(input_ids, labels)
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

            if validation_loader:
                val_acc = self.evaluate(validation_loader)
                print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.state_dict(), save_path)
                    print(f"New best model saved with val acc: {val_acc:.2f}%")
    
        if validation_loader:
            self.load_state_dict(torch.load(save_path))
            print(f"Loaded best model with val acc: {best_val_acc:.2f}%")


    def evaluate(self, dataloader):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.forward(input_ids)
                _, preds = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        return acc


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden_dim=128, num_layers=1, lr=1e-3, device='cpu',
                 embedding_weights=None, freeze_embeddings=False):
        super(RNN, self).__init__()
        self.device = device

        if embedding_weights is not None:
            embedding_tensor = torch.tensor(embedding_weights, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, hidden = self.rnn(embedded)
        logits = self.fc(hidden[-1])
        return logits

    def backward(self, input_ids, labels):
        self.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.forward(input_ids)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(self, dataloader, validation_loader=None, epochs=5, save_path="best_model_rnn.pt"):
        best_val_acc = 0.0
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['label'].long()
                loss = self.backward(input_ids, labels)
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

            if validation_loader:
                val_acc = self.evaluate(validation_loader)
                print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.state_dict(), save_path)
                    print(f"New best model saved with val acc: {val_acc:.2f}%")
        
        if validation_loader:
            self.load_state_dict(torch.load(save_path))
            print(f"Loaded best model with val acc: {best_val_acc:.2f}%")

    def evaluate(self, dataloader):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.forward(input_ids)
                _, preds = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        return acc


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden_dim=128, num_layers=1, lr=1e-3, device='cpu',
                 embedding_weights=None, freeze_embeddings=False):
        super(LSTM, self).__init__()
        self.device = device

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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(hidden[-1])
        return logits

    def backward(self, input_ids, labels):
        self.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.forward(input_ids)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(self, dataloader, validation_loader=None, epochs=5, save_path="best_model_lstm.pt"):
        best_val_acc = 0.0
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['label'].long()
                loss = self.backward(input_ids, labels)
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

            if validation_loader:
                val_acc = self.evaluate(validation_loader)
                print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.state_dict(), save_path)
                    print(f"New best model saved with val acc: {val_acc:.2f}%")
        
        if validation_loader:
            self.load_state_dict(torch.load(save_path))
            print(f"Loaded best model with val acc: {best_val_acc:.2f}%")

    def evaluate(self, dataloader):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.forward(input_ids)
                _, preds = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        return acc