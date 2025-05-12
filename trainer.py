import torch
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython import display

class Trainer:
    def __init__(self, model, optimizer, criterion, device, best_model_path="best_model.pt"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_model_path = best_model_path
        self.best_val_acc = 0.0
        self.logs = []

    def train(self, train_loader, val_loader=None, epochs=20):
        pbar = tqdm(range(1, epochs + 1), desc="Training")
        table_output = display.display(display_id=True)

        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = running_loss / total
            train_acc = 100. * correct / total

            log_entry = {
                "Epoch": epoch,
                "Train Loss": avg_loss,
                "Train Acc (%)": train_acc
            }

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                log_entry["Val Loss"] = val_loss
                log_entry["Val Acc (%)"] = val_acc

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    torch.save(self.model.state_dict(), self.best_model_path)

            self.logs.append(log_entry)

            if table_output is not None:
                df = pd.DataFrame(self.logs)
                style_format = {
                    "Train Loss": "{:.4f}",
                    "Train Acc (%)": "{:.2f}",
                }
                if "Val Loss" in df.columns:
                    style_format["Val Loss"] = "{:.4f}"
                if "Val Acc (%)" in df.columns:
                    style_format["Val Acc (%)"] = "{:.2f}"

                styler = df.style.format(style_format).hide(axis="index")
                table_output.update(styler)
            else:
                log_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in self.logs[-1].items())
                print(log_str)
        
        if val_loader:
            epochs_range = range(1, len(self.logs) + 1)
            val_losses = [log["Val Loss"] for log in self.logs]
            val_accs = [log["Val Acc (%)"] for log in self.logs]

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

            axs[0].plot(epochs_range, val_losses, marker='o')
            axs[0].set_title("Validation Loss per Epoch")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].grid(True)

            axs[1].plot(epochs_range, val_accs, marker='o')
            axs[1].set_title("Validation Accuracy per Epoch")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy (%)")
            axs[1].grid(True)

            plt.tight_layout()
            display.display(plt.gcf())
            plt.close()


    def evaluate(self, data_loader):
        self.model.eval()
        loss_total = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids)

                loss = self.criterion(outputs, labels)
                loss_total += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        avg_loss = loss_total / total
        accuracy = 100. * correct / total

        return avg_loss, accuracy
