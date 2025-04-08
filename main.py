import argparse
from models import *
from utils import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to use")
parser.add_argument("--model_config", type=str, required=True, help="Path to model config JSON")
parser.add_argument("--device", type=str, required=True, help="Device to use: cuda/mps/cpu")
parser.add_argument("--language", type=str, required=True, help="Language code for fastText (en, ur)")

args = parser.parse_args()

def main():
    dataset = load_imdb(args.dataset_name)
    train_texts, train_labels = dataset["train"]['text'], dataset["train"]['label']
    val_texts, val_labels = dataset["validation"]['text'], dataset["validation"]['label']
    test_texts, test_labels = dataset["test"]['text'], dataset["test"]['label']

    train_texts = [clean_text(t) for t in train_texts]
    val_texts = [clean_text(t) for t in val_texts]
    test_texts = [clean_text(t) for t in test_texts]

    vocab = build_vocab(train_texts)
    config = read_config(args.model_config, vocab, args.language)

    if config.get("type") == "NaiveBayes":
        model = NaiveBayesTextClassifier(vocab)
        model.train(train_texts, train_labels)
        val_acc = model.evaluate(val_texts, val_labels)
        test_acc = model.evaluate(test_texts, test_labels)
        print(f"Test Accuracy: {test_acc:.2f}%")
    else:
        batch_size = config["training_args"].get("batch_size", 64)

        train_dataset = IMDBDataset(train_texts, train_labels, vocab)
        val_dataset = IMDBDataset(val_texts, val_labels, vocab)
        test_dataset = IMDBDataset(test_texts, test_labels, vocab)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        device = args.device
        network_args = config["network_args"]

        if config.get("type") == "LogisticRegression":
            model = LogisticRegression(
                vocab_size=len(vocab),
                embed_dim=network_args["embed_dim"],
                num_classes=network_args["num_classes"],
                lr=config["training_args"].get("learning_rate", 1e-3),
                device=device,
                embedding_weights=config["embedding_weights"],
                freeze_embeddings=config["freeze_embeddings"]
            )
        elif config.get("type") == "NeuralNetwork":
            model = NeuralNetwork(
                vocab_size=len(vocab),
                embed_dim=network_args["embed_dim"],
                num_classes=network_args["num_classes"],
                hidden_dims=network_args["hidden_dims"],
                lr=config["training_args"].get("learning_rate", 1e-3),
                device=device,
                embedding_weights=config["embedding_weights"],
                freeze_embeddings=config["freeze_embeddings"]
            )
        elif config.get("type") == "RNN":
            model = RNN(
                vocab_size=len(vocab),
                embed_dim=network_args["embed_dim"],
                num_classes=network_args["num_classes"],
                hidden_dim=network_args.get("hidden_dim", 128),
                num_layers=network_args.get("num_layers", 1),
                lr=config["training_args"].get("learning_rate", 1e-3),
                device=device,
                embedding_weights=config["embedding_weights"],
                freeze_embeddings=config["freeze_embeddings"]
            )
        elif config.get("type") == "LSTM":
            model = LSTM(
                vocab_size=len(vocab),
                embed_dim=network_args["embed_dim"],
                num_classes=network_args["num_classes"],
                hidden_dim=network_args.get("hidden_dim", 128),
                num_layers=network_args.get("num_layers", 1),
                lr=config["training_args"].get("learning_rate", 1e-3),
                device=device,
                embedding_weights=config["embedding_weights"],
                freeze_embeddings=config["freeze_embeddings"]
            )


        model.train_model(train_loader, epochs=config["training_args"]["epochs"], validation_loader=val_loader)
        test_acc = model.evaluate(test_loader)
        print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
