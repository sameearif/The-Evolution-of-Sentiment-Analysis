import numpy as np
import fasttext
import fasttext.util

def load_fasttext_model(language='en'):
    fasttext.util.download_model(language, if_exists='ignore')
    model_path = f'cc.{language}.300.bin'
    return fasttext.load_model(model_path)

def create_embedding_matrix_from_fasttext(vocab, ft_model, embedding_dim=300):
    matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        try:
            matrix[idx] = ft_model.get_word_vector(word)
        except KeyError:
            matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return matrix
