import numpy as np
import fasttext
import fasttext.util

def create_embedding_matrix_from_fasttext(vocab, language):
    fasttext.util.download_model(language[0:2], if_exists='ignore')
    model_path = f'cc.{language[0:2]}.300.bin'
    ft_model = fasttext.load_model(model_path)
    matrix = np.zeros((len(vocab), 300))
    for word, idx in vocab.items():
        try:
            matrix[idx] = ft_model.get_word_vector(word)
        except KeyError:
            matrix[idx] = np.random.normal(scale=0.6, size=(300,))
    return matrix
