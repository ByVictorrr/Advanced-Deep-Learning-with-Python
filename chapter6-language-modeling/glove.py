import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)

        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        nn.init.zeros_(self.word_bias.weight)
        nn.init.zeros_(self.context_bias.weight)

    def forward(self, word_idx, context_idx, cooccurrence):
        word_vec = self.word_embeddings(word_idx)
        context_vec = self.context_embeddings(context_idx)
        word_bias = self.word_bias(word_idx).squeeze()
        context_bias = self.context_bias(context_idx).squeeze()

        dot_product = (word_vec * context_vec).sum(dim=1)

        # 🔹 Ensure cooccurrence has no negative values
        cooccurrence = torch.clamp(cooccurrence, min=1e-10)  # Ensures positivity

        # 🔹 Prevent NaN in the weighting function
        max_value = torch.clamp(cooccurrence.max(), min=1e-10)  # Ensure max > 0
        weighting_function = torch.pow(torch.clamp(cooccurrence / max_value, max=1.0), 0.75)

        # 🔹 Safe log computation
        log_cooccurrence = torch.log(cooccurrence)  # No NaN since cooccurrence >= 1e-10

        # Compute final loss
        loss = weighting_function * (dot_product + word_bias + context_bias - log_cooccurrence).pow(2)

        return loss.mean()


def build_cooccurrence_matrix(tokenized_corpus, window_size=2):
    vocab_size = len(vocab)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for sentence in tokenized_corpus:
        for i, word in enumerate(sentence):
            word_id = word_to_id[word]
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(sentence))

            for j in range(start, end):
                if i != j:
                    neighbor_id = word_to_id[sentence[j]]
                    cooccurrence_matrix[word_id, neighbor_id] += 1.0 / abs(j - i)

    return cooccurrence_matrix


if __name__ == "__main__":
    corpus = [
        "I love deep learning",
        "deep learning is amazing",
        "machine learning is fun",
        "natural language processing is exciting"
    ]

    # Tokenizing the corpus
    tokenized_corpus = [sentence.lower().split() for sentence in corpus]

    # Building the vocabulary
    from collections import Counter

    word_counts = Counter(word for sentence in tokenized_corpus for word in sentence)
    vocab = list(word_counts.keys())
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    cooccurrence_matrix = build_cooccurrence_matrix(tokenized_corpus, window_size=2)
    X = cooccurrence_matrix
    X[X == 0] = 1e-10  # Avoid log(0)
    X_log = np.log(X)
    vocab_size = len(vocab)
    embedding_dim = 50
    epochs = 500
    learning_rate = 0.01

    word_indices = []
    context_indices = []
    cooccurrences = []

    for i in range(vocab_size):
        for j in range(vocab_size):
            if cooccurrence_matrix[i, j] > 0:
                word_indices.append(i)
                context_indices.append(j)
                cooccurrences.append(X_log[i, j] + 1e-10)

    word_indices = torch.tensor(word_indices, dtype=torch.long)
    context_indices = torch.tensor(context_indices, dtype=torch.long)
    cooccurrences = torch.tensor(cooccurrences, dtype=torch.float32)

    model = GloVe(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(word_indices, context_indices, cooccurrences)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    word_vectors = model.word_embeddings.weight.data.numpy()


    def get_word_vector(word):
        if word not in word_to_id:
            return None
        idx = word_to_id[word]
        return word_vectors[idx]


    def find_similar_words(word, top_n=5):
        if word not in word_to_id:
            return "Word not found in vocabulary."

        word_vec = get_word_vector(word).reshape(1, -1)
        similarities = {}

        for other_word in vocab:
            if other_word != word:
                other_vec = get_word_vector(other_word).reshape(1, -1)
                similarity = cosine_similarity(word_vec, other_vec)[0][0]
                similarities[other_word] = similarity

        return sorted(similarities, key=similarities.get, reverse=True)[:top_n]


    # Example usage
    print("Similar words to 'learning':", find_similar_words("learning"))


    def plot_embeddings(words):
        vectors = np.array([get_word_vector(word) for word in words if get_word_vector(word) is not None])

        if vectors.shape[0] == 0:
            print("No valid word vectors found.")
            return

        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        plt.figure(figsize=(8, 6))
        for word, coord in zip(words, reduced_vectors):
            plt.scatter(coord[0], coord[1])
            plt.text(coord[0] + 0.02, coord[1] + 0.02, word, fontsize=12)

        plt.title("Word Embeddings Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid()
        plt.show()
    plot_embeddings(vocab)

