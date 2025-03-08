import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import numpy as np
from nltk.corpus import brown

# Download the Brown corpus
nltk.download('brown')


def one_hot_encode(index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec


class ContinuousBagOfWords(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        """Forward pass of CBOW:

        1. convert context words to embeddings
        2. compute the average of the embeddings
        3. pass through linear layer to get vocabulary probabilities
        """
        embedded = self.embeddings(context_words)  # shape: (batch_size, context_size, embedding_size)
        avg_embedded = embedded.mean(dim=1)
        output = self.linear(avg_embedded)  # Linear Layer projection to vocab space
        return output  # return logits (before applying softmax)

    def predict(self, context_words, word_to_index, index_to_word):
        context_indices = torch.tensor(
            [word_to_index.get(word.lower(), word_to_index["<UNK>"]) for word in context_words],
            dtype=torch.long
        ).unsqueeze(0)  # Ensure correct batch shape
        output_probs = self.forward(context_indices)
        predicted_index = torch.argmax(output_probs, dim=1).item()
        return index_to_word[predicted_index]


if __name__ == "__main__":
    # Extract a vocabulary from the Brown corpus
    corpus_words = [word.lower() for word in brown.words()[:10000]]
    unique_words = list(set(corpus_words)) + ["<UNK>"]
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for word, i in word_to_index.items()}

    VOCAB_SIZE = len(word_to_index)
    EMBEDDING_DIM = 100
    CONTEXT_BEFORE = 2
    CONTEXT_AFTER = 2
    EPOCHS = 500

    emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

    # Initialize model, loss, and optimizer
    model = ContinuousBagOfWords(VOCAB_SIZE, EMBEDDING_DIM)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate training data
    training_data = []
    training_labels = []
    for i in range(CONTEXT_BEFORE, len(corpus_words) - CONTEXT_AFTER):
        context = [
            word_to_index.get(corpus_words[j], word_to_index["<UNK>"])
            for j in range(i - CONTEXT_BEFORE, i + CONTEXT_AFTER + 1) if j != i
        ]
        target = word_to_index.get(corpus_words[i], word_to_index["<UNK>"])
        training_data.append(context)
        training_labels.append(target)

    # Convert training data to tensors
    train_inputs, train_labels = torch.tensor(training_data, dtype=torch.long), \
        torch.tensor(training_labels, dtype=torch.long)

    # Training loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output_probs = model(train_inputs)
        loss = loss_function(output_probs, train_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

    # Example prediction
    example_context = ["the", "quick", "brown", "fox"]
    predicted_word = model.predict(example_context, word_to_index, index_to_word)
    print("Predicted center word:", predicted_word)
