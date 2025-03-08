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


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # word embedding layer
        self.linear = nn.Linear(embedding_dim, vocab_size)  # output layer

    def forward(self, target_word):
        """Forward pass of SkipGram:

        1. convert target word to embeddings
        2. pass through linear layer to predict context words
        """
        embedded = self.embeddings(target_word)  # shape: (batch_size, embedding_dim)
        output = self.linear(embedded)  # Linear Layer projection to vocab space
        return output  # return logits (before applying softmax)

    def predict(self, target_word, word_to_index, index_to_word, top_k=4):
        target_index = torch.tensor([word_to_index.get(target_word.lower(), word_to_index["<UNK>"])],
                                    dtype=torch.long)
        output_probs = self.forward(target_index)
        top_context_indices = torch.topk(output_probs, k=top_k, dim=1)[1].squeeze(0).tolist()
        return [index_to_word[idx] for idx in top_context_indices]
if __name__ == "__main__":
    # Extract vocabulary from Brown corpus
    corpus_words = [word.lower() for word in brown.words()[:10000]]
    unique_words = list(set(corpus_words)) + ["<UNK>"]
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for word, i in word_to_index.items()}

    VOCAB_SIZE = len(word_to_index)
    EMBEDDING_DIM = 100
    CONTEXT_BEFORE = 2
    CONTEXT_AFTER = 2
    EPOCHS = 10000

    # Initialize model, loss function, optimizer
    model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate training data for Skip-Gram
    training_data = []
    training_labels = []

    for i in range(CONTEXT_BEFORE, len(corpus_words) - CONTEXT_AFTER):
        target = word_to_index.get(corpus_words[i], word_to_index["<UNK>"])
        context = [
            word_to_index.get(corpus_words[j], word_to_index["<UNK>"])
            for j in range(i - CONTEXT_BEFORE, i + CONTEXT_AFTER + 1) if j != i
        ]
        # Store (target, context_word) pairs separately
        for context_word in context:
            training_data.append(target)
            training_labels.append(context_word)

    # Convert training data to tensors
    train_inputs = torch.tensor(training_data, dtype=torch.long)
    train_labels = torch.tensor(training_labels, dtype=torch.long)

    # Training loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output_probs = model(train_inputs)
        loss = loss_function(output_probs, train_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

    # Example Prediction
    example_target = "fox"
    predicted_context_words = model.predict(example_target, word_to_index, index_to_word)
    print(f"Predicted context words for '{example_target}':", predicted_context_words)