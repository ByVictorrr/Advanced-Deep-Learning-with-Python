import torch
import torch.nn as nn
import torch.optim as optim
import random
import nltk
from nltk.corpus import brown

# Download a lightweight corpus
nltk.download('brown')


class NeuralProbabilisticLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NeuralProbabilisticLanguageModel, self).__init__()

        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hidden Layer
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.tanh = nn.Tanh()

        # Output Layer (Softmax)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, context_words):
        nn.functional.one_hot()
        # Lookup embeddings for context words
        embeddings = self.embedding(context_words)

        # Flatten and pass through hidden layer
        x = embeddings.view(embeddings.shape[0], -1)
        h = self.tanh(self.hidden(x))

        # Compute output probabilities
        output = self.softmax(self.output(h))
        return output


if __name__ == "__main__":
    # Extract a vocabulary from the Brown corpus
    corpus_words = [word.lower() for word in brown.words()]  # Increased to 10000 words
    unique_words = list(set(corpus_words))
    unique_words.append("<UNK>")  # Add unknown token
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    VOCAB_SIZE = len(word_to_index)  # Vocabulary size based on actual words
    EMBEDDING_DIM = 100  # Word embedding dimension
    CONTEXT_SIZE = 3  # Number of previous words used as context
    HIDDEN_DIM = 128  # Hidden layer size
    EPOCHS = 250  # Number of training epochs

    # Initialize model, loss, and optimizer
    model = NeuralProbabilisticLanguageModel(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Generate example context using Brown corpus words
    training_data = [
        [word_to_index.get(random.choice(corpus_words), word_to_index["<UNK>"]) for _ in range(CONTEXT_SIZE)]
        for _ in range(1000)  # Generating 1000 context samples
    ]
    training_labels = [word_to_index.get(random.choice(corpus_words), word_to_index["<UNK>"]) for _ in range(1000)]

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


    # Example function to predict the next word
    def predict_next_word(context_words):
        context_indices = [word_to_index.get(word.lower(), word_to_index["<UNK>"]) for word in context_words]
        context_tensor = torch.tensor([context_indices], dtype=torch.long)
        output_probs = model(context_tensor)
        predicted_index = torch.argmax(output_probs, dim=1).item()
        return index_to_word[predicted_index]


    # Example prediction
    example_context = ["the", "quick", "brown"]  # Provide some example words from the corpus
    predicted_word = predict_next_word(example_context)
    print("Predicted next word:", predicted_word)

    # Example forward pass with a batch of training data
    output_probs = model(train_inputs)
    print(output_probs.shape)  # Output should have shape (1000, VOCAB_SIZE) indicating word probabilities
