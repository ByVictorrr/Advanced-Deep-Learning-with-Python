import torch
import torchtext


class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, pad_idx):
        super().__init__()
        # Embedding field
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_idx,
        )
        # LSTM Cell
        self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)
        # Fully connected output
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, text_sequence, text_lengths):
        # Extract embedding vectors
        embeddings = self.embedding(text_sequence)
        # Pad the sequences to equal length
        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_sequence)
        return self.fc(hidden)


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, batch in enumerate(data_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        text, text_lengths = batch.text

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(text, text_lengths).squeeze()
            loss = loss_function(outputs, batch.label)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * text_lengths.shape[0]
        current_acc += torch.sum(torch.round(torch.sigmoid(outputs)).round() == batch.label.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print(f"Train Loss: {total_loss:.4f}; Accuracy: {total_acc:.4f}")


def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, batch in enumerate(data_loader):
        text, text_lengths = batch.text

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(text, text_lengths).squeeze()
            loss = loss_function(outputs, batch.label)

        # statistics
        current_loss += loss.item() * text_lengths.shape[0]
        current_acc += torch.sum(torch.round(torch.sigmoid(outputs)).round() == batch.label.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print(f"Test Loss: {total_loss:.4f}; Accuracy: {total_acc:.4f}")

    return total_loss, total_acc


if __name__ == "__main__":
    EMBEDDING_SIZE = 100
    HIDDEN_SIZE = 256
    TEXT = torchtext.data.Field(
        tokenize="spacy",  # use SpaCy tokenizer
        lower=True,  # convert all letters to lower case
        include_lengths=True,  # include the length of the movie review
    )
    LABEL = torchtext.data.LabelField(dtype=torch.float)
    # Dataset splits
    train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)
    # Build glove vocabulary
    TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=100))
    LABEL.build_vocab(train)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make iterator for splits
    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, test),
        sort_within_batch=True,
        batch_size=64,
        device=device
    )
    model = LSTMModel(
        vocab_size=len(TEXT.vocab),
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=1,
        pad_idx=TEXT.vocab.stoi[TEXT.pad_token]
    )

    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model.embedding.weight.data[TEXT.vocab.stoi[TEXT.unk_token]] = torch.zeros(EMBEDDING_SIZE)
    model.embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(EMBEDDING_SIZE)

    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCEWithLogitsLoss().to(device)

    model = model.to(device)

    for epoch in range(5):
        print(f"Epoch {epoch + 1}/5")
        train_model(model, loss_function, optimizer, train_iter)
        test_model(model, loss_function, test_iter)
