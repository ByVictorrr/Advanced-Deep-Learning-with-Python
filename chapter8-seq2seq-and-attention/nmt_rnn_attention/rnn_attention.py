from typing import Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nmt_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_SIZE = 40000
HIDDEN_SIZE = 128


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Embedding for the input words
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # the actual rnn cell
        self.rnn_cell = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, inpt, hidden):
        """Single sequence encoder step."""

        # Embed the input token:
        # x_t ∈ ℝ^E, where E is the embedding dimension
        embedded = self.embedding(inpt).view(1, 1, -1)

        # Update the encoder hidden state:
        # h_t = RNN(x_t, h_{t-1}), where h_t ∈ ℝ^d
        output, hidden = self.rnn_cell(embedded, hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(torch.nn.Module):
    """Regular decoder RNN (no attention)"""

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Embedding for the current input word
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        # decoder cell
        self.rnn_cell = torch.nn.GRU(hidden_size, hidden_size)
        # current output word; FC layer
        self.out = torch.nn.Linear(hidden_size, output_size)
        # self.out LogSoftmax activation
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inpt, hidden, *args):
        # Embed the input token:
        # x_t ∈ ℝ^E, where E is the embedding dimension
        embedded = self.embedding(inpt).view(1, 1, -1)

        # Apply nonlinearity to the embedding (optional, task-dependent):
        # x̃_t = ReLU(x_t)
        embedded = torch.nn.functional.relu(embedded)

        # Update the decoder hidden state:
        # s_t = RNN(x̃_t, s_{t-1}), where s_t ∈ ℝ^d
        output, hidden = self.rnn_cell(embedded, hidden)

        # Compute vocabulary distribution over the next token:
        # y_t = log softmax(W · s_t + b), projecting hidden state to ℝ^{|V|}
        output = self.log_softmax(self.out(output[0]))

        return output, hidden, args

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(torch.nn.Module):
    """RNN decoder with the attention."""

    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        # Embedding for the input word
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        # Attention portion
        self.attn = torch.nn.Linear(hidden_size, hidden_size)
        self.w_c = torch.nn.Linear(hidden_size * 2, hidden_size)
        # RNN
        self.rnn_cell = torch.nn.GRU(hidden_size, hidden_size)
        # output word
        self.w_y = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inpt, hidden, encoder_outputs):
        # Embed the current input token and apply dropout:
        # x_t ∈ ℝ^E, where E is the embedding dimension
        embedded = self.embedding(inpt).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Update the decoder hidden state:
        # s_t = RNN(x_t, s_{t-1}), where s_t ∈ ℝ^d
        rnn_out, hidden = self.rnn_cell(embedded, hidden)

        # Compute attention alignment scores for all encoder time steps:
        # e_{t,i} = score(s_t, h_i) = s_t^T * h_i  (Luong dot-product attention)
        alignment_scores = torch.mm(self.attn(hidden[0]), encoder_outputs.t())

        # Compute normalized attention weights:
        # α_{t,i} = softmax(e_{t,i}) = exp(e_{t,i}) / Σ_j exp(e_{t,j})
        attn_weights = torch.nn.functional.softmax(alignment_scores, dim=1)

        # Compute the context vector as the weighted sum over encoder hidden states:
        # c_t = Σ_i α_{t,i} * h_i = attn_weights · H, where H ∈ ℝ^{T×d}
        c_t = torch.mm(attn_weights, encoder_outputs)

        # Concatenate decoder state s_t and context vector c_t:
        # [s_t ; c_t] ∈ ℝ^{2d}, combining local decoding and global attention context
        hidden_s_t = torch.cat([hidden[0], c_t], dim=1)

        # Apply a linear transformation and nonlinearity to get attentional hidden state:
        # ṡ_t = tanh(W_c · [s_t ; c_t]), projecting to ℝ^d
        hidden_s_t = torch.tanh(self.w_c(hidden_s_t))

        # Compute final output logits over the vocabulary and apply log-softmax:
        # y_t = log softmax(W_y · ṡ_t), where output ∈ ℝ^|V|
        output = torch.nn.functional.log_softmax(self.w_y(hidden_s_t), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(encoder: EncoderRNN, decoder: Union[DecoderRNN, AttnDecoderRNN], data_loader, max_len=MAX_LENGTH):
    # Initialize optimizers for encoder and decoder parameters
    encoder_optimizer = torch.optim.Adam(encoder.parameters())
    decoder_optimizer = torch.optim.Adam(decoder.parameters())

    # Define negative log-likelihood loss for classification over vocabulary
    loss_func = torch.nn.NLLLoss()

    print_loss_total = 0

    # Iterate over the training dataset
    for i, (input_tensor, target_tensor) in enumerate(data_loader):
        # Move data to the appropriate device and remove singleton batch dimension
        input_tensor = input_tensor.to(device).squeeze(0)
        target_tensor = target_tensor.to(device).squeeze(0)

        # Initialize the initial hidden state of the encoder: h_0
        encoder_hidden = encoder.init_hidden()

        # Clear gradients from the previous step
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        # Pre-allocate tensor to store encoder outputs for attention: H = {h_1, ..., h_T}
        encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

        # Initialize loss accumulator
        loss = torch.Tensor([0]).squeeze().to(device)

        with torch.set_grad_enabled(True):
            # Encode the input sequence: h_t = Encoder(x_t, h_{t-1})
            for ei in range(input_length):
                enc_out, encoder_hidden = encoder(inpt=input_tensor[ei], hidden=encoder_hidden)
                encoder_outputs[ei] = enc_out[0, 0]  # Store output for attention

            # Initialize decoder input with <GO> token: y_0 = <GO>
            decoder_input = torch.tensor([[GO_token]], device=device)

            # Initialize decoder hidden state with the final encoder hidden state: s_0 = h_T
            decoder_hidden = encoder_hidden

            # Decode target sequence using teacher forcing
            for di in range(target_length):
                # Predict next token and compute attention context
                # y_t, s_t, α_t = Decoder(y_{t-1}, s_{t-1}, H)
                dec_out, decoder_hidden, dec_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

                # Compute loss: L += -log P(y_t = target_t)
                loss += loss_func(dec_out, target_tensor[di])

                # Teacher forcing: feed the actual target token as next input
                decoder_input = target_tensor[di]

            # Backpropagate loss and update model parameters
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Accumulate normalized loss for reporting
        print_loss_total += loss.item() / target_length

        # Print average loss every 1000 iterations
        it = i + 1
        if it % 1000 == 0:
            print_loss_avg = print_loss_total / 1000
            print_loss_total = 0
            print(f"Iteration {it}\n"
                  f"Loss: {print_loss_avg:.4f}\n")


def evaluate(encoder: EncoderRNN, decoder: Union[DecoderRNN, AttnDecoderRNN], input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():  # Disable gradient tracking during inference

        # Get input sequence length (T)
        input_length = input_tensor.size()[0]

        # Initialize encoder hidden state: h_0
        encoder_hidden = encoder.init_hidden()

        # Move input tensor to device
        input_tensor = input_tensor.to(device)

        # Pre-allocate tensor to store encoder outputs H = {h_1, ..., h_T}
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # Encode the input sequence: h_t = Encoder(x_t, h_{t-1})
        for ei in range(input_length):
            enc_out, encoder_hidden = encoder(inpt=input_tensor[ei], hidden=encoder_hidden)
            encoder_outputs[ei] = enc_out[0, 0]  # Save encoder output for attention

        # Initialize decoder input with <GO> token: y_0 = <GO>
        decoder_input = torch.tensor([[GO_token]], device=device)

        # Initialize decoder hidden state with final encoder state: s_0 = h_T
        decoder_hidden = encoder_hidden

        # Containers for the decoded sequence and attention weights
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        # Generate output sequence without teacher forcing
        for di in range(max_length):
            # y_t, s_t, α_t = Decoder(y_{t-1}, s_{t-1}, H)
            dec_out, decoder_hidden, dec_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Store attention weights for visualization or analysis
            decoder_attentions[di] = dec_attn.data

            # Select the token with the highest probability: y_t = argmax P(y)
            _, topi = dec_out.data.topk(1)

            if topi.item() != EOS_token:
                # Convert predicted index back to word
                decoded_words.append(dataset.output_lang.index2word[topi.item()])
            else:
                # Stop if end-of-sequence token is predicted
                break

            # Use predicted token as input for next step: y_{t-1} = y_t
            decoder_input = topi.squeeze().detach()

        # Return decoded word list and attention matrix (up to decoded length)
        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        sample = random.randint(0, len(dataset.dataset) - 1)
        pair = dataset.pairs[sample]
        input_sequence = dataset[sample][0].to(device)

        output_words, attentions = evaluate(encoder, decoder, input_sequence)

        print(f"INPUT: {pair[0]}; TARGET: {pair[1]}; RESULT: {' '.join(output_words)}")


def plot_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_plot_attention(input_sentence, encoder, decoder):
    input_tensor = dataset.sentence_to_sequence(input_sentence).to(device)

    output_words, attentions = evaluate(encoder=encoder,
                                        decoder=decoder,
                                        input_tensor=input_tensor)

    print(f"INPUT: {input_sentence}; OUTPUT: {' '.join(output_words)}")
    plot_attention(input_sentence, output_words, attentions)


if __name__ == "__main__":
    dataset = NMTDataset('eng-fra.txt', DATASET_SIZE)

    enc = EncoderRNN(dataset.input_lang.n_words, HIDDEN_SIZE).to(device)
    dec = AttnDecoderRNN(HIDDEN_SIZE, dataset.output_lang.n_words, dropout=0.1).to(device)

    # dec = DecoderRNN(HIDDEN_SIZE, dataset.output_lang.n_words).to(device)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False)

    train(enc, dec, data_loader=train_loader)
    evaluate_randomly(enc, dec)

    output_words, attentions = evaluate(
        enc, dec, dataset.sentence_to_sequence("je suis trop froid .").to(device))
    plt.matshow(attentions.numpy())
    evaluate_and_plot_attention("elle a cinq ans de moins que moi .", enc, dec)

    evaluate_and_plot_attention("elle est trop petit .", enc, dec)

    evaluate_and_plot_attention("je ne crains pas de mourir .", enc, dec)

    evaluate_and_plot_attention("c est un jeune directeur plein de talent .", enc, dec)
