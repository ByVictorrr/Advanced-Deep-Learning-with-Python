import numpy as np


def step(s, x, U, W):
    return x * U + s * W


def forward(x, U, W):
    # Number of samples in the mini-batch
    number_of_samples = len(x)
    # Length of each sample
    sequence_length = len(x[0])
    # Initialize the state activation
    s = np.zeros((number_of_samples, sequence_length + 1))

    # Update the states over the sequence
    for t in range(0, sequence_length):
        s[:, t + 1] = step(s[:, t], x[:, t], U, W)

    return s


def backward(x, s, y, W):
    sequence_length = len(x[0])
    # The network output is just the last activation of the sequence
    s_t = s[:, -1]
    # compute the gradient of the output w.r.t. MSE cost function at final state
    gS = 2 * (s_t - y)

    # set the gradient accumulations to 0
    gU, gW = 0, 0

    # Accumulate gradients backwards
    for k in range(sequence_length, 0, -1):
        # Compute the parameter gradients and accumulate the results.
        gU += np.sum(gS * x[:, k - 1])
        gW += np.sum(gS * s[:, k - 1])

        # Compute the gradient at the output of the previous layer
        gS = gS * W
    return gU, gW


def train(x, y, epochs, learning_rate=0.0005):
    """Train the network."""
    # set initial parameters
    weights = (-2, 0)  # (U, W)
    # Accumulate the losses and their respective weights
    losses, gradients_u, gradients_w = [], [], []

    # Perform iterative gradient descent
    for i in range(epochs):
        # Perform forward and backward pass to get the gradients
        s = forward(x, weights[0], weights[1])
        # Compute the loss
        loss = (y[0] - s[-1, -1]) ** 2
        # Store the loss and weights values for later display
        losses.append(loss)
        gradients = backward(x, s, y, weights[1])
        gradients_u.append(gradients[0])
        gradients_w.append(gradients[1])

        # Update each parameter 'p' by p= p - (gradient * learning_rate)
        # 'gp' is the gradient of parameter 'p'
        weights = tuple((p - gp * learning_rate) for p, gp in zip(weights, gradients))
    print(f"Weights: {weights}")
    return np.array(losses), np.array(gradients_u), np.array(gradients_w)


def plot_training(losses, gradients_u, gradients_w, vanishing_grad=False):
    import matplotlib.pyplot as plt

    # Remove NaN and Inf values
    losses = losses[np.isfinite(losses)]
    gradients_u = gradients_u[np.isfinite(gradients_u)]
    gradients_w = gradients_w[np.isfinite(gradients_w)]

    # plot the weights U and W
    fig, ax1 = plt.subplots(figsize=(5, 3.4))

    ax1.set_ylim(-3, 600 if vanishing_grad else 20)
    ax1.set_xlabel("epochs")
    ax1.plot(gradients_u, label="grad U", color="blue", linestyle=":")
    ax1.plot(gradients_w, label="grad W", color="red", linestyle="--")
    ax1.legend(loc="upper left")

    # instantiate a second axis that shares the same x-axis
    # plot the loss on the second axis
    ax2 = ax1.twinx()

    # uncomment to plot exploding gradients
    ax2.set_ylim(-3, 200 if vanishing_grad else 10)
    ax2.plot(losses, label="Loss", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def regular_training(epochs):
    # Use these inputs for normal training
    # The first dimension represents the mini-batch
    x = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
    y = np.array([3])
    losses, gradients_u, gradients_w = train(x, y, epochs=epochs)
    plot_training(losses, gradients_u, gradients_w, vanishing_grad=False)


def vanishing_and_exploding_gradients(epochs):
    # Use these inputs to reproduce the exploding gradients scenario
    x = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                   1, 0, 1, 0, 1, 0]])
    y = np.array([12])
    losses, gradients_u, gradients_w = train(x, y, epochs=epochs)
    plot_training(losses, gradients_u, gradients_w, vanishing_grad=True)




if __name__ == "__main__":
    # regular_training(150)
    vanishing_and_exploding_gradients(150)
