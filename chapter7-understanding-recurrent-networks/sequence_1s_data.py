import numpy as np
import torch


def generate_dataset(sequence_length: int, samples: int):
    """
    Generate training/testing datasets
    :param sequence_length: length of the binary sequence
    :param samples: number of samples
    """

    sequences = list()
    labels = list()
    for i in range(samples):
        a = np.random.randint(sequence_length) / sequence_length
        sequence = list(np.random.choice(2, sequence_length, p=[a, 1 - a]))
        sequences.append(sequence)
        labels.append(int(np.sum(sequence)))

    sequences = np.array(sequences)
    labels = np.array(labels, dtype=np.int8)

    result = torch.utils.data.TensorDataset(
        torch.from_numpy(sequences).float().unsqueeze(-1),
        torch.from_numpy(labels).float())

    return result
