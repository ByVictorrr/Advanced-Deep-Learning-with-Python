# objective apply an advanced ImageNet pretrained network on the CIFAR-10 images
# we will use both types of TL's
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms

batch_size = 50

# Define the training dataset. We have considered a few things:
#   1. Upsampled due to the CIFAR-10 images are 32x32, while the ImageNet network expects 224x224 input.
#   2. Standardized the CIFAR-10 data using the imageNet mean and std deviation since this is what the network expected.
#   3. Also add some data augmentation in the form of random horizontal or vertical flips
train_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4821, 0.4465), std=(0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True,
                                         download=True,
                                         transform=train_data_transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           )

# Follow the same steps with the validation/test data, but this time without augmentation
val_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    # (R, G, B) all the channels
    transforms.Normalize(mean=(0.4914, 0.4821, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])
val_set = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=val_data_transform)
val_order = torch.utils.data.DataLoader(val_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=2)
# Choose device, preferably a GPU with a fallback on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define the training of the model - iterates once over the whole training set (one epoch) and applies the optimizer after each forward pass
def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()
    current_loss = 0.0
    current_acc = 0
    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += loss_function(outputs, labels)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Accuracy: {total_acc:.4f}")


# define the testing/validation of the model; skipping the backpropagation part
def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()
    current_loss = 0.0
    current_acc = 0
    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Accuracy: {total_acc:.4f}")
    return total_loss, total_acc


# define the first TL scenario, where we use the pretrained network as a feature extractor:
#   1. We will use a popular network known as ResNet-18. Pytorch automatically download the pretrained weights
#   2. Replace the last network layer with a new layer with 10 outputs (on for each CIFAR-10 classes)
#   3. Exclude the existing network layers from the backward pass and only pass the newly added fully-connected layer to the Adam optimizer
#   4. Run the training for epochs and evaluate the network accuracy after each epoch
#   5. Plot the test accuracy with the help of the `plot_accuracy` function.
def tl_feature_extractor(epochs=5):
    # load the pretrained model
    model = torchvision.models.resnet18(pretrained=True)
    # exclude existing parameters from backward pass for performance
    for param in model.parameters():
        param.requires_grad = False

    # newly constructed layers have `requires_grad=True` by default
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features, out_features=10)

    # transfer to GPU (if available)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    # only parameters of the final layers are being optimized
    optimizer = optim.Adam(model.fc.parameters())
    # train
    test_acc = []  # collect accuracy for plotting
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_model(model, loss_function, optimizer, train_loader)
        _, acc = test_model(model, loss_function, val_order)
        test_acc.append(acc)

    plot_accuracy(test_acc)


# implement the fine-tuning approach - similar to `tl_feature_extractor`, but here, we are training the whole network
def tl_fine_tuning(epochs=5):
    # load the pretained model
    model = models.resnet18(pretrained=True)
    # replace the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_features, out_features=10)
    # transfer the model to the gpu
    model = model.to(device)

    # loss function
    loss_function = nn.CrossEntropyLoss()
    # we will optimize all parameters`
    optimizer = optim.Adam(model.parameters())
    # train
    test_acc = []  # collect accuracy for plotting
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_model(model, loss_function, optimizer, train_loader)
        _, acc = test_model(model, loss_function, val_order)
        test_acc.append(acc)
    plot_accuracy(test_acc)


def plot_accuracy(accuracy):

    # Ensure accuracy is a CPU-based NumPy array
    if isinstance(accuracy, torch.Tensor):
        accuracy = accuracy.cpu().numpy()

    plt.plot(accuracy)
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transfer learning with feature extraction or fine tuning")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-fe', action='store_true',
                       help="Feature extraction - removing fc layer and only train new layer.")
    group.add_argument('-ft', action='store_true', help="Fine tuning - keeping the fc layer and training")
    args = parser.parse_args()

    if args.ft:
        print("Transfer learning: fine tuning with PyTorch ResNet18 network for CIFAR-10")
        tl_fine_tuning()
    elif args.fe:
        print("Transfer learning: feature extractor with PyTorch ResNet18 network for CIFAR-10")
        tl_feature_extractor()
