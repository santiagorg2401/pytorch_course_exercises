import os
import torch
import argparse
from torchvision import transforms
import data_setup, engine, model, utils

# Set Hyperparameters
parser = argparse.ArgumentParser(description="Train a TinyVGG model on PyTorch")
parser.add_argument("--num_epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="Number of samples per batch")
parser.add_argument("--hidden_units", type=int, help="TinyVGG hidden units")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--plot", type=bool, help="Plot loss curves")
parser.add_argument("--model_name", help="Model name, include .pt ot .pth")
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
IMG_SIZE = (64, 64)
LEARNING_RATE = args.lr
PLOT = args.plot
MODEL_NAME = args.model_name

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# Create DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                               test_dir,
                                                                               data_transform,
                                                                               BATCH_SIZE)

# Create model
model = model.TinyVGG(3, HIDDEN_UNITS, len(class_names)).to(device)

# Train
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
results = engine.train(model, NUM_EPOCHS, loss_fn, optimizer,
                       train_dataloader, test_dataloader, device)

# Use utils
if PLOT:
  utils.plot_loss_curves(results)
utils.save_model(model, "models", MODEL_NAME)
