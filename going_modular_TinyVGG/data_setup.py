"""
Handles functionality for handling PyTorch DataLoaders
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """
  Creates Training and testing DataLoaders
  Args:
    train_dir: Path to training directory,
    test_dir: Path to testing directory,
    transform: torchvision transforms to perform on data,
    batch_size: Batch size,
    num_workers: Number of workers per dataloader

  Returns:
    (train_dataloader, test_dataloader, class_names)
  """
  # use ImageFolder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform)
  test_data = datasets.ImageFolder(test_dir, transform)

  # Get class names
  class_names = train_data.classes

  # Turn datasets into DataLoaders
  train_dataloader = DataLoader(train_data, batch_size, True,
                                num_workers=num_workers, pin_memory=True)
  test_dataloader = DataLoader(test_data, batch_size, False,
                               num_workers=num_workers, pin_memory=True)

  return train_dataloader, test_dataloader, class_names
