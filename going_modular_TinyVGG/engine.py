import torch

from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
from timeit import default_timer as timer

def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device):

  model.train()
  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)

  # Compute average loss and accuracy per batch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc

def test_step(model: nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module,
              device):
  model.eval()
  test_loss, test_acc = 0, 0
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      test_loss += loss
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1),dim=-1)
      test_acc += (y_pred_class==y).sum().item()/len(y_pred)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss, test_acc

def train(model: nn.Module,
          epochs: int,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          device):

  results = {"train_loss":[],
             "train_acc":[],
             "test_loss":[],
             "test_acc":[],
             "train_time":None}
  t0 = timer()
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model,
                                       train_dataloader,
                                       loss_fn,
                                       optimizer,
                                       device)
    test_loss, test_acc = test_step(model,
                                    test_dataloader,
                                    loss_fn,
                                    device)
    # Update results per epoch
    results["train_loss"].append(train_loss.to("cpu").detach().numpy())
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss.to("cpu").detach().numpy())
    results["test_acc"].append(test_acc)

    log = f"Epoch: {epoch} | " \
          f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}% " \
          f"Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.2f}%"
    print(log)

  t1 = timer()
  results["train_time"] = t1 - t0
  print(f"Training time: {t1 - t0:.3f} seconds")

  return results
