import torch
import matplotlib.pyplot as plt

from pathlib import Path

def plot_loss_curves(results):
  epochs = range(len(results["train_loss"]))
  # Loss
  plt.figure(figsize=(15,7))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, results["train_loss"], label="Train")
  plt.plot(epochs, results["test_loss"], label="Test")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, results["train_acc"], label="Train")
  plt.plot(epochs, results["test_acc"], label="Test")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
