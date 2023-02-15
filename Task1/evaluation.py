import torch
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple

class Evaluator:
    # Model evaluation for MNIST classification 
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        # Model evaluation: return the model accuracy over test set 
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

