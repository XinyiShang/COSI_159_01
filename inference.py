import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class Inferencer:
    # Model inference for MNIST classification
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def infer(self, sample: Tensor) -> int:
        # Model inference: input an image, return its class index 
        with torch.no_grad():
            output = self.model(sample.unsqueeze(0))
            pred = output.argmax(dim=1).item()
        return pred
