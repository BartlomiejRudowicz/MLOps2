import sys
import os

# 1. Mówimy Pythonowi: "Szukaj plików również w folderze, w którym jesteś (czyli w 'src')"
sys.path.append(os.path.dirname(__file__))

import bentoml
import torch
import numpy as np

# 2. Jawnie importujemy klasę Twojego modelu, żeby PyTorch nie miał problemów przy rozpakowywaniu
from model import MLPClassifier

@bentoml.service
class BreastCancerService:
    # Używamy nowszego sposobu ładowania, zgodnie z sugestią BentoML
    bento_model = bentoml.models.BentoModel("breast_cancer_mlp:latest")

    def __init__(self):
        self.model = bentoml.pytorch.load_model(self.bento_model)
        self.model.eval()

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        tensor_data = torch.tensor(input_data, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(tensor_data)
            predictions = torch.argmax(logits, dim=1)
            
        return predictions.numpy()