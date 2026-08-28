import torch.nn as nn
from torchvision import models

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    
    if architecture == "resnet18":
        
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    else:
        raise ValueError(f"Architecture '{architecture}' is not supported.")