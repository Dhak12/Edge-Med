import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config import NUM_CLASSES


def create_model():

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Freeze the pretrained feature extractor initially
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace final classification layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )

    return model
