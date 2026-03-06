import torch
from torchvision import models

def load_model():

    model = models.resnet18(pretrained=False)

    model.fc = torch.nn.Linear(model.fc.in_features,2)

    model.load_state_dict(
        torch.load("models/malaria_model.pth", map_location="cpu")
    )

    model.eval()

    return model