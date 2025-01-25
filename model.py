import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0()
        n_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(n_features, 3)
        
    def forward(self, x):
        return self.model(x)