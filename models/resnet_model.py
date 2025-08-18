import torch.nn as nn
from torchvision.models import resnet18

class ShapeCounterResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18(pretrained=True)
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Entrada: 1 canal
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, 3)  # salida: [círculo, triángulo, cuadrado]

    def forward(self, x):
        return self.base(x)