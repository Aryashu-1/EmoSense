import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):  # Adjust based on your dataset
        super(ResNetEmotion, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify first conv layer to accept 1-channel grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final FC layer with custom classifier
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    model = ResNetEmotion()
    print(model)
