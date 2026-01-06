import torch
import torch.nn as nn
import torchvision.models as models
from .layers import ResNetBasicBlockWithCBAM

class FusionNet(nn.Module):
    def __init__(self, num_classes=1):
        super(FusionNet, self).__init__()
        
        # Audio Branch (ResNet18 + CBAM)
        self.audio_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Inject CBAM
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.audio_net, name)
            for i in range(len(layer)):
                layer[i] = ResNetBasicBlockWithCBAM(layer[i])
        
        self.audio_dim = self.audio_net.fc.in_features
        self.audio_net.fc = nn.Identity()

        # Skin Branch (ResNet18 Standard)
        self.skin_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.skin_dim = self.skin_net.fc.in_features
        self.skin_net.fc = nn.Identity()

        # Fusion Head (Concatenation)
        self.fusion_dim = self.audio_dim + self.skin_dim + 1 # +1 for Mass
        
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio, skin, mass):
        a_feat = self.audio_net(audio)
        s_feat = self.skin_net(skin)
        
        combined = torch.cat((a_feat, s_feat, mass), dim=1)
        
        return self.head(combined)