import torch
import torch.nn as nn
import torchvision.models as models
from .layers import GatedMultimodalUnit

class GatedNet(nn.Module):
    def __init__(self, num_classes=1):
        super(GatedNet, self).__init__()
        
        # Audio Branch (EfficientNetV2 Small)
        self.audio_net = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.feat_dim = self.audio_net.classifier[1].in_features
        self.audio_net.classifier = nn.Identity()

        # Skin Branch (EfficientNetV2 Small)
        self.skin_net = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.skin_net.classifier = nn.Identity()

        # Fusion (Gated Unit)
        self.gmu = GatedMultimodalUnit(input_dim=self.feat_dim, hidden_dim=512)

        # Head
        self.head = nn.Sequential(
            nn.Linear(512 + 1, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio, skin, mass):
        a_feat = self.audio_net(audio)
        s_feat = self.skin_net(skin)
        
        fusion = self.gmu(a_feat, s_feat)
        
        final_input = torch.cat([fusion, mass], dim=1)
        
        return self.head(final_input)