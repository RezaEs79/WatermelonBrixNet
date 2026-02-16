import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

class ConvNeXtNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNeXtNet, self).__init__()
        
        # --- 1. Audio Branch: ConvNeXt Tiny ---
        self.audio_net = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Extract the output feature dimension (768)
        self.feat_dim = self.audio_net.classifier[2].in_features 
        
        # Remove the final classification layer
        self.audio_net.classifier = nn.Identity()

        # --- 2. Skin Branch: ConvNeXt Tiny ---
        self.skin_net = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.skin_net.classifier = nn.Identity()

        # --- 3. Fusion Head ---
        # 768 + 768 + 1 = 1537
        self.fusion_dim = self.feat_dim * 2 + 1
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio, skin, mass):
        # Flatten is CRITICAL here because ConvNeXt output is (B, 768, 1, 1)
        a_feat = self.audio_net(audio).flatten(1)
        s_feat = self.skin_net(skin).flatten(1)
        
        combined = torch.cat([a_feat, s_feat, mass], dim=1)
        return self.head(combined)