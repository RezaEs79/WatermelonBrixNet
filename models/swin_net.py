import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Swin_V2_T_Weights

class SwinNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SwinNet, self).__init__()
        
        # --- 1. Audio Branch: Swin Transformer Tiny ---
        self.audio_net = models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        self.feat_dim = self.audio_net.head.in_features  # 768
        self.audio_net.head = nn.Identity()

        # --- 2. Skin Branch: Swin Transformer Tiny ---
        self.skin_net = models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        self.skin_net.head = nn.Identity()

        # --- 3. Fusion Head (Simplified per Colab) ---
        self.fusion_dim = self.feat_dim * 2 + 1 # 1537
        
        # Architecture exactly matching your Colab implementation:
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes) # Final output (1 or 5)
        )

    def forward(self, audio, skin, mass):
        a_feat = self.audio_net(audio)
        s_feat = self.skin_net(skin)
        
        combined = torch.cat([a_feat, s_feat, mass], dim=1)
        
        return self.head(combined)