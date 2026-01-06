import torch
import torch.nn as nn

# ==========================================
# 1. Attention Blocks (CBAM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class ResNetBasicBlockWithCBAM(nn.Module):
    def __init__(self, original_block, cbam_ratio=16):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.cbam = CBAM(original_block.conv2.out_channels, ratio=cbam_ratio)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# ==========================================
# 2. Fusion Blocks (GMU)
# ==========================================
class GatedMultimodalUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedMultimodalUnit, self).__init__()
        self.audio_fc = nn.Linear(input_dim, hidden_dim)
        self.skin_fc = nn.Linear(input_dim, hidden_dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU() 

    def forward(self, audio, skin):
        h_audio = self.activation(self.audio_fc(audio))
        h_skin = self.activation(self.skin_fc(skin))
        
        combined = torch.cat([h_audio, h_skin], dim=1)
        z = self.gate_fc(combined)
        
        h_fusion = z * h_audio + (1 - z) * h_skin
        return self.norm(h_fusion)