import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. Cross-Attention Block
# ==========================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key_value):
        q = self.norm_q(query)
        k = self.norm_k(key_value)
        v = k 
        
        # Attention
        attn_out, _ = self.mha(q, k, v)
        
        # Residual + FF
        x = query + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x

# ==========================================
# 2. CrossNet Architecture
# ==========================================
class CrossNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CrossNet, self).__init__()

        # --- Audio Encoder (ResNet18 truncated) ---
        base_audio = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # حذف لایه‌های Pooling آخر (AvgPool + FC) برای حفظ ابعاد مکانی (4x4)
        self.audio_extractor = nn.Sequential(*list(base_audio.children())[:-2]) 
        
        # --- Skin Encoder (ResNet18 truncated) ---
        base_skin = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # حذف لایه‌های Pooling آخر برای حفظ ابعاد مکانی (7x7)
        self.skin_extractor = nn.Sequential(*list(base_skin.children())[:-2]) 

        # --- Cross Attention Setup ---
        self.embed_dim = 512
        self.cross_attn = CrossAttentionBlock(embed_dim=self.embed_dim, num_heads=8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # --- Fusion Head ---
        # ورودی: 512 (Audio) + 512 (Skin) + 1 (Mass) = 1025
        self.fusion_dim = 512 + 512 + 1
        
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes) # خروجی نهایی (1 یا 5)
        )

    def forward(self, audio, skin, mass):
        # 1. Extract Spatial Features
        # Audio: [B, 512, 4, 4] -> Flatten -> [B, 512, 16] -> Permute -> [B, 16, 512]
        a_feat_map = self.audio_extractor(audio)
        b, c, h, w = a_feat_map.shape
        a_seq = a_feat_map.view(b, c, -1).permute(0, 2, 1) 
        
        # Skin: [B, 512, 7, 7] -> Flatten -> [B, 512, 49] -> Permute -> [B, 49, 512]
        s_feat_map = self.skin_extractor(skin)
        b, c, h, w = s_feat_map.shape
        s_seq = s_feat_map.view(b, c, -1).permute(0, 2, 1)

        # 2. Cross Attention (Audio attends to Skin)
        # خروجی: ویژگی‌های صدا که با اطلاعات تصویر اصلاح شده‌اند
        a_enriched = self.cross_attn(query=a_seq, key_value=s_seq) # [B, 16, 512]

        # 3. Pooling (Sequence to Vector)
        a_vec = self.avg_pool(a_enriched.permute(0, 2, 1)).squeeze(-1) # [B, 512]
        s_vec = self.avg_pool(s_seq.permute(0, 2, 1)).squeeze(-1)      # [B, 512]

        # 4. Final Fusion
        combined = torch.cat([a_vec, s_vec, mass], dim=1)
        
        return self.head(combined)