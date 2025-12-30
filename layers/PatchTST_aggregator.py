import torch
import torch.nn as nn

class PatchTSTAggregator(nn.Module):
    def __init__(self, C, D, Nk, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.C, self.D, self.Nk = C, D, Nk
        
        # 1. Learnable Global Token (Summary 역할을 할 빈 토큰)
        self.learnable_token = nn.Parameter(torch.randn(C, 1, D))
        
        # 2. PatchTST Encoding Layer (Transformer Encoder Layer)
        # 각 채널별로 패치 간의 관계를 학습하도록 설정 (Channel-Independence 가속)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, activation='gelu', batch_first=True
        )

    def forward(self, patch_latent):
        B, C, Nk, D = patch_latent.shape
        
        # Learnable Token 확장 및 패치와 결합: (B, C, N+1, D)
        l_token = self.learnable_token.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([patch_latent, l_token], dim=2) 
        
        # 인코딩 과정 (PatchTST 스타일의 MHA -> FFN)
        # (B*C, N+1, D) 차원으로 변환하여 병렬 처리
        x_reshaped = x.reshape(B * C, Nk + 1, D)
        encoded_x = self.encoder_layer(x_reshaped)
        
        # 다시 (B, C, N+1, D)로 복구하여 반환
        return encoded_x.view(B, C, Nk + 1, D)