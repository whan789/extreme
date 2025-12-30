import torch
import torch.nn as nn
import torch.nn.functional as F

class EVTAggregator(nn.Module):
    def __init__(self, configs, C, D, Nk, t_high_np, t_low_np):
        super().__init__()
        # ★ 수정: configs.enc_in(전체 채널)이 아니라 인자로 받은 C(라우팅된 채널 수)를 사용해야 합니다.
        self.enc_in = C 
        self.d_model = D 
        
        # Normal Context Summary (Attention 기반)
        # 쿼리 파라미터 역시 라우팅된 채널 수에 맞춰 생성됩니다.
        self.query = nn.Parameter(torch.randn(self.enc_in, 1, D))
        self.attn_normal = nn.MultiheadAttention(embed_dim=D, num_heads=1, batch_first=True)
        self.norm_normal = nn.LayerNorm(D)

        # EVT 구성 요소 (Threshold 고정 지표)
        # 이제 t_high_np의 크기(C)와 self.enc_in(C)이 일치하므로 에러가 발생하지 않습니다.
        self.register_buffer('t_high', torch.from_numpy(t_high_np).float().view(1, self.enc_in, 1, 1))
        self.register_buffer('t_low', torch.from_numpy(t_low_np).float().view(1, self.enc_in, 1, 1))
        
        self.peak_evt_proj = nn.Linear(2, self.d_model)
        self.trough_evt_proj = nn.Linear(2, self.d_model)
        
        # 채널별 융합 가중치 (Normal, Peak, Trough)
        # self.alpha = nn.Parameter(torch.randn(self.enc_in, 3))
        self.alpha = nn.Parameter(torch.zeros(self.enc_in, 3))

    def forward(self, patch_latent, raw_patches):
        B, C, Nk, D = patch_latent.shape
        
        # 1. Normal Context Summary 생성
        patch_latent_reshaped = patch_latent.reshape(B * C, Nk, D)
        # self.enc_in이 C와 같으므로 q의 배치 차원이 B*C와 일치하게 됩니다.
        q = self.query.repeat(B, 1, 1) 
        pooled_norm, _ = self.attn_normal(q, patch_latent_reshaped, patch_latent_reshaped)
        z_normal = self.norm_normal(pooled_norm).view(B, C, 1, D)

        # 2. EVT Statistics 추출 (POT 방식)
        # Peak (초과분 통계)
        mask_high = raw_patches > self.t_high
        exceed_high = torch.where(mask_high, raw_patches - self.t_high, 0.)
        stat_high = torch.cat([
            exceed_high.max(dim=-1, keepdim=True)[0],
            exceed_high.sum(dim=-1, keepdim=True) / (mask_high.sum(dim=-1, keepdim=True) + 1e-6)
        ], dim=-1)
        z_peak = self.peak_evt_proj(stat_high).mean(dim=2, keepdim=True)

        # Trough (미달분 통계)
        mask_low = raw_patches < self.t_low
        exceed_low = torch.where(mask_low, self.t_low - raw_patches, 0.)
        stat_low = torch.cat([
            exceed_low.max(dim=-1, keepdim=True)[0],
            exceed_low.sum(dim=-1, keepdim=True) / (mask_low.sum(dim=-1, keepdim=True) + 1e-6)
        ], dim=-1)
        z_trough = self.trough_evt_proj(stat_low).mean(dim=2, keepdim=True)

        # 3. Summary Token 융합 (B, C, 1, D)
        pooled_stack = torch.stack([z_normal, z_peak, z_trough], dim=-1)
        score = F.softmax(self.alpha, dim=-1).view(1, C, 1, 1, 3)
        summary_token = (pooled_stack * score).sum(dim=-1)

        # 4. 최종 결합: 원본 패치 + 요약 토큰 (B, C, N+1, D)
        return torch.cat([patch_latent, summary_token], dim=2)