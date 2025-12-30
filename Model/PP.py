import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from layers.Spectral_Block import SpectralBlock
# 연구자님께서 정의하신 AdaptiveAggregator와 PatchSummary 경로를 임포트합니다.
from layers.Adaptive_Router import Router 
from layers.Patch_Summary import PatchSummary 

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.kd_lambda = configs.kd_lambda
        self.use_teacher = configs.use_teacher
        self.rank = configs.rank
        self.modes = configs.modes
        self.revin = RevIN(self.enc_in)
        self.hidden_dim = configs.kd_hidden_dim
        self.top_k = configs.top_k

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = math.floor((self.seq_len - self.patch_len) / self.stride) + 1
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # 1. run.py에서 주입된 통계량(kurt, t_high, t_low)을 사용하여 라우터 초기화
        # AdaptiveAggregator 내부에서 EVTAggregator와 PatchTSTAggregator를 생성합니다.
        self.patch_router = Router(
            configs=configs,
            kurtosis_np=configs.kurt,
            t_high_np=configs.t_high,
            t_low_np=configs.t_low
        )

        # 2. PatchSummary는 라우터에서 나온 (B, C, N+1, D)를 최종 처리합니다.
        self.patch_summary = PatchSummary(configs, self.seq_len, self.pred_len,
                                             self.patch_len, self.stride, self.d_model, self.rank)
        
        if self.use_teacher:
            self.spectral_block = SpectralBlock(self.enc_in, self.modes, self.top_k)
            self.proj_t = nn.Linear(self.seq_len, self.hidden_dim)
            self.proj_s = nn.Linear(self.d_model, self.hidden_dim)

    def patching(self, x):
        return x.unfold(2, self.patch_len, self.stride)

    def forward(self, x_enc, x_mark=None, x_dec=None, x_mark_dec=None,
                mask=None, return_loss=False, **kwargs):
        
        # [Step 1] EVT 연산을 위한 Raw Patches (정규화 전 상태 보존)
        x_enc_permuted = x_enc.permute(0, 2, 1) # (B, C, T)
        raw_patches = self.patching(x_enc_permuted) # (B, C, Nk, P)
        
        # [Step 2] 모델 학습을 위한 RevIN 정규화 및 패치 임베딩
        x_norm = self.revin(x_enc, mode='norm').permute(0, 2, 1) # (B, C, T)
        patches = self.patching(x_norm)
        patch_latent = self.patch_embedding(patches) # (B, C, Nk, D)

        # [Step 3] Adaptive Router (AdaptiveAggregator) 실행
        # 각 채널의 첨도에 따라 EVT 또는 PatchTST 경로를 거쳐 요약 토큰이 추가됨
        # 출력 차원: (B, C, N+1, D)
        total_latent = self.patch_router(patch_latent, raw_patches) 

        # [Step 4] Patch Branch 최종 출력 생성
        # 주의: PatchSummary.forward는 total_latent 하나만 받도록 수정되어야 합니다.
        y_patch = self.patch_summary(total_latent)  # (B, C, pred_len)

        # [Step 5] Knowledge Distillation (KD)
        if self.use_teacher and return_loss:
            y_freq = self.spectral_block(x_norm)
            teacher = self.proj_t(y_freq) 
            
            # Student: total_latent의 마지막에 위치한 요약 토큰(Summary Token)만 사용
            # (B, C, N+1, D) -> (B, C, D)
            summary_token = total_latent[:, :, -1, :] 
            student = self.proj_s(summary_token)
            
            kd_loss = self.kd_lambda * F.mse_loss(student, teacher.detach())
        else:
            kd_loss = torch.tensor(0.0, device=y_patch.device)

        # [Step 6] 역정규화 및 최종 반환
        output = y_patch.permute(0, 2, 1)
        output = self.revin(output, mode='denorm')

        return output[:, -self.pred_len:, :], kd_loss