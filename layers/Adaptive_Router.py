import torch
import torch.nn as nn
from layers.EVT_aggregator import EVTAggregator
from layers.PatchTST_aggregator import PatchTSTAggregator
import numpy as np
import math

# class Router(nn.Module):
#     def __init__(self, configs, kurtosis_np, t_high_np, t_low_np):
#         super().__init__()
#         # 1. 인덱스 미리 계산 (Initialization 단계에서 한 번만 수행)
#         self.heavy_idx = np.where(kurtosis_np > 3.0)[0]
#         self.normal_idx = np.where(kurtosis_np <= 3.0)[0]
#         self.patch_num = math.floor((configs.seq_len - configs.patch_len) / configs.stride) + 1
        
#         # 2. 각 모듈은 담당하는 채널 수(C_heavy, C_normal)만큼만 생성
#         self.evt_agg = EVTAggregator(configs, len(self.heavy_idx), configs.d_model, self.patch_num, 
#                                      t_high_np[self.heavy_idx], t_low_np[self.heavy_idx])
#         self.tst_agg = PatchTSTAggregator(len(self.normal_idx), configs.d_model, self.patch_num)

#     def forward(self, patch_latent, raw_patches):
#         B, C, N, D = patch_latent.shape
        
#         # 결과를 담을 빈 텐서 생성 (B, C, N+1, D)
#         final_output = torch.zeros((B, C, N + 1, D), device=patch_latent.device)

#         # 3. 실제 물리적 분기 (Routing)
#         if len(self.heavy_idx) > 0:
#             # 첨도가 높은 채널만 EVT 모듈 통과
#             heavy_latent = patch_latent[:, self.heavy_idx, :, :]
#             heavy_raw = raw_patches[:, self.heavy_idx, :, :]
#             final_output[:, self.heavy_idx, :, :] = self.evt_agg(heavy_latent, heavy_raw)

#         if len(self.normal_idx) > 0:
#             # 첨도가 낮은 채론만 PatchTST 모듈 통과
#             normal_latent = patch_latent[:, self.normal_idx, :, :]
#             final_output[:, self.normal_idx, :, :] = self.tst_agg(normal_latent)

#         return final_output # 모든 채널이 합쳐진 (B, C, N+1, D) 반환

class Router(nn.Module):
    def __init__(self, configs, kurtosis_np, t_high_np, t_low_np):
        super().__init__()
        self.enc_in = configs.enc_in
        self.patch_num = math.floor((configs.seq_len - configs.patch_len) / configs.stride) + 1
        
        # 1. 두 모듈 모두 전체 채널을 처리하도록 생성
        # 하드 라우팅보다 메모리는 더 쓰지만, 학습의 안정성은 비약적으로 상승합니다.
        self.evt_agg = EVTAggregator(configs, self.enc_in, configs.d_model, self.patch_num, 
                                     t_high_np, t_low_np)
        self.tst_agg = PatchTSTAggregator(self.enc_in, configs.d_model, self.patch_num)
        
        # 2. 통계 기반 게이트 파라미터
        self.register_buffer('kurtosis', torch.from_numpy(kurtosis_np).float()) # (C,)
        self.gate_gain = nn.Parameter(torch.ones(self.enc_in)) # 학습을 통해 결정됨
        
    def forward(self, patch_latent, raw_patches):
        B, C, N, D = patch_latent.shape
        
        # 각 모듈의 결과 계산 (B, C, N+1, D)
        z_evt = self.evt_agg(patch_latent, raw_patches)
        z_tst = self.tst_agg(patch_latent)

        # 3. 소프트 게이트 계산
        # kurtosis가 3.0보다 크면 gate 값이 0.5 이상으로 올라감
        gate = torch.sigmoid(self.gate_gain * (self.kurtosis - 3.0))
        gate = gate.view(1, C, 1, 1) # 브로드캐스팅을 위한 차원 확장

        # 4. 최종 융합 (Weighted Sum)
        # 이 과정에서 두 모듈의 그래디언트가 첨도 비율에 맞게 섞여 흐릅니다.
        final_output = (z_evt * gate) + (z_tst * (1 - gate))
        
        return final_output