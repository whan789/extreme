import torch
import torch.nn as nn
from layers.EVT_aggregator import EVTAggregator
from layers.PatchTST_aggregator import PatchTSTAggregator
import numpy as np
import math

lass Router(nn.Module):
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