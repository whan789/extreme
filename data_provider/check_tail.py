import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import genpareto, kurtosis

def analyze_tail_properties(data_path, target_col='OT', threshold_percentile=95):
    """
    데이터의 Tail Index(Shape Parameter)와 Kurtosis를 분석하여 Heavy-tail 여부를 판별함
    """
    print(f"\n{'='*20} Analyzing {data_path} {'='*20}")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv(data_path)
        data = df[target_col].values
    except Exception as e:
        print(f"[Error] 파일을 찾을 수 없습니다: {data_path}")
        return

    # 2. 기본 통계량 (첨도)
    # 정규분포의 첨도는 3 (Fisher 기준 0). 첨도가 매우 높으면 Heavy-tail 가능성 큼.
    kurt = kurtosis(data, fisher=False) 
    print(f"1. Kurtosis (첨도): {kurt:.4f} (Normal Distribution ≈ 3.0)")
    if kurt > 5.0:
        print("   -> 첨도가 높아 꼬리가 두꺼울 가능성이 있습니다.")
    else:
        print("   -> 첨도가 낮아 정규분포에 가깝거나 꼬리가 얇을 수 있습니다.")

    # 3. EVT - GPD Fitting (POT Method)
    # 임계값 설정 (상위 100 - threshold_percentile %)
    threshold = np.percentile(data, threshold_percentile)
    
    # 임계값 초과분 추출 (Peaks Over Threshold)
    excesses = data[data > threshold] - threshold
    
    if len(excesses) < 20:
        print("   [Warning] 초과 데이터가 너무 적어 GPD 피팅이 부정확할 수 있습니다.")

    # GPD 파라미터 추정 (shape, loc, scale)
    # shape (xi) > 0 : Heavy-tailed (Pareto-like)
    # shape (xi) approx 0 : Light-tailed (Exponential-like)
    # shape (xi) < 0 : Bounded (Short-tailed)
    xi, loc, sigma = genpareto.fit(excesses)
    
    print(f"2. GPD Shape Parameter (xi): {xi:.4f}")
    print(f"   (Threshold: 상위 {100-threshold_percentile}% = {threshold:.4f})")
    
    if xi > 0.05:
        print(f"   => ★ 판정: Heavy-tailed (매우 두꺼운 꼬리)")
        print("      (EVT 기반 모듈이 효과적일 확률이 높음)")
    elif xi > -0.05:
        print(f"   => ★ 판정: Light-tailed (지수 분포 유사)")
        print("      (극단값이 존재하지만 예측 가능한 범위 내일 수 있음)")
    else:
        print(f"   => ★ 판정: Short-tailed (꼬리가 닫혀있음)")
        print("      (극단값이 거의 발생하지 않음)")

    # 4. 시각화 (Q-Q Plot)
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of {target_col}')
    plt.yscale('log') # 로그 스케일로 봐야 꼬리가 보임
    plt.ylabel('Log Density')

    # Tail Q-Q Plot (vs Exponential)
    # 꼬리가 지수분포(직선)보다 위로 휘어지면 Heavy-tail
    plt.subplot(1, 2, 2)
    stats.probplot(excesses, dist="expon", plot=plt)
    plt.title('Q-Q Plot of Excesses (vs Exponential)')
    
    plt.tight_layout()
    plt.show()

# --- 실행 예시 ---
# 경로를 실제 데이터 파일 경로로 수정하세요.
# 예: args.root_path + args.data_path
file_path_etth1 = './dataset/ETT-small/ETTh1.csv'
file_path_etth2 = './dataset/ETT-small/ETTh2.csv'
file_path_ettm1 = './dataset/ETT-small/ETTm1.csv'
file_path_ettm2 = './dataset/ETT-small/ETTm2.csv'
file_path_exchange_rate = './dataset/exchange_rate/exchange_rate.csv'
file_path_weather = './dataset/weather/weather.csv'
file_path_electricity = './dataset/electricity/electricity.csv'

# 분석 실행
analyze_tail_properties(file_path_etth1, target_col='OT')
analyze_tail_properties(file_path_etth2, target_col='OT')
analyze_tail_properties(file_path_ettm1, target_col='OT')
analyze_tail_properties(file_path_ettm2, target_col='OT')
analyze_tail_properties(file_path_exchange_rate, target_col='OT')
analyze_tail_properties(file_path_weather, target_col='OT')
analyze_tail_properties(file_path_electricity, target_col='OT')