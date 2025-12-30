import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
import json
import pandas as pd
import numpy as np

def load_evt_thresholds(args):
    """
    JSON 파일에서 [t_high, t_low, kurtosis]를 로드합니다.
    컬럼명을 기준으로 데이터를 정렬하여 feature mismatch를 방지합니다.
    """
    json_path = '/data/tspaper/threshold_5(1).json' 
    print(f"Loading EVT thresholds from {json_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Threshold file not found at {json_path}.")
    
    with open(json_path, 'r') as f:
        all_thresholds = json.load(f)
        
    dataset_name = args.data_path.split('.')[0]
    if dataset_name not in all_thresholds:
        raise KeyError(f"Thresholds for dataset '{dataset_name}' not found.")
    
    data_stats = all_thresholds[dataset_name]
    
    # JSON에 저장된 정보들
    json_cols = data_stats['columns']
    json_t_high = np.array(data_stats['t_high'])
    json_t_low = np.array(data_stats['t_low'])
    json_kurt = np.array(data_stats['kurtosis'])

    # 실제 데이터의 컬럼 순서 가져오기
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    actual_cols = list(df_raw.columns[1:]) # 'date' 제외

    if args.features == 'S':
        # --- 단변량(S) 모드: Target 컬럼 1개만 선택 ---
        try:
            target_idx = json_cols.index(args.target)
            t_high = np.array([json_t_high[target_idx]])
            t_low = np.array([json_t_low[target_idx]])
            kurt = np.array([json_kurt[target_idx]])
            print(f"Loaded 'S' mode: target='{args.target}' (Kurtosis: {kurt[0]:.2f})")
        except ValueError:
            raise ValueError(f"Target '{args.target}' not found in JSON columns.")
            
    else:
        # --- 다변량(M, MS) 모드: 실제 데이터 컬럼 순서에 맞게 Re-ordering ---
        t_high_list, t_low_list, kurt_list = [], [], []
        
        for col in actual_cols:
            if col in json_cols:
                idx = json_cols.index(col)
                t_high_list.append(json_t_high[idx])
                t_low_list.append(json_t_low[idx])
                kurt_list.append(json_kurt[idx])
            else:
                # JSON에 없는 컬럼이 데이터에 있을 경우 에러 처리 (연구 객관성 유지)
                raise KeyError(f"Column '{col}' in CSV not found in JSON stats.")
        
        t_high = np.array(t_high_list)
        t_low = np.array(t_low_list)
        kurt = np.array(kurt_list)
        print(f"Loaded '{args.features}' mode: {len(t_high)} channels re-ordered.")

    # 최종 채널 수 검증
    if len(t_high) != args.enc_in:
        raise ValueError(f"Mismatch! args.enc_in({args.enc_in}) != loaded channels({len(t_high)})")
    
    return t_high, t_low, kurt

def set_seed(seed_value=42):
    """실험 재현을 위해 시드를 고정하는 함수"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [PP, Autoformer, FEDformer, iTransformer, PatchTST, RLinearCI]')
    


    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/data/tspaper/dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # Mamba
    parser.add_argument('--d_state', type=int, default=16)

    # PP
    parser.add_argument('--stride', type=int, default=24, help='stride size for patching')
    parser.add_argument('--patch_len', type=int, default=24, help='patch length')
    parser.add_argument('--modes', type=int, default=8, help='the number of modes used in Spectral Filter')
    parser.add_argument('--top_k', type=int, default=8, help='to select the top-k dominant frequencies from the low-frequency components')
    parser.add_argument('--kd_lambda', type=float, default=0.1, help='the weight of KD loss')
    parser.add_argument('--kd_hidden_dim', type=int, default=512, help='the dimension of the hidden space for feature projection in KD')
    parser.add_argument('--use_teacher', action='store_true', help='whether to use the teacher model (Spectral Filter) in training', default=True)
    parser.add_argument('--rank', type=int, default=32, help='the rank of the low-rank MLP in Local Branch')
    parser.add_argument('--fuse_mode', type=str, default='kd_loss', help='the way how to fuse freq branch and patch branch (for ablation)')
    


    # 1. 반복 실험을 위한 시드 리스트 정의
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='list of seeds for multiple runs')

    args = parser.parse_args()

    t_high, t_low, kurt = load_evt_thresholds(args)
    args.t_high = t_high # (C,) shape의 numpy array
    args.t_low = t_low   # (C,) shape의 numpy array
    args.kurt = kurt
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        all_mses, all_maes = [], [] 
        for i, seed in enumerate(args.seeds):
            print(f"\n>>>>>>> Starting Run {i+1}/{len(args.seeds)} with Seed: {seed} <<<<<<<")
            
            # 4. 루프 안에서 매번 새로운 시드 설정
            set_seed(seed)

            # 5. setting 이름에 시드를 포함시켜 결과 파일이 덮어쓰이는 것을 방지
            setting = '{}_{}_ft{}_pl{}_el{}_kd{}_modes{}'.format(
                args.model,
                args.data,
                args.features,
                args.pred_len,
                args.e_layers,
                args.kd_lambda,
                args.modes) # 'ii' 대신 'seed'를 사용
            
            exp = Exp(args)  # 실험 객체 생성
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1) # test=1로 best model 로드

            # 6. 실험 후 결과 파일(metrics.npy)을 불러와서 저장
            # exp.test는 './results/{setting}/metrics.npy'에 결과를 저장합니다.
            results_path = f'./results/{setting}/metrics.npy'
            if os.path.exists(results_path):
                metrics = np.load(results_path)
                mae, mse = metrics[0], metrics[1]
                print(f"Run {i+1} Results - MAE: {mae:.4f}, MSE: {mse:.4f}")
                all_maes.append(mae)
                all_mses.append(mse)
            else:
                print(f"Warning: Results file not found at {results_path}")

            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
            elif args.gpu_type == 'mps':
                # torch.backends.mps.empty_cache() # 최신 PyTorch 버전에 따라 이 함수가 없을 수 있습니다.
                pass
        
        # 7. 최종 평균 및 표준편차 계산 후 출력
        if all_maes and all_mses:
            mean_mae = np.mean(all_maes)
            std_mae = np.std(all_maes)
            mean_mse = np.mean(all_mses)
            std_mse = np.std(all_mses)

            print("\n\n" + "="*50)
            print("Final Summary over all runs")
            print(f"Total runs: {len(args.seeds)}")
            print(f"MSE: {mean_mse:.4f} ± {std_mse:.4f}")
            print(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            print("="*50 + "\n")
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_ft{}_pl{}_el{}_kd{}_modes{}'.format(
                args.model,
                args.data,
                args.features,
                args.pred_len,
                args.e_layers,
                args.kd_lambda,
                args.modes)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()