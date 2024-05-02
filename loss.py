# import torch
# import torch.nn as nn


# def basic_loss(g1_wav,fg1_wav,g2_wav): # g1=noisy fg1=pred g2=clean
    
#     # SAMPLE_RATE = 48000
#     # N_FFT = SAMPLE_RATE * 64 // 1000 + 4
#     # HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4
#     # pred_stft = torch.squeeze(pred_raw, 1)
#     # noisy_stft = torch.squeeze(noisy_raw, 1)
#     # window= torch.hann_window(N_FFT,device=pred_stft.device)
#     # pred = torch.view_as_complex(pred_stft)
#     # pred_raw = torch.istft(pred, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#     # noisy = torch.view_as_complex(noisy_stft)
#     # noisy_raw = torch.istft(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#     time_loss = mse_loss(fg1_wav,g2_wav)
#     # freq_loss = time_frequency_loss(fg1_wav,g2_wav)
    
#     return wsdr_loss(g1_wav,fg1_wav,g2_wav) + time_loss
    


# def regloss(self, g1, g2, G1, G2):
#         return torch.mean((g1-g2-G1+G2)**2)

# def mse_loss(pred,truth):
#     return nn.MSELoss()(pred,truth)

# def time_frequency_loss(pred,true):
#     # truth_real, truth_imag, pred_real, pred_imag はすべて同じ形状のTensorである必要があります
#     # これらのTensorは [バッチサイズ, フレーム数T, 周波数ビン数F] の形状をしていると仮定します
    
#     # 真の信号と予測信号の実部と虚部の絶対値を計算
#     abs_truth = torch.abs(truth_real) + torch.abs(truth_imag)
#     abs_pred = torch.abs(pred_real) + torch.abs(pred_imag)
    
#     # 損失の計算
#     loss = torch.abs(abs_truth - abs_pred)  # 各要素の差の絶対値を取る
#     loss = torch.mean(loss)  # バッチ全体の平均を計算
    
#     return loss

# def wsdr_loss(x_, y_pred, y_true, eps=1e-8):
    

#     y_pred = y_pred.flatten(1)
#     y_true = y_true.flatten(1)
#     x = x.flatten(1)


#     def sdr_fn(true, pred, eps=1e-8):
#         num = torch.sum(true * pred, dim=1)
#         den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
#         return -(num / (den + eps))

#     # true and estimated noise
#     z_true = x - y_true
#     z_pred = x - y_pred

#     a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
#     wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
#     return torch.mean(wSDR)