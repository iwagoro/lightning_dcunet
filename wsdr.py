import torch

def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    
    SAMPLE_RATE = 48000
    N_FFT = SAMPLE_RATE * 64 // 1000 + 4
    HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    x_ = torch.squeeze(x_, 1)
    
    window= torch.hann_window(N_FFT,device=x_.device)
    y_true_ = torch.view_as_complex(y_true_)
    y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
    x_ = torch.view_as_complex(x_)

    x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)