import torch
from stft import istft

def wsdr_fn( x, y_pred, y_true,n_fft,hop_length, eps=1e-8):  # g1_wav, fg1_wav, g2_wav
    
        x = istft(x, n_fft, hop_length)
        y_pred = istft(y_pred, n_fft, hop_length)
        y_true = istft(y_true, n_fft, hop_length)
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)
        x = x.flatten(1)

        def sdr_fn(true, pred, eps=1e-8):
            num = torch.sum(true * pred, dim=1)
            den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
            return -(num / (den + eps))

        # true and estimated noise
        z_true = x - y_true
        z_pred = x - y_pred

        a = torch.sum(y_true ** 2, dim=1) / (torch.sum(y_true ** 2, dim=1) + torch.sum(z_true ** 2, dim=1) + eps)
        wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
        return torch.mean(wSDR)