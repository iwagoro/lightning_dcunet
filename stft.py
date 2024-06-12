import torch
def stft(wav, n_fft=64, hop_length=16):
    window = torch.hann_window(n_fft,device=wav.device)
    stft = torch.stft(input=wav, n_fft=n_fft,hop_length=hop_length, normalized=True,return_complex=True,window=window)
    stft = torch.view_as_real(stft)
    return stft

def istft(stft, n_fft=64, hop_length=16):
    window = torch.hann_window(n_fft,device=stft.device)
    stft = torch.view_as_complex(stft)
    stft = torch.squeeze(stft, 1)
    wav = torch.istft(stft, n_fft=n_fft, hop_length=hop_length, normalized=True, window=window)
    return wav