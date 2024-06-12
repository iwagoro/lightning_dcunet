import torch
import torchaudio
from pypesq import pesq
from rich.progress import Progress
from torchmetrics.audio import SignalNoiseRatio
from stft import istft
def getPesqList(cleans,preds,N_FFT,HOP_LENGTH):
    psq = [];
    for i in range(min(len(cleans),len(preds))):
        pred = istft(preds[i],N_FFT,HOP_LENGTH)
        clean = istft(cleans[i],N_FFT,HOP_LENGTH)
        clean = torchaudio.transforms.Resample(48000, 16000)(clean.detach().cpu())
        pred = torchaudio.transforms.Resample(48000, 16000)(pred.detach().cpu())
        psq.append(pesq(pred.flatten(),clean.flatten()))
    return sum(psq)

def getSNRList(cleans,preds,N_FFT,HOP_LENGTH):
    snr = [];
    for i in range(min(len(cleans),len(preds))):
        pred = istft(preds[i],N_FFT,HOP_LENGTH)
        clean = istft(cleans[i],N_FFT,HOP_LENGTH)
        clean = torchaudio.transforms.Resample(48000, 16000)(clean.detach().cpu())
        pred = torchaudio.transforms.Resample(48000, 16000)(pred.detach().cpu())
        snr.append(SignalNoiseRatio()(pred.flatten(),clean.flatten()))
    return sum(snr)