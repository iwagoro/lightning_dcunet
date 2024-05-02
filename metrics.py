import torch
import torchaudio
from pypesq import pesq
from rich.progress import Progress
from torchmetrics.audio import SignalNoiseRatio
from stft import istft
def getPesqList(cleans,preds,N_FFT,HOP_LENGTH):
    psq = [];
    with Progress(transient=True) as progress:

        task = progress.add_task("[#ff1493]Processing PESQ...", total=min(len(cleans),len(preds)))
        for i in range(min(len(cleans),len(preds))):
            progress.update(task, advance=1)
            pred = istft(preds[i],N_FFT,HOP_LENGTH)
            clean = istft(cleans[i],N_FFT,HOP_LENGTH)

            clean = torchaudio.transforms.Resample(48000, 16000)(clean.detach().cpu())
            pred = torchaudio.transforms.Resample(48000, 16000)(pred.detach().cpu())

            psq.append(pesq(pred.flatten(),clean.flatten()))
            
    progress.console.print(f"[#ff1493]Completed. Total PESQ = {psq}")
    return psq

def getSNRList(cleans,preds,N_FFT,HOP_LENGTH):
    snr = [];
    with Progress(transient=True) as progress:

        task = progress.add_task("[#ff1493]Processing", total=min(len(cleans),len(preds)))
        for i in range(min(len(cleans),len(preds))):
            progress.update(task, advance=1)
            pred = istft(preds[i],N_FFT,HOP_LENGTH)
            clean = istft(cleans[i],N_FFT,HOP_LENGTH)

            clean = torchaudio.transforms.Resample(48000, 16000)(clean.detach().cpu())
            pred = torchaudio.transforms.Resample(48000, 16000)(pred.detach().cpu())

            snr.append(SignalNoiseRatio()(pred.flatten(),clean.flatten()))
            
    progress.console.print(f"[#ff1493]Completed")
    return snr