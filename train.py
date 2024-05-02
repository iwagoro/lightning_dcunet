from pathlib import Path
from dataset import SpeechDataset
import torch
from DCUNet import DCUnet10,Encoder,Decoder
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pypesq import pesq 
from torchinfo import summary
import torchaudio
from rich.progress import Progress
from rich.live import Live
from rich.console import Console, Group
from rich.panel import Panel

from torchmetrics.audio import SignalNoiseRatio
from rich.progress import track


torch.set_float32_matmul_precision('high')
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

noisy_train_dir = Path("./dataset/noisy_trainset_28spk_wav/")
clean_train_dir = Path("./dataset/clean_trainset_28spk_wav/")
noisy_test_dir = Path("./dataset/noisy_testset_wav/")
clean_test_dir = Path("./dataset/clean_testset_wav/")

train_noisy_files = sorted(list(noisy_train_dir.rglob('*.wav')))
train_clean_files = sorted(list(clean_train_dir.rglob('*.wav')))
test_noisy_files = sorted(list(noisy_test_dir.rglob('*.wav')))
test_clean_files = sorted(list(clean_test_dir.rglob('*.wav')))

trainset = SpeechDataset(train_noisy_files,train_clean_files,N_FFT,HOP_LENGTH)
testset = SpeechDataset(test_noisy_files,test_clean_files,N_FFT,HOP_LENGTH)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,num_workers=27)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False,num_workers=27)


logger = TensorBoardLogger("tb_logs", name="my_model")
strategy=DDPStrategy(find_unused_parameters=True)
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green_yellow",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="cyan",
        processing_speed="#ff1493",
        metrics="#ff1493",
        metrics_text_delimiter="\n",
    )
)
model = DCUnet10()
trainer = Trainer(
    accelerator="cuda",
    callbacks=[progress_bar],#EarlyStopping(monitor="val_loss", mode="min")
    default_root_dir="./",
    logger=logger,
    max_epochs=20,
    strategy=strategy,
    devices=2,
    # check_val_every_n_epoch=1  # 毎エポックで検証
)
# print(trainset[0][0].shape,trainset[0][1].shape,trainset[0][2].shape)
# summary(DCUnet10(),input_size=(32,1,523,322,2))
# summary(Encoder(),input_size=(32,1,1539,107,2))
trainer.fit(model, train_loader)  
# model = DCUnet10.load_from_checkpoint("./tb_logs/my_model/version_7/checkpoints/epoch=19-step=460.ckpt")

# model.eval()

# psq = 0;
# snr = 0;
# with Progress(transient=True) as progress:
    
#     task = progress.add_task("[#ff1493]Processing SNR...", total=len(testset))
#     for i in range(len(testset)):
#         progress.update(task, advance=1)
#         y_hat = model(testset[i][0].unsqueeze(0).cuda())
#         x = torch.view_as_complex(testset[i][0])
#         window = torch.hann_window(N_FFT, device=x.device)
#         x = torch.istft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#         y = torch.view_as_complex(testset[i][1])
#         y = torch.istft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)

#         noisy = torchaudio.transforms.Resample(48000, 16000)(x.detach().cpu())
#         pred= torchaudio.transforms.Resample(48000, 16000)(y_hat.detach().cpu())

#         snr += SignalNoiseRatio()(pred.flatten(),noisy.flatten())
        

#     snr /= len(testset)
#     progress.console.print(f"[#ff1493]Completed SNR = {snr}")
#     task = progress.add_task("[#ff1493]Processing PESQ...", total=len(testset))
#     for i in range(len(testset)):
#         progress.update(task, advance=1)
#         y_hat = model(testset[i][0].unsqueeze(0).cuda())
#         x = torch.view_as_complex(testset[i][0])
#         window = torch.hann_window(N_FFT, device=x.device)
#         x = torch.istft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#         y = torch.view_as_complex(testset[i][1])
#         y = torch.istft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)

#         clean = torchaudio.transforms.Resample(48000, 16000)(y.detach().cpu())
#         pred= torchaudio.transforms.Resample(48000, 16000)(y_hat.detach().cpu())

#         psq += pesq(clean.flatten(), pred.flatten(), 16000)
        
#     psq /= len(testset)
#     progress.console.print(f"[#ff1493]Completed PESQ = {psq}")
    
#     task = progress.add_task("[#ff1493]Processing export wav...", total=100)
#     for i in range(100):
#         progress.update(task, advance=1)
#         y_hat = model(testset[i][0].unsqueeze(0).cuda())
#         x = torch.view_as_complex(testset[i][0])
#         window = torch.hann_window(N_FFT, device=x.device)
#         x = torch.istft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#         y = torch.view_as_complex(testset[i][1])
#         y = torch.istft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#         # torchaudio.save(uri='noisy/noisy'+str(i)+'.wav', src=x, sample_rate=SAMPLE_RATE)
#         torchaudio.save(uri='clean/clean'+str(i)+'.wav', src=y, sample_rate=SAMPLE_RATE)
#         torchaudio.save(uri='pred/pred'+str(i)+'.wav', src=y_hat.to('cpu'), sample_rate=SAMPLE_RATE)
