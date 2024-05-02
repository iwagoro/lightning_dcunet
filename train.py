from pathlib import Path
from dataset import SpeechDataset
import torch
from dcunet import DCUnet10
from torchinfo import summary
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


################################################################################################################

torch.set_float32_matmul_precision('high')
SAMPLE_RATE = 48000
N_FFT = SAMPLE_RATE * 64 // 1000 + 4
HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4

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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False,num_workers=2)

################################################################################################################

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
model = DCUnet10(n_fft=N_FFT,hop_length=HOP_LENGTH)
trainer = Trainer(
    accelerator="gpu",
    callbacks=[progress_bar],
    logger=logger,
    max_epochs=20,
    strategy=strategy,
    devices=2,
)
trainer.fit(model, train_loader)  
# print(testset[0][1].shape)

# summary(DCUnet10(N_FFT,HOP_LENGTH),(32,1,1539,214,2))