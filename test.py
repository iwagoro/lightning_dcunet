from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from dataset import SpeechDataset
import torch
from DCUNet import DCUnet10
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)

model = DCUnet10.load_from_checkpoint(
    checkpoint_path="/path/to/pytorch_checkpoint.ckpt",
    hparams_file="/path/to/experiment/version/hparams.yaml",
    map_location=None,
)
