from pathlib import Path

from dataset import SpeechDataset
import torch
from DCUNet import DCUnet10
import torchaudio
from tqdm import tqdm
# `pytorch_lightning` を使用していた部分を `lightning.pytorch` に変更
# `pytorch_lightning` を使用していた部分を `lightning.pytorch` に変更
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
from torchinfo import summary
from lightning.pytorch.strategies import DDPStrategy

torch.set_float32_matmul_precision('medium')
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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False,num_workers=4)


logger = TensorBoardLogger("tb_logs", name="my_model")
strategy=DDPStrategy(find_unused_parameters=True)
model = DCUnet10()
trainer = Trainer(
    accelerator="cuda",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"),RichProgressBar()],
    default_root_dir="./",
    logger=logger,
    max_epochs=100,
    strategy=strategy,
    devices=8,
    # check_val_every_n_epoch=1  # 毎エポックで検証
)
# print(testset[0][0].shape)
# summary(DCUnet10(),input_size=(32,1,1539,214,2))
trainer.fit(model, train_loader,test_loader)  
# model = DCUnet10.load_from_checkpoint("./tb_logs/my_model/version_3/checkpoints/epoch=6-step=161.ckpt")

# model.eval()

# for i in tqdm(range(10)):
#     y_hat = model(testset[i][0].unsqueeze(0))
#     x = torch.view_as_complex(testset[i][0])
#     window = torch.hann_window(N_FFT, device=x.device)
#     x = torch.istft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
#     y = torch.view_as_complex(testset[i][1])
#     y = torch.istft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)


#     torchaudio.save(uri='noisy/noisy'+str(i)+'.wav', src=x, sample_rate=SAMPLE_RATE)
#     torchaudio.save(uri='clean/clean'+str(i)+'.wav', src=y, sample_rate=SAMPLE_RATE)
#     torchaudio.save(uri='pred/pred'+str(i)+'.wav', src=y_hat.to('cpu'), sample_rate=SAMPLE_RATE)
