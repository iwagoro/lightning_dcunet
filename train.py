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
import numpy as np


################################################################################################################

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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True,num_workers=16)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False,num_workers=16)

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
# model = DCUnet10(n_fft=N_FFT,hop_length=HOP_LENGTH)
# trainer = Trainer(
#     accelerator="gpu",
#     callbacks=[EarlyStopping(monitor="val_loss", mode="min"),progress_bar],
#     logger=logger,
#     max_epochs=1,
#     strategy=strategy,
#     devices=[0,1,2,3,4,5,6,7],
# )

# trainer.fit(model, train_loader,test_loader)
# trainer.predict(model,test_loader)


checkpoint = "./tb_logs/my_model/version_1/checkpoints/epoch=0-step=46.ckpt"
pred_model = DCUnet10.load_from_checkpoint(checkpoint)
# pred_model.eval()

trainer = Trainer(
    accelerator="gpu",
    callbacks=[progress_bar],
    devices=[0]
)

# Make predictions on the test dataset
predictions = trainer.predict(pred_model, dataloaders=test_loader)