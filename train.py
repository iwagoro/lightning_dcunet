from pathlib import Path
from dataset2 import SpeechDataset
import torch
from network.dcunet import DCUnet10
from network.dcunet_rtstm import DCUnet10_rTSTM
from torchinfo import summary
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar,ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import numpy as np


################################################################################################################

torch.set_float32_matmul_precision('high')
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

# noisy_train_dir = Path("./dataset/noisy_trainset_28spk_wav/")
# noisy_train_dir = Path("/workspace/app/Only-Noisy-Training/Datasets/WhiteNoise_Train_Input/")
noisy_train_dir = Path("./dataset/urbansound/US_Class0_Train_Input/")
clean_train_dir = Path("./dataset/clean_trainset_28spk_wav/")

# noisy_test_dir = Path("./dataset/noisy_testset_wav/")
# noisy_test_dir = Path("/workspace/app/Only-Noisy-Training/Datasets/WhiteNoise_Test_Input/")
noisy_test_dir = Path("./dataset/urbansound/US_Class0_Test_Input/")
clean_test_dir = Path("./dataset/clean_testset_wav/")
train_noisy_files = sorted(list(noisy_train_dir.rglob('*.wav')))
train_clean_files = sorted(list(clean_train_dir.rglob('*.wav')))
test_noisy_files = sorted(list(noisy_test_dir.rglob('*.wav')))
test_clean_files = sorted(list(clean_test_dir.rglob('*.wav')))

trainset = SpeechDataset(train_noisy_files,train_clean_files,N_FFT,HOP_LENGTH)
testset = SpeechDataset(test_noisy_files,test_clean_files,N_FFT,HOP_LENGTH)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True,num_workers=112)
test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False,num_workers=112)

################################################################################################################

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',           # 監視するメトリクス
    dirpath='./checkpoints',      # チェックポイントの保存ディレクトリ
    filename='model-{epoch:02d}-{step:04d}-{val_loss:.2f}',  # ファイル名のフォーマット
    save_top_k=1,
    verbose=True
)


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

model =DCUnet10(n_fft=N_FFT,hop_length=HOP_LENGTH)
# model =DCUnet10_rTSTM(n_fft=N_FFT,hop_length=HOP_LENGTH)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # 監視するメトリクス
    patience=10,          # 改善が見られないエポック数
    verbose=True,        # メッセージの出力
    mode='min'           # 最小化を目指す（val_lossが減少する方向）
)



# trainer = Trainer(
#     accelerator="gpu",
#     callbacks=[checkpoint_callback,progress_bar,early_stopping_callback],
#     logger=logger,
#     max_epochs=1000,
#     strategy=strategy,
#     devices=[0,1,2,3,4,5,6,7]
#     # devices=[0]
# )

# trainer.fit(model,train_loader,test_loader)


checkpoint = "/workspace/app/ont/checkpoints/NONE-SCHEDULER/normal_urban0_model-epoch=111-step=2576-val_loss=-0.97.ckpt"
# pred_model = DCUnet10_rTSTM.load_from_checkpoint(checkpoint,n_fft=N_FFT,hop_length=HOP_LENGTH)
pred_model = DCUnet10.load_from_checkpoint(checkpoint,n_fft=N_FFT,hop_length=HOP_LENGTH)
pred_noisy_files = sorted(list(noisy_test_dir.rglob('*.wav'))[0:16])
pred_clean_files = sorted(list(clean_test_dir.rglob('*.wav'))[0:16])


testset = SpeechDataset(pred_noisy_files,pred_clean_files,N_FFT,HOP_LENGTH)
pred_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False,num_workers=111)

trainer = Trainer(
    accelerator="gpu",
    callbacks=[progress_bar],
    devices=[0]
)

# Make predictions on the test dataset
predictions = trainer.predict(pred_model, dataloaders=pred_loader)


