import argparse
from pathlib import Path
from dataset import SpeechDataset 
from network.dcunet import DCUnet10
from network.dcunet_rtstm import DCUnet10_rTSTM
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import os

from dotenv import load_dotenv

load_dotenv()

mp.set_start_method('spawn', force=True)
torch.set_float32_matmul_precision('high')

def main(args):
    TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
    TEST_BATCH_SIZE = int(os.getenv('TEST_BATCH_SIZE'))

    noisy_train_dir = ""
    noisy_test_dir = ""

    if args.dataset == 'white':
        noisy_train_dir = os.getenv("WHITE_NOISY_TRAIN")
        noisy_test_dir = os.getenv("WHITE_NOISY_TEST")
    elif args.dataset == 'urban0':
        noisy_train_dir = os.getenv("URBAN0_NOISY_TRAIN")
        noisy_test_dir = os.getenv("URBAN0_NOISY_TEST")
    elif args.dataset == 'urban1':
        noisy_train_dir = os.getenv("URBAN1_NOISY_TRAIN")
        noisy_test_dir = os.getenv("URBAN1_NOISY_TEST")
    elif args.dataset == 'urban2':
        noisy_train_dir = os.getenv("URBAN2_NOISY_TRAIN")
        noisy_test_dir = os.getenv("URBAN2_NOISY_TEST")
    else:
        raise ValueError("Invalid dataset. Choose from 'white', 'urban0', 'urban1', 'urban2'.")
    noisy_train_dir = Path(noisy_train_dir)
    noisy_test_dir = Path(noisy_test_dir)
    clean_train_dir = Path("./dataset/clean_trainset_28spk_wav/")
    clean_test_dir = Path("./dataset/clean_testset_wav/")

    train_noisy_files = sorted(list(noisy_train_dir.rglob('*.wav')))
    train_clean_files = sorted(list(clean_train_dir.rglob('*.wav')))
    test_noisy_files = sorted(list(noisy_test_dir.rglob('*.wav')))
    test_clean_files = sorted(list(clean_test_dir.rglob('*.wav')))
    

    trainset = SpeechDataset(train_noisy_files, train_clean_files)
    testset = SpeechDataset(test_noisy_files, test_clean_files)
    train_loader = DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8,persistent_workers=True)
    test_loader = DataLoader(testset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8,persistent_workers=True)

    # Update checkpoint and logger paths with model and dataset names
    checkpoint_dir = f'./checkpoints/{args.model}-{args.dataset}'
    tb_log_dir = f'tb_logs/{args.model}-{args.dataset}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='model-{epoch:02d}-{step:04d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True
    )

    logger = TensorBoardLogger(tb_log_dir, name="my_model")
    strategy = DDPStrategy(find_unused_parameters=True)
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

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    if args.mode == 'train':
        
        if args.loss != "nct" and args.loss != "nb2nb":
            raise ValueError("Invalid loss type. Choose from 'nct' or 'nb2nb'")

        if args.model == 'dcunet':
            model = DCUnet10(loss_type=args.loss)
        elif args.model == 'dcunet-rtstm':
            model = DCUnet10_rTSTM(loss_type=args.loss)
        else:
            raise ValueError("Invalid model. Choose from 'dcunet' or 'dcunet-rtstm'.")
        trainer = Trainer(
            accelerator="gpu",
            # callbacks=[checkpoint_callback, progress_bar, early_stopping_callback],
            callbacks=[checkpoint_callback, progress_bar],
            logger=logger,
            max_epochs=150,
            strategy=strategy,
            devices=args.devices
        )
        trainer.fit(model, train_loader, test_loader)
    elif args.mode == 'predict':
        checkpoint = args.checkpoint
        if args.model == 'dcunet':
            pred_model = DCUnet10.load_from_checkpoint(checkpoint,dataset=args.dataset)
        elif args.model == 'dcunet-rtstm':
            pred_model = DCUnet10_rTSTM.load_from_checkpoint(checkpoint,dataset=args.dataset)

        pred_noisy_files = sorted(list(noisy_test_dir.rglob('*.wav'))[:TEST_BATCH_SIZE])
        pred_clean_files = sorted(list(clean_test_dir.rglob('*.wav'))[:TEST_BATCH_SIZE])

        testset = SpeechDataset(pred_noisy_files, pred_clean_files)
        pred_loader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8,persistent_workers=True)

        trainer = Trainer(
            accelerator="gpu",
            callbacks=[progress_bar],
            devices=args.devices
        )
        trainer.predict(pred_model, dataloaders=pred_loader)
    else:
        raise ValueError("Invalid mode. Choose from 'train' or 'predict'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or predict with DCUnet models.")
    parser.add_argument('--mode', type=str, required=True, help="Mode: 'train' or 'predict'")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset: 'white', 'urban0', 'urban1', 'urban2'")
    parser.add_argument('--loss',type=str, required=False,help="Loss function: 'nct' or 'nb2nb'")
    parser.add_argument('--model', type=str, required=True, help="Model: 'dcunet' or 'dcunet-rtstm'")
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help="List of GPU devices to use")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint file for prediction")

    args = parser.parse_args()
    main(args)
