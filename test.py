from stft_loss import stft_loss,mse_loss,reg_loss,wsdr_fn,basic_loss
from stft import stft,istft
from dataset3 import SpeechDataset
import torch

from pathlib import Path
import torchaudio
N_FFT = 1022
HOP_LENGTH = 256


noisy_train_dir = Path("./dataset/white/WhiteNoise_Train_Input/")
noisy_test_dir = Path("./dataset/white/WhiteNoise_Test_Input/")

clean_train_dir = Path("./dataset/clean_trainset_28spk_wav/")
clean_test_dir = Path("./dataset/clean_testset_wav/")

train_noisy_files = sorted(list(noisy_train_dir.rglob('*.wav'))[:16])
train_clean_files = sorted(list(clean_train_dir.rglob('*.wav'))[:16])
test_noisy_files = sorted(list(noisy_test_dir.rglob('*.wav'))[:16])
test_clean_files = sorted(list(clean_test_dir.rglob('*.wav'))[:16])

trainset = SpeechDataset(train_noisy_files, train_clean_files, N_FFT, HOP_LENGTH)
testset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8,persistent_workers=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8,persistent_workers=True)

def subsample2(wav):
    wav1 = []
    wav2 = []
    for i in range(len(wav)):
        channel, length = wav[i].shape
        new_length = length // 2 - 128

        # ランダムなインデックスを生成
        indices = (torch.arange(new_length) * 2 - 127).unsqueeze(0).repeat(channel, 1)

        # インデックスの境界を処理
        indices = indices.clamp(0, length - 2)

        random_choice = torch.randint(0, 2, (channel, new_length))

        # ランダムに選択されたインデックスを作成
        index1 = indices + random_choice
        index2 = indices + (1 - random_choice)

        # 新しいテンソルを作成
        wav1.append( torch.gather(wav[i], 1, index1))
        wav2.append( torch.gather(wav[i], 1, index2))
    wav1 = torch.cat(wav1,dim=0).unsqueeze(1)
    wav2 = torch.cat(wav2,dim=0).unsqueeze(1)
    return wav1, wav2

def tensor_stft(wav,n_fft,hop_length):
    result = []
    
    for i in range(len(wav)):
        wav_stft = stft(wav[i],n_fft,hop_length)
        result.append(wav_stft)
    
    result = torch.cat(result,dim=0).unsqueeze(1)
    return result


def tensor_istft(stft,n_fft,hop_length):
    result = []
    
    for i in range(len(stft)):
        wav = istft(stft[i],n_fft,hop_length)
        result.append(wav)
    
    result = torch.cat(result,dim=0).unsqueeze(1)
    return result
    


for noisy , clean in train_loader:
    # # print(noisy.shape)
    g1,g2 = subsample2(clean)
    fg1,fg2 = subsample2(noisy)
    # g1_stft = tensor_stft(g1,N_FFT,HOP_LENGTH)
    # print(g1_stft.shape)
    loss = basic_loss(g1,g2,fg,N_FFT,HOP_LENGTH)
    print(loss)
    # wav = tensor_istft(noisy_stft,N_FFT,HOP_LENGTH)
    # g1,g2=subsample2(noisy)
    # fg1,fg2 = subsample2(clean)
    # loss = mse_loss(g1,g2)
    # print(loss)
    