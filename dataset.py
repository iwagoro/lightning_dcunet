import torchaudio
import torch
from torch.utils.data import Dataset
import numpy as np
from stft import stft, istft

def subsample2(wav):
    channel,length = wav.shape
    length = length  // 2 - 128
    wav1,wav2 = torch.zeros([channel,length]),torch.zeros([channel,length])
    for channel in range(channel):
        for i in range(length):
            random = np.random.choice([0,1])
            index = i * 2 - 127
            if random == 0:
                wav1[channel, i], wav2[channel, i] = wav[channel, index], wav[channel, index+1]
            elif random == 1:
                wav1[channel, i], wav2[channel, i] = wav[channel, index+1], wav[channel, index]
    
    return wav1,wav2

class SpeechDataset(Dataset):
    
    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        
        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # len
        self.len = len(self.noisy_files)
        self.max_len = 165000

    
    def __len__(self):
        return self.len
      
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform
  
    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])
        
        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        
        
        g1,g2 = subsample2(x_noisy)
        
        # Short-time Fourier transform
        # x_noisy_stft = stft(x_noisy, self.n_fft, self.hop_length)
        # x_clean_stft = stft(x_clean, self.n_fft, self.hop_length)
        g1_stft = stft(g1,self.n_fft,self.hop_length)
        g2_stft = stft(g2,self.n_fft,self.hop_length)
        
        
        return g1_stft,g2_stft
        # return x_noisy_stft,x_clean_stft
        
    def _prepare_sample(self, waveform):
        current_len = waveform.shape[1]  # オーディオデータの現在の長さを取得
    
        # 出力テンソルをゼロで初期化
        output = torch.zeros((1, self.max_len), dtype=torch.float32, device=waveform.device)
        # 必要な部分のデータをコピー
        output[0, -min(current_len, self.max_len):] = waveform[0, :min(current_len, self.max_len)]
    
        return output

    