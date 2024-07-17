import torchaudio
import torch
from torch.utils.data import Dataset
from stft import stft, istft

SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

class SpeechDataset(Dataset):
    
    def __init__(self, noisy_files, clean_files, n_fft=N_FFT, hop_length=HOP_LENGTH):
        super().__init__()
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.len = len(self.noisy_files)
        self.max_len = 65280

    def __len__(self):
        return self.len
      
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
  
    def __getitem__(self, index):
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])
        
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        
        x_noisy_stft = stft(x_noisy, self.n_fft, self.hop_length)
        x_clean_stft = stft(x_clean, self.n_fft, self.hop_length)
        
        return x_noisy_stft, x_clean_stft
        
    def _prepare_sample(self, waveform):
        channels, current_len = waveform.shape
        output = torch.zeros((channels, self.max_len), dtype=torch.float32, device=waveform.device)
        output[:, -min(current_len, self.max_len):] = waveform[:, :min(current_len, self.max_len)]
        return output
