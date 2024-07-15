import torchaudio
import torch
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    
    def __init__(self, noisy_files, clean_files):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        
        # len
        self.len = len(self.noisy_files)
        self.max_len = 65280

    
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
        
        
        return x_noisy,x_clean
        
        
        
        
    def _prepare_sample(self, waveform):
        # Assume waveform is of shape (channels, time)
        channels, current_len = waveform.shape
        
        # Initialize output tensor with zeros
        output = torch.zeros((channels, self.max_len), dtype=torch.float32, device=waveform.device)
        
        # Copy the necessary part of the data
        output[:, -min(current_len, self.max_len):] = waveform[:, :min(current_len, self.max_len)]
    
        return output
    