from lightning.pytorch import LightningModule
import torch
from Encoder import Encoder
from Decoder import Decoder 
from loss import wsdr_fn
from metrics import getPesqList,getSNRList
from stft import istft
import torchaudio
import os
class DCUnet10(LightningModule):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        self.pesq_scores = []
        self.snr_scores = []
        self.total_samples = 0
        self.saved = False  # Add a flag to keep track if the audio has been saved

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.save_hyperparameters()
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(3,4), stride_size=(2,2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, output_padding=(1,1), out_channels=1, last_layer=True)
        
    def forward(self, x, is_istft=True):
        if isinstance(x, list):
            x = torch.stack(x) 
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)        
        d3 = self.downsample3(d2)        
        d4 = self.downsample4(d3)
        
        # upsampling/decoding 
        u0 = self.upsample0(d4)
        # skip-connection
        c0 = torch.cat((u0, d3), dim=1)
        
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        
        u4 = self.upsample4(c3)
        
        # u4 - the mask
        output = u4 * x
        
        return output
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = wsdr_fn(x, pred, y, self.n_fft, self.hop_length)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = wsdr_fn(x, pred, y, self.n_fft, self.hop_length)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        snr = getSNRList(pred, y, self.n_fft, self.hop_length)
        pesq = getPesqList(pred, x, self.n_fft, self.hop_length)
        
        self.pesq_scores.append(pesq)
        self.snr_scores.append(snr)
        self.total_samples += batch[0].size(0)

        # Save only the first example
        if not self.saved:
            x_audio = istft(x[0], self.n_fft, self.hop_length)
            y_audio = istft(y[0], self.n_fft, self.hop_length)
            pred_audio = istft(pred[0], self.n_fft, self.hop_length)
            
            
            # Save the first example only
            torchaudio.save("input.wav", x_audio.cpu(), 48000)
            torchaudio.save("target.wav", y_audio.cpu(), 48000)
            torchaudio.save("predicted.wav", pred_audio.cpu(), 48000)
            
            self.saved = True  # Update the flag to avoid saving again
        
    def on_predict_end(self):
        average_pesq = sum(self.pesq_scores) / self.total_samples
        average_snr = sum(self.snr_scores) / self.total_samples
        
        print(f"pesq :{average_pesq}")
        print(f"snr : {average_snr}")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
