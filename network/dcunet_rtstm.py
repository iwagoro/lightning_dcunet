from lightning.pytorch import LightningModule
import torch
from network.Encoder import Encoder
from network.Decoder import Decoder 
from metrics import getPesqList,getSNRList,getSTOIList
from stft import istft
import torchaudio
import numpy as np
import os
from network.Dual_Transformer import Dual_Transformer
from pathlib import Path
from wsdr import wsdr_fn
from loss import RegularizedLoss

def subsample2(wav):  
    # This function only works for k = 2 as of now.
    k = 2
    channels, dim= np.shape(wav) 

    dim1 = dim // k -128     # 128 is used to correct the size of the sampled data, you can change it
    wav1, wav2 = np.zeros([channels, dim1]), np.zeros([channels, dim1])   # [2, 1, 32640]
    #print("wav1:", wav1.shape)
    #print("wav2:", wav2.shape)

    wav_cpu = wav.cpu()
    for channel in range(channels):
        for i in range(dim1):
            i1 = i * k
            num = np.random.choice([0, 1])
            if num == 0:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1], wav_cpu[channel, i1+1]
            elif num == 1:
                wav1[channel, i], wav2[channel, i] = wav_cpu[channel, i1+1], wav_cpu[channel, i1]

    return torch.from_numpy(wav1).cuda(), torch.from_numpy(wav2).cuda()

class DCUnet10_rTSTM(LightningModule):
    """
    Deep Complex U-Net with real TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        
        self.loss_fn = RegularizedLoss()
        self.cnt = 0
        self.pesqNb_scores = []
        self.pesqWb_scores = []
        self.snr_scores = []
        self.stoi_scores = []
        self.total_samples = 0
        self.saved = False  # Add a flag to keep track if the audio has been saved
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x):
        # encoder
        d0 = self.downsample0(x)
        # print("d0:",d0.shape)
        d1 = self.downsample1(d0) 
        # print("d1:",d1.shape)
        d2 = self.downsample2(d1)
        # print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    
        # print("d3:",d3.shape)    
        d4 = self.downsample4(d3)
        # print("d4:",d4.shape)
        
        # real TSTM
        d4_1 = d4[:, :, :, :, 0]
        d4_2 = d4[:, :, :, :, 1]
        d4_1 = self.dual_transformer(d4_1)
        d4_2 = self.dual_transformer(d4_2)

        out = torch.rand(d4.shape)
        out[:, :, :, :, 0] = d4_1
        out[:, :, :, :, 1] = d4_2
        out= out.to('cuda')

        # decoder
        u0 = self.upsample0(out)    # upsampling/decoding 
        c0 = torch.cat((u0, d3), dim=1)   # skip-connection
        # print("c0:",c0.shape)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        # print("c1:",c1.shape)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        # print("c2:",c2.shape)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        # print("c3:",c3.shape)
        u4 = self.upsample4(c3)
        # print("u4:",u4.shape)

        output = u4 * x    # u4 - the mask

        
        return output
    
    
    # def training_step(self, batch, batch_idx):
    #     x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft= batch
    #     fg1_wav = self.forward(g1_stft)
    #     fg1_wav = istft(fg1_wav,self.n_fft,self.hop_length)
    #     with torch.no_grad():
    #         fx_wav = self.forward(x_noisy_stft)
    #         fx_wav = istft(fx_wav,self.n_fft,self.hop_length)
    #         g1fx, g2fx = subsample2(fx_wav)
    #         g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)
    #     g1_wav, fg1_wav, g2_wav, g1fx, g2fx = g1_wav.cuda(), fg1_wav.cuda(), g2_wav.cuda(), g1fx.cuda(), g2fx.cuda()
    #     loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
    #     self.log("train_loss", loss, prog_bar=True, sync_dist=True)
    #     return loss
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        pred = self.forward(x)
        loss = wsdr_fn(x,pred,y,self.n_fft,self.hop_length)
        self.log("train_loss", loss, prog_bar=True,sync_dist=True)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft= batch
    #     fg1_wav = self.forward(g1_stft)
    #     fg1_wav = istft(fg1_wav,self.n_fft,self.hop_length)
    #     with torch.no_grad():
    #         fx_wav = self.forward(x_noisy_stft)
    #         fx_wav = istft(fx_wav,self.n_fft,self.hop_length)
    #         g1fx, g2fx = subsample2(fx_wav)
    #         g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)
    #     g1_wav, fg1_wav, g2_wav, g1fx, g2fx = g1_wav.cuda(), fg1_wav.cuda(), g2_wav.cuda(), g1fx.cuda(), g2fx.cuda()
    #     loss = self.loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
    #     self.log("val_loss", loss, prog_bar=True, sync_dist=True)
    #     return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        # print(x.shape)
        pred = self.forward(x)
        loss = wsdr_fn(x,pred,y,self.n_fft,self.hop_length)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        return loss
    
    # def predict_step(self, batch, batch_idx):
    #     x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft= batch
    #     x = x_noisy_stft
    #     y = x_clean_stft
    #     pred = self.forward(x)
    #     snr = getSNRList(x, y, self.n_fft, self.hop_length)
    #     pesq = getPesqList(y, x, self.n_fft, self.hop_length)
        
    #     self.pesq_scores.append(pesq)
    #     self.snr_scores.append(snr)
    #     self.total_samples += batch[0].size(0)

    #     # Ensure the 'pred' directory exists
    #     Path("pred").mkdir(parents=True, exist_ok=True)

    #     if not self.saved:
    #         for i in range(len(batch)):
    #             x_audio = istft(x[self.cnt], self.n_fft, self.hop_length)
    #             y_audio = istft(y[self.cnt], self.n_fft, self.hop_length)
    #             pred_audio = istft(pred[self.cnt], self.n_fft, self.hop_length)
                
    #             # Save the audio files
    #             torchaudio.save("pred/input"+str(self.cnt)+".wav", x_audio.cpu(), 48000)
    #             torchaudio.save("pred/target"+str(self.cnt)+".wav", y_audio.cpu(), 48000)
    #             torchaudio.save("pred/predicted"+str(self.cnt)+".wav", pred_audio.cpu(), 48000)
                
    #             self.cnt += 1
            
    #         # self.saved = True  # Update the flag to avoid saving again
        
    # def on_predict_end(self):
    #     average_pesq = sum(self.pesq_scores) / self.total_samples
    #     average_snr = sum(self.snr_scores) / self.total_samples
        
    #     print(f"pesq :{average_pesq}")
    #     print(f"snr : {average_snr}")
    
    
    def predict_step(self, batch, batch_idx):
        x,y = batch
        pred = self.forward(x)
        pesqNb = getPesqList(pred,x,self.n_fft,self.hop_length,"nb")
        pesqWb = getPesqList(pred,x,self.n_fft,self.hop_length,"wb")
        snr = getSNRList(pred,y,self.n_fft,self.hop_length)
        stoi = getSTOIList(pred,y,self.n_fft,self.hop_length)
        
        self.pesqNb_scores.append(pesqNb)
        self.pesqWb_scores.append(pesqWb)
        self.snr_scores.append(snr)
        self.stoi_scores.append(stoi)
        self.total_samples += batch[0].size(0)

        # Ensure the 'pred' directory exists
        Path("pred").mkdir(parents=True, exist_ok=True)

        if not self.saved:
            for i in range(len(batch)):
                x_audio = istft(x[self.cnt], self.n_fft, self.hop_length)
                y_audio = istft(y[self.cnt], self.n_fft, self.hop_length)
                pred_audio = istft(pred[self.cnt], self.n_fft, self.hop_length)
                
                # Save the audio files
                torchaudio.save("/workspace/app/ont/pred/input"+str(self.cnt)+".wav", x_audio.cpu(), 48000)
                torchaudio.save("/workspace/app/ont/pred/target"+str(self.cnt)+".wav", y_audio.cpu(), 48000)
                torchaudio.save("/workspace/app/ont/pred/predicted"+str(self.cnt)+".wav", pred_audio.cpu(), 48000)
                
                self.cnt += 1
            
            # self.saved = True  # Update the flag to avoid saving again
        
    def on_predict_end(self):
        average_pesqNb = sum(self.pesqNb_scores) / self.total_samples
        average_pesqWb = sum(self.pesqWb_scores) / self.total_samples
        average_snr = sum(self.snr_scores) / self.total_samples
        average_stoi = sum(self.stoi_scores) / self.total_samples
        
        print(f"pesq-nb :{average_pesqNb}")
        print(f"pesq-wb :{average_pesqWb}")
        print(f"snr : {average_snr}")
        print(f"stoi : {average_stoi}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]