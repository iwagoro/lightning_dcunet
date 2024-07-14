from lightning.pytorch import LightningModule
import torch
from network.Encoder import Encoder
from network.Decoder import Decoder 
from metrics import getPesqList,getSNRList,getSTOIList
from stft import istft
import torchaudio
import numpy as np
import os
from pathlib import Path
from wsdr import wsdr_fn
from stft import stft ,istft
from loss import RegularizedLoss
from stft_loss import basic_loss,reg_loss

def subsample2(wav, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav1 = []
    wav2 = []
    for i in range(len(wav)):
        channel, length = wav[i].shape
        new_length = length // 2 - 128

        # ランダムなインデックスを生成
        indices = (torch.arange(new_length, device=device) * 2 - 127).unsqueeze(0).repeat(channel, 1)

        # インデックスの境界を処理
        indices = indices.clamp(0, length - 2)

        random_choice = torch.randint(0, 2, (channel, new_length), device=device)

        # ランダムに選択されたインデックスを作成
        index1 = indices + random_choice
        index2 = indices + (1 - random_choice)

        # 新しいテンソルを作成
        wav1.append( torch.gather(wav[i].to(device), 1, index1))
        wav2.append( torch.gather(wav[i].to(device), 1, index2))
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
    

class DCUnet10(LightningModule):
    def __init__(self, n_fft, hop_length,dataset=""):
        super().__init__()
        
        
        self.cnt = 0
        self.pesqNb_scores = []
        self.pesqWb_scores = []
        self.snr_scores = []
        self.stoi_scores = []
        self.total_samples = 0
        self.saved = False  # Add a flag to keep track if the audio has been saved

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = "dcunet"
        self.dataset = dataset
        self.loss_fn = RegularizedLoss()
        self.gamma = self.gamma = 1
        
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
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
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
        noisy,clean= batch
        g1,g2 = subsample2(noisy,self.device)
        g1_stft = tensor_stft(g1,self.n_fft,self.hop_length)
        fg1_stft = self.forward(g1_stft)
        fg1 = tensor_istft(fg1_stft,self.n_fft,self.hop_length)
        with torch.no_grad():
            noisy_stft = tensor_stft(noisy,self.n_fft,self.hop_length)
            f_stft = self.forward(noisy_stft)
            f = tensor_istft(f_stft,self.n_fft,self.hop_length)
            g1f, g2f = subsample2(f,self.device)
        loss = basic_loss(g1,g2,fg1,self.n_fft,self.hop_length,self.device)+ reg_loss(fg1,g2,g1f,g2f)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    # def training_step(self, batch, batch_idx):
    #     x,y = batch
    #     pred = self.forward(x)
    #     loss = wsdr_fn(x,pred,y,self.n_fft,self.hop_length)
    #     self.log("train_loss", loss, prog_bar=True,sync_dist=True)
    #     return loss
    
    def validation_step(self, batch, batch_idx):
        noisy,clean= batch
        g1,g2 = subsample2(noisy,self.device)
        g1_stft = tensor_stft(g1,self.n_fft,self.hop_length)
        fg1_stft = self.forward(g1_stft)
        fg1 = tensor_istft(fg1_stft,self.n_fft,self.hop_length)
        with torch.no_grad():
            noisy_stft = tensor_stft(noisy,self.n_fft,self.hop_length)
            f_stft = self.forward(noisy_stft)
            f = tensor_istft(f_stft,self.n_fft,self.hop_length)
            g1f, g2f = subsample2(f,self.device)
        loss = basic_loss(g1,g2,fg1,self.n_fft,self.hop_length,self.device) + self.gamma * reg_loss(fg1,g2,g1f,g2f)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    # def validation_step(self,batch,batch_idx):
    #     x,y = batch
    #     # print(x.shape)
    #     pred = self.forward(x)
    #     loss = wsdr_fn(x,pred,y,self.n_fft,self.hop_length)
    #     self.log("val_loss", loss, prog_bar=True,sync_dist=True)
    #     return loss
    
    # def predict_step(self, batch, batch_idx):
    #     x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft= batch
    #     x = x_noisy_stft
    #     y = x_clean_stft
    #     pred = self.forward(x)
    #     snr = getSNRList(pred, y, self.n_fft, self.hop_length)
    #     pesq = getPesqList(pred, x, self.n_fft, self.hop_length)
        
    #     self.pesq_scores.append(pesq)
    #     self.snr_scores.append(snr)
    #     self.total_samples += batch[0].size(0)

    #     # Save only the first example
    #     if not self.saved:
    #         x_audio = istft(x[5], self.n_fft, self.hop_length)
    #         y_audio = istft(y[5], self.n_fft, self.hop_length)
    #         pred_audio = istft(pred[5], self.n_fft, self.hop_length)
            
            
    #         # Save the first example only
    #         torchaudio.save("input.wav", x_audio.cpu(), 48000)
    #         torchaudio.save("target.wav", y_audio.cpu(), 48000)
    #         torchaudio.save("predicted.wav", pred_audio.cpu(), 48000)
            
    #         self.saved = True  # Update the flag to avoid saving again
        
    # def on_predict_end(self):
    #     average_pesq = sum(self.pesq_scores) / self.total_samples
    #     average_snr = sum(self.snr_scores) / self.total_samples
        
    #     print(f"pesq :{average_pesq}")
    #     print(f"snr : {average_snr}")
    
    def predict_step(self, batch, batch_idx):
        x,y = batch
        x = tensor_stft(x,self.n_fft,self.hop_length)
        y = tensor_stft(y,self.n_fft,self.hop_length)
        pred = self.forward(x)
        pesqNb = getPesqList(pred,y,self.n_fft,self.hop_length,"nb")
        pesqWb = getPesqList(pred,y,self.n_fft,self.hop_length,"wb")
        snr = getSNRList(pred,x,self.n_fft,self.hop_length)
        stoi = getSTOIList(pred,y,self.n_fft,self.hop_length)
        
        self.pesqNb_scores.append(pesqNb)
        self.pesqWb_scores.append(pesqWb)
        self.snr_scores.append(snr)
        self.stoi_scores.append(stoi)
        self.total_samples += batch[0].size(0)

        # Ensure the 'pred' directory exists
        Path("pred/"+ self.model + "-" + self.dataset).mkdir(parents=True, exist_ok=True)

        if not self.saved:
            for i in range(len(x)):
                x_audio = istft(x[self.cnt], self.n_fft, self.hop_length)
                y_audio = istft(y[self.cnt], self.n_fft, self.hop_length)
                pred_audio = istft(pred[self.cnt], self.n_fft, self.hop_length)
                
                # Save the audio files
                torchaudio.save("./pred/" + self.model + "-" + self.dataset + "/noisy"+str(self.cnt)+".wav", x_audio.cpu(), 48000)
                torchaudio.save("./pred/" + self.model + "-" + self.dataset + "/clean"+str(self.cnt)+".wav", y_audio.cpu(), 48000)
                torchaudio.save("./pred/" + self.model + "-" + self.dataset + "/pred"+str(self.cnt)+".wav", pred_audio.cpu(), 48000)
                
                self.cnt += 1
            
            # self.saved = True  # Update the flag to avoid saving again
        
    def on_predict_end(self):
        average_pesqNb = sum(self.pesqNb_scores) / self.total_samples
        average_pesqWb = sum(self.pesqWb_scores) / self.total_samples
        average_snr = sum(self.snr_scores) / self.total_samples
        average_stoi = sum(self.stoi_scores) / self.total_samples
        
        print("-----------------------------------")
        print("model : "+self.model)
        print("dataset : "+ self.dataset)
        print("-----------------------------------")
        print(f"pesq-nb :{average_pesqNb}")
        print(f"pesq-wb :{average_pesqWb}")
        print(f"snr : {average_snr}")
        print(f"stoi : {average_stoi}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]
