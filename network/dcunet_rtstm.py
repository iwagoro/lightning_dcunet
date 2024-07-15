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
    

class DCUnet10_rTSTM(LightningModule):
    """
    Deep Complex U-Net with real TSTM.
    """
    def __init__(self, n_fft, hop_length,dataset=""):
        super().__init__()
        
        
        self.pesqNb_scores = []
        self.pesqWb_scores = []
        self.snr_scores = []
        self.stoi_scores = []
        self.total_samples = 0
        self.saved = False  # Add a flag to keep track if the audio has been saved

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = "dcunet-rtstm"
        self.dataset = dataset
        self.loss_fn = RegularizedLoss()
        self.gamma = self.gamma = 1
        
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
    
    
    def predict_step(self, batch, batch_idx):
        x,y = batch
        pred = self.forward(x)
        pesqNb = getPesqList(pred,y,self.n_fft,self.hop_length,"nb")
        pesqWb = getPesqList(pred,y,self.n_fft,self.hop_length,"wb")
        snr = getSNRList(pred,y,self.n_fft,self.hop_length)
        stoi = getSTOIList(pred,y,self.n_fft,self.hop_length)
        
        self.pesqNb_scores.append(pesqNb)
        self.pesqWb_scores.append(pesqWb)
        self.snr_scores.append(snr)
        self.stoi_scores.append(stoi)
        self.total_samples += batch[0].size(0)

        # Ensure the 'pred' directory exists
        Path("pred/"+ self.model + "-" + self.dataset).mkdir(parents=True, exist_ok=True)
        for i in range(len(x)):
            x_audio = istft(x[i], self.n_fft, self.hop_length)
            y_audio = istft(y[i], self.n_fft, self.hop_length)
            pred_audio = istft(pred[i], self.n_fft, self.hop_length)
            
            # Save the audio files
            torchaudio.save("./pred/" + self.model + "-" + self.dataset + "/noisy"+str(i)+".wav", x_audio.cpu(), 48000)
            torchaudio.save("./pred/" + self.model + "-" + self.dataset + "/clean"+str(i)+".wav", y_audio.cpu(), 48000)
            torchaudio.save("./pred/" + self.model + "-" + self.dataset + "/pred"+str(i)+".wav", pred_audio.cpu(), 48000)
            
        
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