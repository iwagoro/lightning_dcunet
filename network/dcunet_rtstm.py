from lightning.pytorch import LightningModule
import torch
from network.Encoder import Encoder
from network.Decoder import Decoder 
from metrics import getPesqList,getSNRList,getSTOIList
import torchaudio
from network.Dual_Transformer import Dual_Transformer
from pathlib import Path
from stft import istft , tensor_stft,tensor_istft
from loss import basic_loss,reg_loss,wsdr_loss
from subsample import subsample2


class DCUnet10_rTSTM(LightningModule):
    def __init__(self,dataset="",loss_type="nct"):
        super().__init__()
        
        
        self.pesqNb_scores = []
        self.pesqWb_scores = []
        self.snr_scores = []
        self.stoi_scores = []
        self.total_samples = 0
        self.saved = False  
        self.loss_type = loss_type

        self.model = "dcunet-rtstm"
        self.dataset = dataset
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
        if(self.loss_type == "nb2nb"):
            noisy,clean= batch
            g1,g2 = subsample2(noisy,self.device)
            g1_stft = tensor_stft(g1)
            fg1_stft = self.forward(g1_stft)
            fg1 = tensor_istft(fg1_stft)
            with torch.no_grad():
                noisy_stft = tensor_stft(noisy)
                f_stft = self.forward(noisy_stft)
                f = tensor_istft(f_stft)
                g1f, g2f = subsample2(f,self.device)
            loss = basic_loss(g1,g2,fg1,self.device)+ reg_loss(fg1,g2,g1f,g2f)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss
        
        elif (self.loss_type == "nct"):
            x,y = batch
            pred = self.forward(x)
            loss = wsdr_loss(x,pred,y)
            self.log("train_loss", loss, prog_bar=True,sync_dist=True)
            return loss

    
    def validation_step(self, batch, batch_idx):
        if(self.loss_type == "nb2nb"):
            noisy,clean= batch
            g1,g2 = subsample2(noisy,self.device)
            g1_stft = tensor_stft(g1)
            fg1_stft = self.forward(g1_stft)
            fg1 = tensor_istft(fg1_stft)
            with torch.no_grad():
                noisy_stft = tensor_stft(noisy)
                f_stft = self.forward(noisy_stft)
                f = tensor_istft(f_stft)
                g1f, g2f = subsample2(f,self.device)
            loss = basic_loss(g1,g2,fg1,self.device) + self.gamma * reg_loss(fg1,g2,g1f,g2f)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            return loss
    
        elif (self.loss_type == "nct"):
            x,y = batch
            pred = self.forward(x)
            loss = wsdr_loss(x,pred,y)
            self.log("train_loss", loss, prog_bar=True,sync_dist=True)
            return loss

    
    def predict_step(self, batch, batch_idx):
        x,y = batch
        pred = self.forward(x)
        pesqNb = getPesqList(pred,"nb")
        pesqWb = getPesqList(pred,"wb")
        snr = getSNRList(pred,y)
        stoi = getSTOIList(pred,y)
        
        self.pesqNb_scores.append(pesqNb)
        self.pesqWb_scores.append(pesqWb)
        self.snr_scores.append(snr)
        self.stoi_scores.append(stoi)
        self.total_samples += batch[0].size(0)

        # Ensure the 'pred' directory exists
        Path("pred/"+ self.model + "-" + self.dataset).mkdir(parents=True, exist_ok=True)
        for i in range(len(x)):
            x_audio = istft(x[i])
            y_audio = istft(y[i])
            pred_audio = istft(pred[i])
            
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