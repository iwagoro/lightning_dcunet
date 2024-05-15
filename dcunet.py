from lightning.pytorch import LightningModule
import torch
from Encoder import Encoder
from Decoder import Decoder 
from loss import wsdr_fn
from metrics import getPesqList,getSNRList
# from loss import basic_loss



class DCUnet10(LightningModule):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        self.pesq_scores = []
        self.snr_scores = []
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.save_hyperparameters()
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(7,5), stride_size=(2,2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(7,5), stride_size=(2,2), in_channels=90, output_padding=(0,1),
                                 out_channels=1, last_layer=True)
        
        
    def forward(self, x, is_istft=True):
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)        
        d3 = self.downsample3(d2)        
        d4 = self.downsample4(d3)
        
        # upsampling/decoding 
        u0 = self.upsample0(d4)
        # # skip-connection
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
        x,y = batch
        pred = self.forward(x)
        loss = wsdr_fn(x,pred,y,self.n_fft,self.hop_length)
        self.log("train_loss", loss, prog_bar=True,sync_dist=True)
        return loss
    
    def validation_step(self,batch):
        x,y = batch
        pred = self.forward(x)
        loss = wsdr_fn(x,pred,y,self.n_fft,self.hop_length)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        return loss
    
    def prediction_step(self,batch):
        x,y = batch
        pred = self.forward(x)
        pesq = getSNRList(pred,y)
        snr = getPesqList(pred,x)
        
        self.pesq_scores.append(pesq)
        self.snr_scores.append(snr)
        
    def on_predict_batch_end(self):
        average_pesq = sum(self.pesq_scores) / len(self.pesq_scores)
        average_snr = sum(self.snr_scores) / len(self.snr_scores)
        
        print(f"pesq :{average_pesq}")
        print(f"snr : {average_snr}")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
