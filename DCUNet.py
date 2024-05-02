from lightning.pytorch import LightningModule
import torch
from Encoder import Encoder
from Decoder import Decoder 
from wsdr import wsdr_fn
from rich.progress import Progress
import torchaudio
from pypesq import pesq
# from loss import basic_loss



class DCUnet10(LightningModule):
    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        
        self.save_hyperparameters()
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(6,4), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(5,4), stride_size=(2,2), in_channels=180, out_channels=90)
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
        # if is_istft:
        #     SAMPLE_RATE = 48000
        #     N_FFT = 1022
        #     HOP_LENGTH = 256
        #     output = torch.squeeze(output, 1)
        #     output = torch.view_as_complex(output)
        #     window = torch.hann_window(N_FFT,device=output.device)
        #     output = torch.istft(output, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True,window=window)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x,y,stft = batch
        pred = self.forward(stft)
        loss = wsdr_fn(x,pred,y)
        # loss = basic_loss()
        self.log("train_loss", loss, prog_bar=True,sync_dist=True)
        return loss

    # def validation_step(self, batch):
    #     SAMPLE_RATE = 48000
    #     N_FFT = SAMPLE_RATE * 64 // 1000 + 4
    #     HOP_LENGTH = SAMPLE_RATE * 16 // 1000 + 4
    #     x, y = batch
    #     pred = self.forward(x)
    #     loss = wsdr_fn(x, pred, y)
    #     self.log("train_loss", loss, prog_bar=True, sync_dist=True)
    #     pred = pred[0].unsqueeze(0)
    #     noisy = torch.view_as_complex(x[0])
    #     clean = torch.view_as_complex(y[0])
    #     window = torch.hann_window(N_FFT, device=clean.device)
    #     clean = torch.istft(clean, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=window)
    #     window = torch.hann_window(N_FFT, device=noisy.device)
    #     noisy = torch.istft(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=window)
    #     self.log("val_loss", loss, prog_bar=True, sync_dist=True)
    #     self.logger.experiment.add_audio("pred", pred, sample_rate=48000)
    #     self.logger.experiment.add_audio("noisy", noisy, sample_rate=48000)
    #     self.logger.experiment.add_audio("clean", clean, sample_rate=48000)

    #     clean = torchaudio.transforms.Resample(48000, 16000)(clean.detach().cpu())
    #     pred= torchaudio.transforms.Resample(48000, 16000)(pred.detach().cpu())

    #     # clean = clean.flatten().numpy()
    #     # pred = pred.flatten().cpu().numpy() 
    #     # Ensure the function call is correctly referencing the 'pesq' function from the 'pesq' module
    #     pesq_score = pesq(clean.flatten(), pred.flatten(), 16000)
    #     pesq_score /= len(batch)
    #     self.log("pesq", pesq_score, prog_bar=True, sync_dist=True)

    
    
    def predict_step(self, batch):
        return self(batch)
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
