import torch
from preprocess import TorchSignalToFrames
import torch.fft

class mse_loss(object):
    def __call__(self, outputs, labels, loss_mask):
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        loss = torch.sum((masked_outputs - masked_labels)**2.0) / torch.sum(loss_mask)
        return loss

class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.frame = TorchSignalToFrames(frame_size=self.frame_size, frame_shift=self.frame_shift)
        W = torch.hamming_window(self.frame_size)
        self.W = W.float().cuda()

    def __call__(self, outputs, labels, loss_mask):
        device = outputs.device  # 出力テンソルのデバイスを取得
        self.W = self.W.to(device)  # 重みテンソルを出力テンソルのデバイスに移動
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        loss_mask = self.frame(loss_mask)
        outputs = self.get_stftm(outputs)
        labels = self.get_stftm(labels)

        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        if self.loss_type == 'mse':
            loss = torch.sum((masked_outputs - masked_labels)**2) / torch.sum(loss_mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(masked_outputs - masked_labels)) / torch.sum(loss_mask)

        return loss

    def get_stftm(self, frames):
        frames = frames * self.W
        stft = torch.fft.fft(frames, n=self.frame_size, dim=-1)
        stftm = torch.abs(stft.real) + torch.abs(stft.imag)
        return stftm

class reg_loss(object):
    def __call__(self, fg1, g2, g1fx, g2fx):
        return torch.mean((fg1 - g2 - g1fx + g2fx)**2)
