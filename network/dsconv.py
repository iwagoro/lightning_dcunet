import torch.nn as nn
import torch

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = depthwise_separable_conv(in_channels=self.in_channels, 
                                                  nout=self.out_channels,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  bias=False)
        
        self.im_conv = depthwise_separable_conv(in_channels=self.in_channels, 
                                                nout=self.out_channels,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                bias=False)
        
        
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        
        x = x.to(torch.float32)     
        
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output
    
class CConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = depthwise_separable_conv(in_channels=self.in_channels, 
                                                    nout=self.out_channels,
                                                    kernel_size=self.kernel_size,
                                                    padding=self.padding,
                                                    bias=False)
        
        self.im_convt = depthwise_separable_conv(in_channels=self.in_channels, 
                                                  nout=self.out_channels,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  bias=False)
        
        
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
            
        x = x.to(torch.float32) 
        
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output
