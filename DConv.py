import torch.nn as nn

class DConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False,stride=(1,1)):
        super(DConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias,stride=stride
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        nn.init.xavier_uniform_(self.depthwise.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out