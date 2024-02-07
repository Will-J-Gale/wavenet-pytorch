import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            dilation=dilation,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stack = nn.Sequential(
            ConvBlock(in_channels, 1),
            ConvBlock(in_channels, 2),
            ConvBlock(in_channels, 4),
            ConvBlock(in_channels, 8),
            ConvBlock(in_channels, 16),
            ConvBlock(in_channels, 32),
            ConvBlock(in_channels, 64),
            ConvBlock(in_channels, 128),
            ConvBlock(in_channels, 256),
            ConvBlock(in_channels, 512),
        )
        self.head = nn.Conv1d(in_channels, out_channels, kernel_size=(1,), stride=(1,), bias=False)
        self.receptive_field = sum([block.dilation for block in self.stack]) * (self.stack[0].conv.kernel_size[0] - 1)

    def forward(self, x):
        out = x
        for layer in self.stack:
            out = layer(out)
        out = self.head(out) 
        return out

class ConvNet(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.input_conv = nn.Conv1d(1, 16, kernel_size=(1,), stride=(1,), bias=False)
        self.layers = nn.Sequential(
            ConvLayer(16, 8),
            ConvLayer(8, 4),
            ConvLayer(4, 1),
        )

        self.receptive_field = sum([layer.receptive_field for layer in self.layers])

    def forward(self, x, pad_start=True):
        if pad_start:
            x = torch.cat(
                (torch.zeros((len(x), self.receptive_field)).to(x.device), x), dim=1
            )

        if(x.ndim == 2):
            x = x[:, None, :]
        
        out = self.input_conv(x)

        for layer in self.layers:
            out = layer(out)

        return out.squeeze()

if __name__ == "__main__":
    convnet = ConvNet()
    input_shape = 8192 + convnet.receptive_field
    x = (torch.rand(16, 1, 14330) * 2) - 1
    y = convnet(x, pad_start=False)
    print(y)
    # print(y.shape)
