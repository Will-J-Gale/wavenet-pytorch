import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            dilation=dilation
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self._1x1 = nn.Conv1d(channels, channels, 1)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        out = self.conv(x)
        out = self.tanh(out) * self.sigmoid(out)
        out = self._1x1(out)
        return x[..., -out.shape[2]:] + out, out

class WavenetLayer(nn.Module):
    def __init__(self, input_channels, output_channels, num_residual_blocks=10):
        super().__init__()
        self.stack = nn.Sequential()

        for i in range(num_residual_blocks):
            block = ResidualBlock(input_channels, 2**i)
            self.stack.append(block)

        self.head = nn.Conv1d(input_channels, output_channels, kernel_size=(1,), stride=(1,), bias=False)
        self.dilation_sum = sum([layer.dilation for layer in self.stack])
    
    def forward(self, x):
        skip_connections = []
        out = x

        for layer in self.stack:
            out, skip = layer(out)
            skip_connections.append(skip)

        out = torch.zeros_like(out)
        self._sum_skips(skip_connections, out)
        out = self.head(out)
        return out
    
    def _sum_skips(self, skips, out):
        out_shape = out.shape[2]
        for skip in skips:
            out += skip[..., -out_shape:]

class WaveNet(nn.Module):
    def __init__(self, num_residual_blocks=10, num_layers=3, start_channels=16):
        super().__init__()

        self.input_conv = nn.Conv1d(1, start_channels, kernel_size=(1,), stride=(1,), bias=False)
        self.layers = nn.Sequential() 

        channels = start_channels
        for i in range(num_layers):
            in_channels = channels
            out_channels = channels // 2 if i != num_layers-1 else 1
            layer = WavenetLayer(in_channels, out_channels, num_residual_blocks) 
            self.layers.append(layer)
            channels = channels // 2
            
        dilation_sum = 0
        for layer in self.layers:
            dilation_sum += layer.dilation_sum

        self.receptive_field = (self.layers[0].stack[0].kernel_size - 1) * dilation_sum

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