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

class WaveNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv1d(1, 16, kernel_size=(1,), stride=(1,), bias=False)
        self.residual_stack1 = nn.Sequential(
            ResidualBlock(16, 1),
            ResidualBlock(16, 2),
            ResidualBlock(16, 4),
            ResidualBlock(16, 8),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32),
            ResidualBlock(16, 64),
            ResidualBlock(16, 128),
            ResidualBlock(16, 256),
            ResidualBlock(16, 512)
        )
        self.head1 = nn.Conv1d(16, 8, kernel_size=(1,), stride=(1,), bias=False)

        self.residual_stack2 = nn.Sequential(
            ResidualBlock(8, 1),
            ResidualBlock(8, 2),
            ResidualBlock(8, 4),
            ResidualBlock(8, 8),
            ResidualBlock(8, 16),
            ResidualBlock(8, 32),
            ResidualBlock(8, 64),
            ResidualBlock(8, 128),
            ResidualBlock(8, 256),
            ResidualBlock(8, 512)
        )
        self.head_2 = nn.Conv1d(8, 4, kernel_size=(1,), stride=(1,), bias=False)

        self.residual_stack3 = nn.Sequential(
            ResidualBlock(4, 1),
            ResidualBlock(4, 2),
            ResidualBlock(4, 4),
            ResidualBlock(4, 8),
            ResidualBlock(4, 16),
            ResidualBlock(4, 32),
            ResidualBlock(4, 64),
            ResidualBlock(4, 128),
            ResidualBlock(4, 256),
            ResidualBlock(4, 512)
        )
        self.output_conv = nn.Conv1d(4, 1, kernel_size=(1,), stride=(1,), bias=False)

        dilation_sum = sum([layer.dilation for layer in self.residual_stack1 + self.residual_stack2 + self.residual_stack3])
        self.receptive_field = (self.residual_stack1[0].kernel_size - 1) * dilation_sum

    def _sum_skips(self, skips, out):
        out_shape = out.shape[2]
        for skip in skips:
            out += skip[..., -out_shape:]

    def forward(self, x, pad_start=True):
        if pad_start:
            x = torch.cat(
                (torch.zeros((len(x), self.receptive_field)).to(x.device), x), dim=1
            )
        
        if(x.ndim == 2):
            x = x[:, None, :]
        
        out = self.input_conv(x)
        skips = []

        for layer in self.residual_stack1:
            out, skip = layer(out)
            skips.append(skip)

        out = torch.zeros_like(out)
        self._sum_skips(skips, out)
        out = self.head1(out)

        skips = []
        for layer in self.residual_stack2:
            out, skip = layer(out)
            skips.append(skip)

        out = torch.zeros_like(out)
        self._sum_skips(skips, out)
        out = self.head_2(out)

        skips = []
        for layer in self.residual_stack3:
            out, skip = layer(out)
            skips.append(skip)
        
        out = torch.zeros_like(out)
        self._sum_skips(skips, out)
        out = self.output_conv(out)
        return out.squeeze()

if __name__ == "__main__":
    wavenet = WaveNet()
    input_shape = 8192 + wavenet.receptive_field
    x = (torch.rand(16, 1, 14330) * 2) - 1
    y = wavenet(x, pad_start=False)
    print(y.shape)