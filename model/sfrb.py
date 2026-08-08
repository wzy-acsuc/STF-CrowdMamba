import torch
import torch.nn as nn


class SFRB(nn.Module):
    """Reference FB block with an input LayerNorm and a raw-feature residual.

    The input and output use the STF-Mamba layout [T, C, H, W]. FFT processing is
    performed independently for each frame; temporal modeling remains in the
    spatio-temporal MB branch.
    """

    def __init__(self, dim, norm_eps=1e-6):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.select1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.select2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        raw = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.input_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        _, _, height, width = x.shape
        spectrum = torch.fft.rfft2(x) + 1e-8
        amplitude = self.select1(torch.abs(spectrum))
        phase = self.select2(torch.angle(spectrum))
        real = amplitude * torch.cos(phase)
        imaginary = amplitude * torch.sin(phase)
        enhanced_spectrum = torch.complex(real, imaginary) + 1e-8
        enhanced = torch.fft.irfft2(enhanced_spectrum, s=(height, width), norm='backward') + 1e-8

        return enhanced + raw


class SFRBBlock(nn.Module):
    """A stack of independent FEB blocks for one STF-Mamba frequency stage."""

    def __init__(self, dim, depth=2, norm_eps=1e-6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.blocks = nn.ModuleList(SFRB(dim, norm_eps=norm_eps) for _ in range(depth))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
