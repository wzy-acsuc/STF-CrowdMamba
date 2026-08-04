"""Spatio-temporal MB blocks used in place of the original Ddcb blocks.

The block preserves the supplied MB architecture: projection and gate, spatial
depth-wise convolutions, SSM output norms, spatial upsampling, output
projection, dropout, and an internal residual. Its low-resolution part differs
from the reference MB: independent average-pooling and max-pooling branches
operate at H/2 x W/2, are fused, and are upsampled once. Only the SSM scan is
changed to a bidirectional scan over a joint T-H-W sequence.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:  # Raised with an actionable message when the block is instantiated.
    selective_scan_fn = None


class SSMTHW(nn.Module):
    """Two-way selective scan over a volume with shape [B, T, C, H, W]."""

    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "Spatio-temporal MB requires mamba_ssm. Install a version that provides "
                "mamba_ssm.ops.selective_scan_interface.selective_scan_fn."
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.num_scans = 2

        x_projs = tuple(
            nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False, **factory_kwargs)
            for _ in range(self.num_scans)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([projection.weight for projection in x_projs], dim=0))

        dt_projs = tuple(
            self._dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(self.num_scans)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([projection.weight for projection in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([projection.bias for projection in dt_projs], dim=0))
        self.dt_projs_bias._no_reinit = True

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=self.num_scans, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=self.num_scans, merge=True)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs):
        projection = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt initialization: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus_dt)
        projection.bias._no_reinit = True
        return projection

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner)
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x):
        """Return [B, T, H, W, D] after forward and reversed T-H-W scans."""
        batch, frames, channels, height, width = x.shape
        length = frames * height * width

        forward = x.permute(0, 2, 1, 3, 4).contiguous().reshape(batch, channels, length)
        scans = torch.stack((forward, torch.flip(forward, dims=(-1,))), dim=1)

        projected = torch.einsum("b k d l, k c d -> b k c l", scans, self.x_proj_weight)
        dts, Bs, Cs = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        output = selective_scan_fn(
            scans.float().reshape(batch, -1, length),
            dts.contiguous().float().reshape(batch, -1, length),
            -torch.exp(self.A_logs.float()).reshape(-1, self.d_state),
            Bs.float().reshape(batch, self.num_scans, self.d_state, length),
            Cs.float().reshape(batch, self.num_scans, self.d_state, length),
            self.Ds.float().reshape(-1),
            z=None,
            delta_bias=self.dt_projs_bias.float().reshape(-1),
            delta_softplus=True,
            return_last_state=False,
        ).reshape(batch, self.num_scans, channels, length)

        output = output[:, 0] + torch.flip(output[:, 1], dims=(-1,))
        return output.reshape(batch, channels, frames, height, width).permute(0, 2, 3, 4, 1).contiguous()


class SpatioTemporalMB(nn.Module):
    """MB with a pre-normalized raw residual and joint T-H-W selective scans."""

    def __init__(self, d_model, d_state=16, expand=2., dropout=0., bias=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        self.input_norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias, **factory_kwargs)
        self.out_norm_high = nn.LayerNorm(self.d_inner, **factory_kwargs)
        self.out_norm_avg = nn.LayerNorm(self.d_inner, **factory_kwargs)
        self.out_norm_max = nn.LayerNorm(self.d_inner, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.avg_pooling = nn.AvgPool2d(kernel_size=(2, 2))
        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca_high = self._spatial_dwconv(**factory_kwargs)
        self.ca_avg = self._spatial_dwconv(**factory_kwargs)
        self.ca_max = self._spatial_dwconv(**factory_kwargs)
        self.ssm_high = SSMTHW(d_model=d_model, d_state=d_state, expand=expand, device=device, dtype=dtype)
        self.ssm_avg = SSMTHW(d_model=d_model, d_state=d_state, expand=expand, device=device, dtype=dtype)
        self.ssm_max = SSMTHW(d_model=d_model, d_state=d_state, expand=expand, device=device, dtype=dtype)

    def _spatial_dwconv(self, **factory_kwargs):
        return nn.Sequential(
            nn.Conv2d(
                self.d_inner,
                self.d_inner,
                kernel_size=3,
                padding=1,
                groups=self.d_inner,
                bias=True,
                **factory_kwargs,
            ),
            nn.SiLU(),
        )

    @staticmethod
    def _apply_spatial_operator(x, operator):
        batch, frames, channels, height, width = x.shape
        x = x.reshape(batch * frames, channels, height, width)
        x = operator(x)
        return x.reshape(batch, frames, x.shape[1], x.shape[2], x.shape[3])

    def _upsample_spatial(self, x):
        batch, frames, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().reshape(batch * frames, channels, height, width)
        x = self.up(x)
        return x.reshape(batch, frames, channels, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x):
        """Accept [T, C, H, W] or [B, T, C, H, W] and preserve that layout."""
        has_batch_dimension = x.ndim == 5
        if not has_batch_dimension:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"Expected [T,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")

        batch, frames, channels, height, width = x.shape
        if channels != self.d_model:
            raise ValueError(f"Expected {self.d_model} channels, got {channels}")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("SpatioTemporalMB requires even H and W for its pooled branches.")

        raw = x.permute(0, 1, 3, 4, 2).contiguous()
        normalized = self.input_norm(raw)
        x, z = self.in_proj(normalized).chunk(2, dim=-1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        high = self._apply_spatial_operator(x, self.ca_high)
        y_high = self.out_norm_high(self.ssm_high(high))

        avg = self._apply_spatial_operator(x, self.avg_pooling)
        avg = self._apply_spatial_operator(avg, self.ca_avg)
        y_avg = self.out_norm_avg(self.ssm_avg(avg))

        maximum = self._apply_spatial_operator(x, self.max_pooling)
        maximum = self._apply_spatial_operator(maximum, self.ca_max)
        y_max = self.out_norm_max(self.ssm_max(maximum))

        y_low = self._upsample_spatial(y_avg + y_max)
        y = y_high + y_low
        output = self.out_proj(y * F.silu(z)) + raw
        if self.dropout is not None:
            output = self.dropout(output)

        output = output.permute(0, 1, 4, 2, 3).contiguous()
        return output if has_batch_dimension else output.squeeze(0)


class SpatioTemporalMBLayer(nn.Module):
    """A stack of residual spatio-temporal MB blocks for one DDCM stage."""

    def __init__(self, in_channel, num_blocks=2, d_state=16, expand=2., dropout=0.):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.blocks = nn.ModuleList(
            SpatioTemporalMB(in_channel, d_state=d_state, expand=expand, dropout=dropout)
            for _ in range(num_blocks)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
