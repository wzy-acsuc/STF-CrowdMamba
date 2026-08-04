import time
import torch
from torch import nn
from model.st_mb import SpatioTemporalMBLayer
from model.feb import FEBLayer
from model.shiftvit_T import ShiftVitCount
from model.up import UpsampleModule
# from torchinfo import summary


class STFMamba(nn.Module):
    def __init__(self, is_train=True, fm_channel=768, dstb_ratio=4, shift_c=1 / 3,
                 ms_stmb_depth_1=2, ms_stmb_depth_2=2,
                 sfrb_depth_1=1, sfrb_depth_2=1) -> None:
        super().__init__()
        stage_depths = (ms_stmb_depth_1, ms_stmb_depth_2, sfrb_depth_1, sfrb_depth_2)
        if any(depth < 1 for depth in stage_depths):
            raise ValueError("MS-STMB and SFRB stage depths must all be at least 1")
        self.shiftvit = ShiftVitCount(is_train=is_train)
        self.up_1 = UpsampleModule(fm_channel)
        self.dense_stb_1 = SpatioTemporalMBLayer(fm_channel // 2, num_blocks=ms_stmb_depth_1)
        self.frequency_enhance_1 = FEBLayer(fm_channel // 2, depth=sfrb_depth_1)
        self.up_2 = UpsampleModule(fm_channel // 2)
        self.dense_stb_2 = SpatioTemporalMBLayer(fm_channel // 4, num_blocks=ms_stmb_depth_2)
        self.frequency_enhance_2 = FEBLayer(fm_channel // 4, depth=sfrb_depth_2)
        # self.outputlayer_att = CrossAttention(fm_channel // 4)
        # self.outputlayer_wight = WightAttendtion(fm_channel // 4)
        self.outputlayer = nn.Sequential(
            nn.Conv2d(fm_channel // 4 * 2, fm_channel // 4, 1),
            nn.GELU(),
            nn.Conv2d(fm_channel // 4, fm_channel // 4 // 2, 1),
            nn.GELU(),
            nn.Conv2d(fm_channel // 4 // 2, 1, 1),
        )

    def forward(self, images):
        b, t, c, h, w = images.size()
        images = images.squeeze(0)
        d32_fm = self.shiftvit(images)
        d16_fm = self.up_1(d32_fm)
        d8_fm_d = self.up_2(self.dense_stb_1(d16_fm))
        d8_fm_f = self.up_2(self.frequency_enhance_1(d16_fm))
        d8_fm_d = self.dense_stb_2(d8_fm_d)
        d8_fm_f = self.frequency_enhance_2(d8_fm_f)
        fm_glo = self.outputlayer(torch.cat([d8_fm_d, d8_fm_f], dim=1))
        # fm_glo = self.outputlayer_att(d8_fm_d, d8_fm_f)
        # fm_glo = self.outputlayer_wight(d8_fm_d, d8_fm_f)

        return fm_glo


# Backward-compatible name for existing scripts and checkpoints.
DDCM = STFMamba


class CrossAttention(nn.Module):
    def __init__(self, dim_in):
        super(CrossAttention, self).__init__()
        self.dim_in = dim_in
        self.linear_q = nn.Linear(dim_in, dim_in)
        self.linear_k = nn.Linear(dim_in, dim_in)
        self.linear_v = nn.Linear(dim_in, dim_in)
        self.linear_out = nn.Linear(dim_in, dim_in)
        self.output = nn.Conv2d(dim_in, 1, 1)

    def forward(self, local_feat, global_feat):
        B, C, H, W = local_feat.shape
        local_feat_reshape = local_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        global_feat_reshape = global_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        q = self.linear_q(local_feat_reshape)  # [B, HW, C/r]
        k = self.linear_k(global_feat_reshape)  # [B, HW, C/r]
        v = self.linear_v(global_feat_reshape)  # [B, HW, C/r]
        attn = torch.bmm(q, k.transpose(1, 2))  # [B, HW, HW]
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # [B, HW, C/r]
        out = self.linear_out(out)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        result = local_feat + out
        return self.output(result)


class WightAttendtion(nn.Module):

    def __init__(self, channels_in):
        super(WightAttendtion, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_in, channels_in // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels_in // 2, channels_in, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 2, 3, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(channels_in // 2, 1, 3, padding='same', bias=False),
            nn.Sigmoid()
        )
        self.output = nn.Conv2d(channels_in, 1, 1)

    def forward(self, local_feat, global_feat):
        chan_att = self.channel_attention(global_feat)
        spat_att = self.spatial_attention(global_feat)
        local_feat = chan_att * local_feat
        end_local = spat_att * local_feat
        return self.output(end_local)
