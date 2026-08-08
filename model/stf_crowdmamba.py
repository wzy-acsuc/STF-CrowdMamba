import time
import torch
from torch import nn
from model.ms_stmb import MSSTMBBlock
from model.sfrb import SFRBBlock
from model.shiftvit_T import ShiftVitCount
from model.up import UpsampleModule
# from torchinfo import summary


class STFCrowdMamba(nn.Module):
    def __init__(self, is_train=True, fm_channel=768,
                 ms_stmb_depth_1=2, ms_stmb_depth_2=2,
                 sfrb_depth_1=1, sfrb_depth_2=1) -> None:
        super().__init__()
        stage_depths = (ms_stmb_depth_1, ms_stmb_depth_2, sfrb_depth_1, sfrb_depth_2)
        if any(depth < 1 for depth in stage_depths):
            raise ValueError("MS-STMB and SFRB stage depths must all be at least 1")
        self.shiftvit = ShiftVitCount(is_train=is_train)
        self.up_1 = UpsampleModule(fm_channel)
        self.dense_stb_1 = MSSTMBBlock(fm_channel // 2, num_blocks=ms_stmb_depth_1)
        self.frequency_enhance_1 = SFRBBlock(fm_channel // 2, depth=sfrb_depth_1)
        self.up_2 = UpsampleModule(fm_channel // 2)
        self.dense_stb_2 = MSSTMBBlock(fm_channel // 4, num_blocks=ms_stmb_depth_2)
        self.frequency_enhance_2 = SFRBBlock(fm_channel // 4, depth=sfrb_depth_2)
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

