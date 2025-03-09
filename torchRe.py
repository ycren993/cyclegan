import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, 256, 1)
        self.conv3x3_1 = nn.Conv2d(in_ch, 256, 3, padding=6, dilation=6)
        self.conv3x3_2 = nn.Conv2d(in_ch, 256, 3, padding=12, dilation=12)
        self.conv3x3_3 = nn.Conv2d(in_ch, 256, 3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Conv2d(4*256+3, 256, 1)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = F.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear')
        return self.final_conv(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))

if __name__ == '__main__':
    Aspp = ASPP(3)

    x = torch.randn(2, 3, 32, 32)
    print(Aspp(x).shape)