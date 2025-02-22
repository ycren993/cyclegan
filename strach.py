import torch
import torch.nn as nn



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class MultiDilationNet(nn.Module):
    def __init__(self, inner_nc, outer_nc,use_bias=False):
        super(MultiDilationNet, self).__init__()
        sub_inner_nc = inner_nc // 4
        sub_outer_nc = outer_nc // 4
        self.sub1 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=1, stride=1, padding=0, dilation=1,bias=use_bias)
        self.sub2 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=3, stride=1, padding=2, dilation=2,bias=use_bias)
        self.sub3 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=3, stride=1, padding=4, dilation=4,bias=use_bias)
        self.sub4 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=3, stride=1, padding=8, dilation=8,bias=use_bias)
        self.cbam = CBAM(sub_outer_nc)
        self.conv1x1 = nn.Conv2d(inner_nc, outer_nc,1,bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(inner_nc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        sub_cbam1 = self.cbam(self.sub1(x))
        sub_cbam2 = self.cbam(self.sub2(x))
        sub_cbam3 = self.cbam(self.sub3(x))
        sub_cbam4 = self.cbam(self.sub4(x))
        x = torch.cat([sub_cbam1, sub_cbam2, sub_cbam3, sub_cbam4],1)
        return x + self.relu(self.batch_norm(self.conv1x1(x)))
if __name__ == '__main__':
    x = torch.randn(1, 512, 640, 640)
    net = MultiDilationNet(512, 512)
    print(net(x).shape)