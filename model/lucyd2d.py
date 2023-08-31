import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class RL_DIV(nn.Module):
    def __init__(self, channel):
        super(RL_DIV, self).__init__()
        self.conv1 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(1, channel, kernel_size=3, stride=1, relu=True)

    def forward(self, x, z):
        z0 = self.conv1(z)
        z1 = torch.mean(z0, dim=1, keepdim=True)
        q =  x/(z1 + 1e-12)
        z2 = self.conv2(q)

        return z2


class LUCYD2d(nn.Module):
    def __init__(self, num_res=1, in_channels=1):
        super(LUCYD2d, self).__init__()

        base_channel = 4

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
        ])

        self.correction_branch = nn.ModuleList([
            BasicConv(in_channels, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, in_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.update_branch = nn.ModuleList([
            BasicConv(in_channels, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            RL_DIV(base_channel),
            BasicConv(base_channel*2, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel, kernel_size=3, relu=True, stride=1),
        ])

        self.bottleneck = nn.ModuleList([
            BasicConv(base_channel*4, base_channel*2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel*2, base_channel*2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel*3, base_channel),
            AFF(base_channel*3, base_channel*2),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res),
            DBlock(base_channel, num_res),
        ])

        self.up = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )


    def forward(self, x):
        a1 = self.correction_branch[0](x)
        a2 = self.Encoder[0](a1)
        a3 = self.correction_branch[1](a2)

        b1 = self.update_branch[0](x)
        b2 = self.Encoder[1](b1)
        b3 = self.update_branch[1](b2)

        z0 = torch.cat([a3, b3], dim=1)
        z1 = self.bottleneck[0](z0)
        z2 = self.Encoder[2](z1)

        az = F.interpolate(a2, scale_factor=0.5)
        za = F.interpolate(z2, scale_factor=2)

        res1 = self.AFFs[0](a2, za)
        res2 = self.AFFs[1](z2, az)

        z3 = self.bottleneck[1](res2)
        z4 = self.Decoder[0](z3)
        zT = self.bottleneck[2](z4)


        a_ = torch.cat([res1, zT], dim=1)
        a4 = self.correction_branch[2](a_)
        a5 = self.Decoder[1](a4)
        cor = self.correction_branch[3](a5)

        y_k = x + cor

        b4 = self.update_branch[2](x, b2) # RL division

        b_ = torch.cat([b4, zT], dim=1)

        b5 = self.update_branch[3](b_)
        b6 = self.Decoder[2](b5 + b4)
        up = self.update_branch[4](b6)

        up = torch.mean(up, dim=1, keepdim=True)

        y = self.up(y_k * up)

        return y, y_k, up