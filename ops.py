
import torch.nn as nn

import torch.nn.functional as F


def init_weights(modules):
    pass

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.LeakyReLU(),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

        
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out






pad, dilation, groups=4),
            nn.LeakyReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class GBasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(GBasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, groups=4),
            nn.Sigmoid()
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class ResidualBlock7(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock7, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 1, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 7, 1, 3),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out
        
class ResidualBlock3(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock3, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out
class ResidualBlock5(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock5, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out
class ResidualBlock1(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock1, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out

class GResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(GResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.leaky_relu(out + x)
        return out


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class ConvertBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 blocks):
        super(ConvertBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels * blocks, out_channels * blocks // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * blocks // 2, out_channels * blocks // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * blocks // 4, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        # out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out
