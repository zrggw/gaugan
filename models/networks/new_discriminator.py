from jittor import nn
from models.networks import spectral_norm
import jittor


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__
        self.dis = nn.ModuleList
        output_channel = opt.semantic_nc + 1  # for N+1 loss
        self.channels = [
            3,
            128,
            128,
            256,
            256,
            512,
            512,
        ]  # The numbel of channel of the each layers of encoder
        self.body_up = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(
                residual_block_D(
                    self.channels[i], self.channels[i + 1], opt, -1, first=(i == 0)
                )
            )
        # decoder part
        self.body_up.append(
            residual_block_D(self.channels[-1], self.channels[-2], opt, 1)
        )
        for i in range(1, opt.num_res_blocks - 1):
            self.body_up.append(
                residual_block_D(
                    2 * self.channels[-1 - i], self.channels[-2 - i], opt, 1
                )
            )
        self.body_up.append(residual_block_D(2 * self.channels[1], 64, opt, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

        self.dis = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 64, 3, padding=1, stride=2),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 1, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 64, 3, padding=1, stride=2),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, False),
                    nn.Conv2d(64, 1, 3, padding=1),
                ),
            ]
        )

    def execute(self, input):
        x = input
        # encoder
        encoder_res = list()
        dis_list = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)

            if i == 3 or i == 5:
                dis_list.append(x)  # patch

        dis_score = [self.dis[0](dis_list[0]), self.dis[1](dis_list[1])]
        # decoder
        return_feats = list()
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](jittor.concat((encoder_res[-i - 1], x), dim=1))
            return_feats.append(x)
        ans = self.layer_up_last(x)
        return ans, dis_score, return_feats


class residual_block_D(nn.Module):
    def __init__(self, up_or_down, fin, fout, opt, first=False) -> None:
        super.__init__()
        self.up_or_down = up_or_down
        norm_layer = spectral_norm
        self.first = first
        self.learned_shortcut = fin != fout
        fmiddle = fout
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if up_or_down > 0:  # up
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(scale=0.2),
                    nn.Upsample(scale_factor=2),
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)),
                )
            else:  # down
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(scale=0.2),
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)),
                )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(scale=0.2), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1))
        )
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
            if self.up_or_down > 0:
                self.sampling = nn.Upsample(scale_factor=2)
            elif self.up_or_down < 0:
                self.sampling = nn.AvgPool2d(2)
            else:
                self.sampling = nn.Sequential()

    def shortcut(self, x):
        if self.first:
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        return x_s

    def execute(self, x):
        x = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x + dx
        return out
