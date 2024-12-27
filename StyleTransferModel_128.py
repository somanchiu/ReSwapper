import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, embedding_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-6)
        self.style = nn.Linear(embedding_dim, num_features * 2)

    def forward(self, x, s):
        """
        x: (B, C, H, W)
        s: (B, embedding_size)
        """
        style = self.style(s).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        out = self.norm(x)
        return gamma * out + beta


class StyleConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embedding_dim,
        activation,
    ):
        super().__init__()
        # self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaIN(embedding_dim, out_channels)
        self.activation = activation

    def forward(self, x, style):
        """
        x: latents (B, C, H, W)
        style: (B, embedding_size)
        """
        x = self.conv(x)
        x = self.adain(x, style)
        return self.activation(x)


class StyleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embedding_dim,
        activation,
    ):
        super().__init__()
        self.conv1 = StyleConvLayer(
            in_channels,
            out_channels,
            embedding_dim,
            activation,
        )
        self.conv2 = StyleConvLayer(
            out_channels,
            out_channels,
            embedding_dim,
            nn.Identity(),
        )

    def forward(self, x, style):
        """
        x: latents (B, C, H, W)
        style: (B, embedding_size)
        """
        x = self.conv1(x, style)
        x = self.conv2(x, style)
        return x


class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()

        # Encoder for target face
        self.down = nn.Sequential(
            nn.Conv2d(3, 128, (7, 7), padding="same", padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Style blocks
        self.style_blocks = nn.ModuleList([
            StyleBlock(1024, 1024, 512, nn.ReLU()) for i in range(6)
        ])

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1024, 512, (3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, (3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 3, (7, 7), padding="same", padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, target, source):
        # Encode target face
        x = self.down(target)

        for style_block in self.style_blocks:
            x = style_block(x, source) + x

        output = self.up(x)

        return (output + 1) / 2
