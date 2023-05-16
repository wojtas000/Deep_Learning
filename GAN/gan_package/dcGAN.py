import torch
import torch.nn as nn

IMG_SHAPE = (3, 256, 256)
n_out = torch.prod(torch.tensor(IMG_SHAPE))


class ConvTransposedBlock(nn.Module):
    """
    Convolutional transpose block for DCGAN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            batch_norm (bool): If True, batch normalization is used.
        """
        super(ConvTransposedBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = nn.ReLU(True)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional block for DCGAN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.LeakyReLU(0.2, inplace=True), batch_norm=True, bias=True):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            padding (int): Padding.
            batch_norm (bool): If True, batch normalization is used.
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x


class DCGAN_Generator(nn.Module):
    """
    Generator for DCGAN.
    """
    def __init__(self, latent_dim=100, feature_maps_size=64, num_channels=3):
        """
        Args:
            latent_dim (int): Dimension of latent vector.
            feature_maps_size (int): Number of feature maps.
            num_channels (int): Number of channels of output image.
        """
        super(DCGAN_Generator, self).__init__()
        self.model = nn.Sequential(
            # input: latent_dim x 1 x 1
            ConvTransposedBlock(latent_dim, feature_maps_size * 32, 4, 1, 0),
            # (feature_maps_size*32) x 4 x 4
            ConvTransposedBlock(feature_maps_size * 32, feature_maps_size * 16, 4, 2, 1),
            # (feature_maps_size*16) x 8 x 8
            ConvTransposedBlock(feature_maps_size * 16, feature_maps_size * 8, 4, 2, 1),
            # (feature_maps_size*8) x 16 x 16
            ConvTransposedBlock(feature_maps_size * 8, feature_maps_size * 4, 4, 2, 1),
            # (feature_maps_size*4) x 32 x 32
            ConvTransposedBlock(feature_maps_size * 4, feature_maps_size * 2, 4, 2, 1),
            # (feature_maps_size*2) x 64 x 64
            ConvTransposedBlock(feature_maps_size * 2, feature_maps_size, 4, 2, 1),
            # (feature_maps_size) x 128 x 128
            nn.ConvTranspose2d(feature_maps_size, num_channels, 4, 2, 1),
            nn.Tanh()
            # (num_channels) x 256 x 256
        )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        return self.model(input)
    

class DCGAN_Discriminator(nn.Module):
    """
    Discriminator for DCGAN.
    """
    def __init__(self, num_channels, feature_maps_size, n_out=1):
        """
        Args:
            num_channels (int): Number of channels of input image.
            feature_maps_size (int): Number of feature maps.
            n_out (int): Number of output features.
        """
        super(DCGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input: 3x256x256
            ConvBlock(num_channels, feature_maps_size, 4, 2, 1, batch_norm=False),
            # (feature_maps_size) x 128 x 128
            ConvBlock(feature_maps_size, feature_maps_size * 2, 4, 2, 1),
            # (feature_maps_size*2) x 64 x 64
            ConvBlock(feature_maps_size * 2, feature_maps_size * 4, 4, 2, 1),
            # (feature_maps_size*4) x 32 x 32
            ConvBlock(feature_maps_size * 4, feature_maps_size * 8, 4, 2, 1),
            # (feature_maps_size*8) x 16 x 16
            ConvBlock(feature_maps_size * 8, feature_maps_size * 16, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(feature_maps_size * 16 * 8 * 8, n_out),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)