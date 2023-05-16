import torch
import torch.nn as nn


IMG_SHAPE = (3, 256, 256)
n_out = torch.prod(torch.tensor(IMG_SHAPE))

class LinearBlock(nn.Module):
    
    """
    Linear block for MLP.
    """
    
    def __init__(self, in_features, out_features, activation=False, batch_norm=True, dropout=False, p=0.5):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            activation (bool): If True, ReLU activation is used.
            batch_norm (bool): If True, batch normalization is used.
        """
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation if activation else None
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(p=p) if dropout else None
        
    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class VanillaGAN_Generator(nn.Module):
    
    """
    Generator for Vanilla GAN.
    """

    def __init__(self, latent_dim=100, img_shape=IMG_SHAPE, n_out=n_out):
        """
        Args:
            latent_dim (int): Dimension of latent vector.
            img_shape (tuple): Shape of real image.
            n_out (int): Number of output features.
        """
        
        super(VanillaGAN_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_out = n_out
        self.model = nn.Sequential(
            LinearBlock(latent_dim, 256, activation=nn.LeakyReLU(0.2), batch_norm=False),
            LinearBlock(256, 512, activation=nn.LeakyReLU(0.2), batch_norm=False),
            LinearBlock(512, 1024, activation=nn.LeakyReLU(0.2), batch_norm=False),
            LinearBlock(1024, 2048, activation=nn.LeakyReLU(0.2), batch_norm=False),
            LinearBlock(2048, torch.prod(torch.tensor(img_shape)), activation=nn.Tanh(), batch_norm=False)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, *self.img_shape)
        return x


class VanillaGAN_Discriminator(nn.Module):
    
    """
    Discriminator for Vanilla GAN.
    """

    def __init__(self, img_shape=IMG_SHAPE):
        """
        Args:
            img_shape (tuple): Shape of real image.
        """
        
        super(VanillaGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            LinearBlock(int(torch.prod(torch.tensor(img_shape))), 512, activation=nn.LeakyReLU(0.2), batch_norm=False, dropout=True, p=0.5),
            LinearBlock(512, 256, activation=nn.LeakyReLU(0.2), batch_norm=False, dropout=True, p=0.5),
            LinearBlock(256, 128, activation=nn.LeakyReLU(0.2), batch_norm=False, dropout=True, p=0.5),
            LinearBlock(128, 1, activation=nn.Sigmoid(), batch_norm=False, dropout=False)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    latent_dim=100
    num_channels=1
    batch_size=64
    img_shape = (1, 256, 256)
    n_out = int(torch.prod(torch.tensor(img_shape)))
    # generator = VanillaGAN_Generator(latent_dim=latent_dim, img_shape=img_shape, n_out=n_out)
    generator = VanillaGAN_Discriminator(img_shape=img_shape)
    # Assume input shape: (batch_size, latent_dim)
    input_shape = (batch_size, *img_shape)

    # Forward pass
    output = generator(torch.randn(*input_shape))

    # Get the output shape
    output_shape = output.shape

    print(output_shape)