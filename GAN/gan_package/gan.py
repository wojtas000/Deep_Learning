import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
import torch
from torch import nn
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from torchvision.transforms import ToTensor
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset

class GAN:
    
    """
    GAN class.
    """

    def __init__(self, generator, discriminator, inception_model):
        """
        Args:
            generator (nn.Module): Generator network.
            discriminator (nn.Module): Discriminator network.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.history = {'g_losses': [], 'd_losses': [], 'fid_scores': []}
        self.inception_model = inception_model
        self.inception_model.eval()

    def train_discriminator_step(self, optimizer, criterion, real_images, fake_images):
        batch_size = real_images.size(0)
        
        # Train the discriminator
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        optimizer.zero_grad()
        
        # Compute discriminator loss on real images
        real_outputs = self.discriminator(real_images.to(self.device))
        d_loss_real = criterion(real_outputs, real_labels)
        
        # Compute discriminator loss on fake images
        fake_outputs = self.discriminator(fake_images.detach().to(self.device))
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        d_loss.backward()
        optimizer.step()
        
        return d_loss.item()
    

    def train_generator_step(self, optimizer, criterion, fake_images, fake_images_outputs):
    
        batch_size = fake_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)

        # Compute generator loss
        g_loss = criterion(fake_images_outputs, real_labels)
        
        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()
        
        return g_loss.item()
    

    def train(self,
              dataset,
              dataloader,
              discriminator_optimizer, 
              generator_optimizer,
              criterion,
              num_epochs=100
              ):
        
        for epoch in range(num_epochs):
            
            g_losses = 0
            d_losses = 0
            
            for real_images, _ in tqdm(dataloader):
                real_images = real_images.to(self.device)
            
                batch_size = real_images.size(0)
                
                z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
                fake_images = self.generator(z)
                
                # Train the discriminator
                d_loss = self.train_discriminator_step(optimizer=discriminator_optimizer, 
                                                       criterion=criterion,
                                                       real_images=real_images, 
                                                       fake_images=fake_images)
                
                # Train the generator
                fake_images_outputs = self.discriminator(fake_images)
                
                g_loss = self.train_generator_step(optimizer=generator_optimizer, 
                                                   criterion=criterion, 
                                                   fake_images=fake_images, 
                                                   fake_images_outputs=fake_images_outputs)
                
                g_losses += g_loss
                d_losses += d_loss
            
            # Calculate FID score
            subset = Subset(dataset, np.random.choice(len(dataset), 100, replace=False))
            dataloader = DataLoader(subset, batch_size=100, shuffle=True)
            real_images = next(iter(dataloader))[0]
            fake_images = self.generator(torch.randn(100, 100))
            fid_score = self.calculate_fid_score(real_images, fake_images)
            
            self.history['fid_scores'].append(fid_score)
            self.history['g_losses'].append(g_losses/len(dataloader.dataset))
            self.history['d_losses'].append(d_losses/len(dataloader.dataset))

            print(f'Epoch {epoch+1}/{num_epochs}: Generator loss: {g_loss/len(dataloader.dataset)}',  
                  f'Discriminator loss: {d_loss/len(dataloader.dataset)}, FID score: {fid_score}')


    def save_generator(self, path):
        torch.save(self.generator.state_dict(), path)


    def save_discriminator(self, path):
        torch.save(self.discriminator.state_dict(), path)


    def calculate_fid_score(self, real_images, generated_images):

        real_activations = self.inception_model(real_images).detach().numpy()
        generated_activations = self.inception_model(generated_images).detach().numpy()

        # Compute mean and covariance of activations
        real_mean = np.mean(real_activations, axis=0)
        real_cov = GAN.cov(real_activations, rowvar=False)
        generated_mean = np.mean(generated_activations, axis=0)
        generated_cov = GAN.cov(generated_activations, rowvar=False)

        # Calculate FID score
        score = GAN.calculate_frechet_distance(real_mean, real_cov, generated_mean, generated_cov)

        return score

    @staticmethod
    def cov(x, rowvar=False):
        print(x.shape)
        mean = np.mean(x, axis=0)
        if rowvar:
            cov = np.matmul((x - mean).T, x - mean) / (x.shape[0] - 1)
        else:
            cov = np.matmul((x - mean).T, x - mean) / (x.shape[1] - 1)
        return cov

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        score = np.real(np.trace(sigma1 + sigma2 - 2 * covmean)) + np.sum(diff * diff)
        return score



if __name__=="__main__":
    
    from vanillaGAN import VanillaGAN_Generator, VanillaGAN_Discriminator
    from torchvision.datasets import MNIST
    from torchvision.transforms import transforms

    img_shape = (3, 256, 256)
    n_out = torch.prod(torch.tensor(img_shape))
    generator = VanillaGAN_Generator(latent_dim=100, img_shape=img_shape, n_out=n_out)
    discriminator = VanillaGAN_Discriminator(img_shape=img_shape)
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
    gan = GAN(generator=generator, discriminator=discriminator, inception_model=inception_model)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_size = 256
    root = 'GAN/lsun/bedroom/0/0/'
    dataset = torchvision.datasets.ImageFolder(root=root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # dataset = MNIST(root='data', download=True, transform=transform)
    subset = Subset(dataset, np.random.choice(len(dataset), 100, replace=False))
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)

    gan.train(    dataset=dataset,
                  dataloader=dataloader,
                  discriminator_optimizer=discriminator_optimizer,
                  generator_optimizer=generator_optimizer,
                  criterion=criterion,
                  num_epochs=5)
