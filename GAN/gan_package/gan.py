import torch.nn as nn
import torch
from tqdm import tqdm

class GAN:
    
    """
    GAN class.
    """

    def __init__(self, generator, discriminator):
        """
        Args:
            generator (nn.Module): Generator network.
            discriminator (nn.Module): Discriminator network.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.history = {'g_losses': [], 'd_losses': []}


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
            
            self.history['g_losses'].append(g_losses/len(dataloader.dataset))
            self.history['d_losses'].append(d_losses/len(dataloader.dataset))

            print(f'Epoch {epoch+1}/{num_epochs}: Generator loss: {g_loss/len(dataloader.dataset)}',  
                  f'Discriminator loss: {d_loss/len(dataloader.dataset)}')


    def save_generator(self, path):
        torch.save(self.generator.state_dict(), path)


    def save_discriminator(self, path):
        torch.save(self.discriminator.state_dict(), path)



if __name__=="__main__":
    
    from vanillaGAN import VanillaGAN_Generator, VanillaGAN_Discriminator
    from torchvision.datasets import MNIST
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim


    img_shape = (1, 256, 256)
    n_out = torch.prod(torch.tensor(img_shape))
    generator = VanillaGAN_Generator(latent_dim=100, img_shape=img_shape, n_out=n_out)
    discriminator = VanillaGAN_Discriminator(img_shape=(1, 256, 256))
    gan = GAN(generator=generator, discriminator=discriminator)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MNIST(root='data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    gan.train(dataloader=dataloader,
                discriminator_optimizer=discriminator_optimizer,
                generator_optimizer=generator_optimizer,
                criterion=criterion,
                num_epochs=100
                )