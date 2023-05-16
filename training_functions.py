import torch


def train_discriminator(optimizer, discriminator, criterion, real_images, fake_images):
    batch_size = real_images.size(0)
    
    # Train the discriminator
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    optimizer.zero_grad()
    
    # Compute discriminator loss on real images
    real_outputs = discriminator(real_images)
    d_loss_real = criterion(real_outputs, real_labels)
    
    # Compute discriminator loss on fake images
    fake_outputs = discriminator(fake_images.detach())
    d_loss_fake = criterion(fake_outputs, fake_labels)
    
    # Total discriminator loss
    d_loss = d_loss_real + d_loss_fake
    
    d_loss.backward()
    optimizer.step()
    
    return d_loss


def train_generator(optimizer, criterion, fake_images, fake_images_outputs):
    
    batch_size = fake_images.size(0)
    real_labels = torch.ones(batch_size, 1)

    
    # Compute generator loss
    g_loss = criterion(fake_images_outputs, real_labels)
    
    optimizer.zero_grad()
    g_loss.backward()
    optimizer.step()
    
    return g_loss