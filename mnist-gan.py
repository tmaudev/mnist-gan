import os
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

LATENT_DIMENSION = 100

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.zeros_(model.bias)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            # Input is 100 x 1 x 1
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32 x 32 x 32

            # Reduce to expected 1 x 28 x 28
            nn.Conv2d(32, 1, 5),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            # Input is 1 x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 7 x 7
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.layers(x)
        return output.view(-1, 1).squeeze(1)

def generate_latent_input(device, batch_size):
    return torch.randn(batch_size, LATENT_DIMENSION, 1, 1, device=device)

def generate_fake_samples(device, generator, batch_size):
    X = generator(generate_latent_input(device, batch_size))
    y = torch.zeros((batch_size))
    return X, y

def train_loop(device, dataloader, discriminator, generator,
               loss_fn, discriminator_optimizer, generator_optimizer):
    size = len(dataloader.dataset) * 2

    for batch, (X_real, _) in enumerate(dataloader):
        y_real = torch.ones((len(X_real))).to(device)

        # Train Discriminator with Real Data
        discriminator_optimizer.zero_grad()

        real_discriminator_y = discriminator(X_real.to(device))
        real_discriminator_loss = loss_fn(real_discriminator_y, y_real)

        real_discriminator_loss.backward()

        # Train Discriminator with Fake Data
        X_fake, y_fake = generate_fake_samples(device, generator, len(X_real))
        y_fake = y_fake.to(device)
        fake_discriminator_y = discriminator(X_fake.detach())
        fake_discriminator_loss = loss_fn(fake_discriminator_y, y_fake)

        fake_discriminator_loss.backward()

        discriminator_loss = real_discriminator_loss + fake_discriminator_loss
        
        discriminator_optimizer.step()


        # Train Generator
        generator_optimizer.zero_grad()

        generator_discriminator_y = discriminator(X_fake)

        # For the generator, we want the discriminator to think the generated image is real.
        # Thus, we use a label of '1' (y_real)
        generator_loss = loss_fn(generator_discriminator_y, y_real)
        generator_loss.backward()
        generator_optimizer.step()

        if batch % 100 == 0:
            g_loss = generator_loss.item()
            d_loss = discriminator_loss.item()
            current = batch * len(X_real) * 2
            print(f"g_loss: {g_loss:>7f}  |  d_loss: {d_loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(device, dataloader, discriminator, generator, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X_real, _ in dataloader:
            X_fake, y_fake = generate_fake_samples(device, generator, len(X_real))
            y_real = torch.ones((len(X_real)))
            X = torch.cat((X_real, X_fake.to('cpu')))
            y = torch.cat((y_real, y_fake))

            X = X.to(device)
            y = y.to(device)
            pred = discriminator(X)
            test_loss += loss_fn(pred, y).item()
            pred_class = pred >= 0.5
            gen_pred_class = pred_class[X_real.size()[0]:]
            correct += gen_pred_class.type(torch.int).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n   Generator Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def visualize_results(device, dataloader, generator, num_examples):
    with torch.no_grad():
        X_real, _ = next(iter(test_dataloader))
        batch_size = dataloader.batch_size
        assert(num_examples <= batch_size)

        X_fake, y_fake = generate_fake_samples(device, generator, batch_size)
        y_real = torch.ones((batch_size))
        X = torch.cat((X_real, X_fake.to('cpu')))
        y = torch.cat((y_real, y_fake))

        X = X.to(device)
        y = y.to(device)
        pred = discriminator(X)
        pred_labels = pred >= 0.5

        fig = plt.figure(figsize=(10, 7))
        fig.canvas.manager.set_window_title("Test Examples")
        rows = 2
        columns = int(num_examples / rows)
        examples_per_row = int(num_examples / 2)
        img_count = 0
        for idx in range(examples_per_row):
            fig.add_subplot(rows, columns, img_count + 1)
            title = "Guess: %d, Actual: %d" % (pred_labels[idx], y[idx])
            plt.title(title)
            plt.axis('off')
            plt.imshow(X.to("cpu")[idx].squeeze(), cmap='gray')
            img_count += 1

        for idx in range(batch_size, batch_size + examples_per_row):
            fig.add_subplot(rows, columns, img_count + 1)
            title = "Guess: %d, Actual: %d" % (pred_labels[idx], y[idx])
            plt.title(title)
            plt.axis('off')
            plt.imshow(X.to("cpu")[idx].squeeze(), cmap='gray')
            img_count += 1

        plt.show()

# Add Gaussian noise to help avoid discriminator from overfitting
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == '__main__':
    # Run on GPU (on M1 Mac) if available
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
        AddGaussianNoise(0., 0.05),
    ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    generator = Generator().to(device)
    print(discriminator)
    print(generator)

    loss_fn = nn.BCELoss()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(device, train_dataloader, discriminator, generator,
                   loss_fn, discriminator_optimizer, generator_optimizer)
        test_loop(device, test_dataloader, discriminator, generator, loss_fn)

    visualize_results(device, test_dataloader, generator, 8)
