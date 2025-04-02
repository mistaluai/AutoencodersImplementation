import torch.optim as optim
import torchvision.datasets as datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from FeedForwardAutoencoders.model import DenoisingAutoencoder
from FeedForwardAutoencoders.utils.plotter import plot_random_reconstructions
from FeedForwardAutoencoders.utils.trainer import Trainer

epochs = 20
batch_size = 64
device = 'mps'

data_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
data_val = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

# model = FFAutoencoder(in_features=784, latent_dim=96, hidden_dimensions=[384, 192]).to(device)
model = DenoisingAutoencoder(in_features=784, latent_dim=96, hidden_dimensions=[384, 192], noise_intensity=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

trainer = Trainer(model, criterion, optimizer, dataloaders, epochs, device)
trainer.train()

output_dir = "denoising_output"

plot_random_reconstructions(model=model, dataloader=val_dataloader, device=device, num_images=10, output_dir=output_dir)