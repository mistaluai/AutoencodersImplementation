import torch
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from ConditionalVariationalAutoencoders.model import CVAE
from ConditionalVariationalAutoencoders.loss import VAELoss
from ConditionalVariationalAutoencoders.utils.plotter import plot_cvae_generations
from ConditionalVariationalAutoencoders.utils.trainer import Trainer

epochs = 30
batch_size = 64
device = 'mps'

data_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
data_val = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = VAELoss()

trainer = Trainer(model, criterion, optimizer, dataloaders, epochs, device)
trainer.train()

output_dir = "cvae_output"

plot_cvae_generations(model=model, device=device, num_images=10, output_dir=output_dir)