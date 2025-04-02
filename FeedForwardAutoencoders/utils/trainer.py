import torch
from tqdm import tqdm
class Trainer:
    def __init__(self, model, criterion, optimizer, dataloaders, epochs, device):
        self.model, self.criterion, self.optimizer, self.dataloaders ,self.epochs, self.device = (
            model,
            criterion,
            optimizer,
            dataloaders,
            epochs,
            device
        )

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            train_loss = self.one_train_epoch()
            val_loss = self.one_val_epoch()
            print(f'epoch {epoch+1} [train_loss: {train_loss:0.4f} , val_loss: {val_loss:0.4f}]')

    def one_train_epoch(self):
        dataloader = self.dataloaders['train']
        optimizer = self.optimizer
        self.model.train()
        total_loss = 0

        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(self.device, dtype=torch.float32)
            optimizer.zero_grad()

            reconstructed_images = self.model(images)
            loss = self.criterion(reconstructed_images, images)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def one_val_epoch(self):
        dataloader = self.dataloaders['val']
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.view(images.size(0), -1).to(self.device, dtype=torch.float32)
                reconstructed_images = self.model(images)
                loss = self.criterion(reconstructed_images, images)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return avg_loss