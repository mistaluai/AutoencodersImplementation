import torch.nn.functional as F
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated, ground_truth, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(input=generated, target=ground_truth, reduction='sum')
        KL = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL