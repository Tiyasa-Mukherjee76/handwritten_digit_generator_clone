#trained model
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((self.label_emb(labels), noise), -1)
        return self.model(x).view(-1, 1, 28, 28)
