#model_arc_standard.py
#description: Autoencoder model architecture
import torch.nn as nn

class MlpAutoencoder(nn.Module):
    def __init__(self, inp_size, latent_size):
        super(MlpAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inp_size, latent_size*10),
            nn.ReLU(),
            nn.Linear(latent_size*10, latent_size*5),
            nn.ReLU(),
            nn.Linear(latent_size*5, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, latent_size*5),
            nn.ReLU(),
            nn.Linear(latent_size*5, latent_size*10),
            nn.ReLU(),
            nn.Linear(latent_size*10, inp_size)
        )

    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb)
        return x, emb