#model_arc_standard_emb.py

import torch
import torch.nn as nn

class MlpAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, latent_size):
        super(MlpAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * sequence_length, latent_size * 10),
            nn.ReLU(),
            nn.Linear(latent_size * 10, latent_size * 5),
            nn.ReLU(),
            nn.Linear(latent_size * 5, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, latent_size * 5),
            nn.ReLU(),
            nn.Linear(latent_size * 5, latent_size * 10),
            nn.ReLU(),
            nn.Linear(latent_size * 10, embedding_dim * sequence_length)
            # 出力層をトークンの確率分布に変更
        )
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x).view(x.size(0), -1)  # [batch_size, sequence_length * embedding_dim]
        latent = self.encoder(emb)  # [batch_size, latent_size]
        decoded = self.decoder(latent)  # [batch_size, sequence_length * embedding_dim]
        decoded = decoded.view(x.size(0), x.size(1), -1)  # [batch_size, sequence_length, embedding_dim]
        logits = self.output_layer(decoded)  # [batch_size, sequence_length, vocab_size]
        return logits, latent

