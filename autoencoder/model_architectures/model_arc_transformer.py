# model_arc_transformer.py

import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        sequence_length: int, 
        latent_size: int, 
        num_encoder_layers: int = 3, 
        num_decoder_layers: int = 3, 
        nhead: int = 8, 
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.latent_size = latent_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Linear layer to project encoder output to latent space
        self.encoder_to_latent = nn.Linear(embedding_dim, latent_size)

        # Linear layer to project latent space back to embedding dimension
        self.latent_to_decoder = nn.Linear(latent_size, embedding_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length]
        Returns:
            logits: [batch_size, sequence_length, vocab_size]
            latent: [batch_size, latent_size]
        """
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.positional_encoding(embedded)  # Add positional encoding
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]

        encoded = self.encoder(embedded)  # [seq_len, batch_size, embedding_dim]
        # Aggregate encoded representations (e.g., mean pooling)
        encoded_mean = encoded.mean(dim=0)  # [batch_size, embedding_dim]
        latent = self.encoder_to_latent(encoded_mean)  # [batch_size, latent_size]

        # Project back to embedding dimension
        decoder_input = self.latent_to_decoder(latent).unsqueeze(0)  # [1, batch_size, embedding_dim]

        decoded = self.decoder(tgt=decoder_input, memory=encoded)  # [1, batch_size, embedding_dim]
        decoded = decoded.squeeze(0)  # [batch_size, embedding_dim]

        # Repeat decoded representation for each position
        decoded = decoded.unsqueeze(1).repeat(1, self.sequence_length, 1)  # [batch_size, seq_len, embedding_dim]

        logits = self.output_layer(decoded)  # [batch_size, seq_len, vocab_size]
        return logits, latent

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
