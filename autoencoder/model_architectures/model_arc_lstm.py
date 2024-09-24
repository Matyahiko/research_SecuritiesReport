import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, max_length):
        super(TextEncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_length = max_length

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # エンコーダー: 双方向LSTM
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)

        # エンコーダーの出力を潜在空間に変換する全結合層
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * max_length, latent_size * 10),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(latent_size * 10, latent_size * 5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(latent_size * 5, latent_size)
        )

    def forward(self, x):
        """
        エンコーダーのフォワードパス

        Args:
            x (Tensor): 入力テンソルの形状 (batch_size, seq_length)

        Returns:
            latent (Tensor): 潜在表現の形状 (batch_size, latent_size)
        """
        # 埋め込み
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)

        # エンコーダーLSTM
        encoder_output, (hidden, cell) = self.encoder_lstm(embedded)  # encoder_output: (batch, seq_length, hidden_size*2)

        # フラット化
        encoder_output_flat = encoder_output.contiguous().view(encoder_output.size(0), -1)  # (batch_size, hidden_size*2*seq_length)

        # 潜在表現
        latent = self.encoder_fc(encoder_output_flat)  # (batch_size, latent_size)

        return latent

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, max_length):
        super(TextDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_length = max_length

        # 潜在空間からデコーダーの入力形状に変換する全結合層
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_size, latent_size * 5),
            nn.ReLU(),
            nn.Linear(latent_size * 5, latent_size * 10),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(latent_size * 10, hidden_size * 2 * max_length),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # デコーダーLSTM
        self.decoder_lstm = nn.LSTM(hidden_size * 2, embed_size, batch_first=True)

        # 出力層
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, latent):
        """
        デコーダーのフォワードパス

        Args:
            latent (Tensor): 潜在表現の形状 (batch_size, latent_size)

        Returns:
            output (Tensor): 再構築された出力の形状 (batch_size, seq_length, vocab_size)
        """
        # 潜在ベクトルをデコーダー入力の形状に変換
        decoded = self.decoder_fc(latent)  # (batch_size, hidden_size*2*seq_length)
        decoded = decoded.view(decoded.size(0), self.max_length, self.hidden_size * 2)  # (batch_size, seq_length, hidden_size*2)

        # デコーダーLSTM
        decoder_output, (hidden, cell) = self.decoder_lstm(decoded)  # (batch_size, seq_length, embed_size)

        # 出力層
        output = self.output_layer(decoder_output)  # (batch_size, seq_length, vocab_size)

        return output
