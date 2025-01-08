import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Dimensionality of the embeddings (must match the input embedding size).
            max_len: Maximum length of input sequences to support.
        """
        super().__init__()
        # Create a matrix to hold positional encodings
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # Scaling terms
        
        # Apply sine and cosine functions to even and odd dimensions
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Add a batch dimension for easy addition to inputs
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class EncoderDecoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim):
        super().__init__()
        """
        input_dim: size of the input vocabulary
        output_dim: size of the output vocabulary
        d_model: dimensional of internal embedding space
        num_heads: number of attention heads
        num_layers: number of layers for encoder/decoder blocks
        ff_dim: dimension of hidden layer in feed-forward network
        """
        self.encoder_embedding = nn.Embedding(input_dim, d_model)
        self.decoder_embedding = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # Source positional encoding
        src_emb = self.encoder_embedding(src)
        src_pos_emb = self.positional_encoding(src_emb)
        src_encoded = self.encoder(src_pos_emb)

        # Target positional encoding
        tgt_emb = self.decoder_embedding(tgt)
        tgt_pos_emb = self.positional_encoding(tgt_emb)
        tgt_decoded = self.decoder(tgt_pos_emb, src_encoded)

        return self.fc_out(tgt_decoded)