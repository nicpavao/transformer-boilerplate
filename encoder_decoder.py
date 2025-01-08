import torch
import torch.nn as nn

def pos_enc(x):
    """
    Applies positional encoding to the input tensor.
    Args:
        x: Input tensor of shape (batch_size, seq_length, d_model).
    Returns:
        Tensor of the same shape with positional encodings added.
    """
    batch_size, seq_length, d_model = x.size()
    position = torch.arange(0, seq_length, device=x.device).unsqueeze(1)  # Shape: (seq_length, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * -(torch.log(torch.tensor(10000.0)) / d_model))
    
    # Compute sinusoidal positional encoding
    pos_enc_matrix = torch.zeros(seq_length, d_model, device=x.device)
    pos_enc_matrix[:, 0::2] = torch.sin(position * div_term)
    pos_enc_matrix[:, 1::2] = torch.cos(position * div_term)
    
    # Add batch dimension and apply to input
    pos_enc_matrix = pos_enc_matrix.unsqueeze(0)  # Shape: (1, seq_length, d_model)
    return x + pos_enc_matrix

    
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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim)
        self.encoder_embedding = nn.Embedding(input_dim, d_model)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim)
        self.decoder_embedding = nn.Embedding(output_dim, d_model)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoderLayer(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # Apply embeddings and positional encoding for the source
        src_emb = self.encoder_embedding(src)
        src_pos_emb = pos_enc(src_emb)
        src_encoded = self.encoder(src_pos_emb)

        # Apply embeddings and positional encoding for the target
        tgt_emb = self.decoder_embedding(tgt)
        tgt_pos_emb = pos_enc(tgt_emb)
        tgt_decoded = self.decoder(tgt_pos_emb, src_encoded)

        # return logits for the output vector
        softmax = torch.nn.Softmax(dim=1)
        return softmax(self.fc_out(tgt_decoded))