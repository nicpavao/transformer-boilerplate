import torch
import torch.nn as nn

    
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
        # first encode the input sequence
        src_emb = self.encoder_embedding(src)
        src_encoded = self.encoder(src_emb)

        # build a sequence of decoded output vectors
        tgt_emb = self.decoder_embedding(tgt)
        tgt_decoded = self.decoder(tgt_emb, src_encoded)

        # return logits for the output vector
        softmax = torch.nn.Softmax(dim=1)
        return softmax(self.fc_out(tgt_decoded))