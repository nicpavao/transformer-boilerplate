import torch.nn as nn
import torch

def causal_mask(tensor_input):
    seq_length = tensor_input.size(1)  # Assume tensor_input has shape (batch_size, seq_length)
    return torch.triu(torch.ones(seq_length,seq_length), diagonal=1) == 0

def padding_mask(tensor_input):
    return (tensor_input == 0)


class DecoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim, dropout = 0):
        super().__init__()

        # Decoder embedding and layer
        self.decoder_embedding = nn.Embedding(output_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout)

        # Encoder/Decoder stack
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)


    def forward(self, src, tgt):

        # Target decoded with causal & padding mask
        tgt_emb = self.decoder_embedding(tgt)
        tgt_padding_mask = padding_mask(tgt).to(src.device)
        tgt_causal_mask = causal_mask(tgt).to(src.device)
        tgt_decoded = self.decoder(tgt_emb, memory = None)

        return self.fc_out(tgt_decoded)
    
    def train_model(self, data, criterion, lr, epochs, batch_size, device="cpu"):
        """
        Train the model with given data.

        Args:
            data: A dataset object that returns (tgt, tgt_labels) pairs.
            criterion: The loss function.
            lr: Learning rate.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            device: Device to train on ('cpu' or 'cuda').
        """
        # Optimizer
        optimizer = Adam(self.parameters(), lr=lr)

        # DataLoader for batching
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # Move model to the specified device
        self.to(device)

        for epoch in range(epochs):
            self.train()  # Set model to training mode
            epoch_loss = 0

            for tgt, tgt_labels in dataloader:
                # Move data to the specified device
                tgt, tgt_labels = tgt.to(device), tgt_labels.to(device)

                # Forward pass
                output = self(tgt)

                # Compute loss
                output_flat = output.view(-1, output.size(-1))  # Flatten for criterion
                tgt_labels_flat = tgt_labels.view(-1)  # Flatten target labels
                loss = criterion(output_flat, tgt_labels_flat)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")