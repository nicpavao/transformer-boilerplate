import torch.nn as nn
import torch

def causal_mask(tensor_input):
    seq_length = tensor_input.size(1)  # Assume tensor_input has shape (batch_size, seq_length)
    return torch.triu(torch.ones(seq_length,seq_length), diagonal=1) == 0

def padding_mask(tensor_input):
    return (tensor_input == 0)


class EncoderDecoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim, dropout = 0):
        super().__init__()

        # Encoder embedding and layer
        self.encoder_embedding = nn.Embedding(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)

        # Decoder embedding and layer
        self.decoder_embedding = nn.Embedding(output_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout)

        # Encoder/Decoder stack
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)


    def forward(self, src, tgt):
        # Source encoded with padding mask
        src_emb = self.encoder_embedding(src)
        src_padding_mask = padding_mask(src).to(src.device)
        src_encoded = self.encoder(src_emb)

        # Target decoded with causal & padding mask
        tgt_emb = self.decoder_embedding(tgt)
        tgt_padding_mask = padding_mask(tgt).to(src.device)
        tgt_causal_mask = causal_mask(tgt).to(src.device)
        tgt_decoded = self.decoder(tgt_emb, src_encoded)

        return self.fc_out(tgt_decoded)
    
    def train_model(self, data, criterion, lr, epochs, batch_size, device='cpu'):
        """
        Train the model with given data.
        
        Args:
            data: A dataset object that returns (src, tgt) pairs.
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

            for src, tgt in dataloader:
                # Move data to the specified device
                src, tgt = src.to(device), tgt.to(device)

                # Forward pass
                output = self(src)

                # Compute loss
                loss = criterion(output.squeeze(-1), tgt.float())

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

def predict(self, tgt, max_length, start_token, end_token, device="cpu"):
    """
    Generate predictions for a given sequence.

    Args:
        tgt: Initial target input tensor (batch_size, seq_length).
        max_length: Maximum length of the predicted sequence.
        start_token: Token indicating the start of decoding.
        end_token: Token indicating the end of decoding.
        device: Device to run prediction on ('cpu' or 'cuda').

    Returns:
        List of predicted sequences.
    """
    self.eval()  # Set model to evaluation mode
    tgt = tgt.to(device)
    batch_size = tgt.size(0)
    predictions = tgt

    for _ in range(max_length):
        output = self(predictions)
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(-1)  # Get the next token
        predictions = torch.cat((predictions, next_token), dim=1)

        # Check if all sequences have ended
        if (next_token == end_token).all():
            break

    return predictions

def save_checkpoint(self, path):
    """
    Save the model checkpoint.

    Args:
        path: File path to save the model.
    """
    torch.save(self.state_dict(), path)
    print(f"Model saved to {path}")

def load_checkpoint(self, path, device="cpu"):
    """
    Load the model checkpoint.

    Args:
        path: File path of the saved model.
        device: Device to load the model onto ('cpu' or 'cuda').
    """
    self.load_state_dict(torch.load(path, map_location=device))
    self.to(device)
    print(f"Model loaded from {path}")