from torch import nn

def padding_mask(tensor_input):
    return (tensor_input == 0)


class EncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim, dropout = 0):
        super().__init__()

        # Encoder embedding and layer
        self.encoder_embedding = nn.Embedding(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)

        # Encoder/Decoder stack
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.fc_out_model = nn.Linear(d_model, output_dim)
        self.fc_out_seq = nn.Linear(d_model, 1)


    def forward(self, src, tgt):
        # Source encoded with padding mask
        src_emb = self.encoder_embedding(src)
        src_encoded = self.encoder(src_emb)
        tgt_compressed = self.fc_out_model(src_encoded)

        return self.fc_out_seq(tgt_compressed.transpose(0,1))
    
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
