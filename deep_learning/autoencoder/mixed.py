import torch
import torch.nn as nn
import torch.optim as optim


class OneLayerAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(OneLayerAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # We return both the reconstruction AND the hidden state
        encoded = self.activation(self.encoder(x))
        reconstructed = self.decoder(encoded)
        return reconstructed


class CompositeModel(nn.Module):
    def __init__(self, input_size, mlp1_hidden, ae_latent, mlp2_hidden, output_size):
        super(CompositeModel, self).__init__()

        self.mlp1 = nn.Sequential(nn.Linear(input_size, mlp1_hidden), nn.ReLU())

        self.autoencoder = OneLayerAutoencoder(mlp1_hidden, ae_latent)

        self.mlp2 = nn.Sequential(
            nn.Linear(mlp1_hidden, mlp2_hidden),
            nn.ReLU(),
            nn.Linear(mlp2_hidden, output_size),
        )

    def forward(self, x):
        ae_in = self.mlp1(x)
        ae_out = self.autoencoder(ae_in)
        final_output = self.mlp2(ae_out)

        # Return both the final prediction and the parts needed for AE loss
        return final_output, ae_in, ae_out


# --- Training with Dual Loss ---

model = CompositeModel(10, 32, 16, 32, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
task_criterion = nn.MSELoss()  # For the final output
ae_criterion = nn.MSELoss()  # For the reconstruction

# Dummy Data
inputs = torch.randn(16, 10)
targets = torch.randn(16, 1)

for epoch in range(100):
    optimizer.zero_grad()

    # 1. Forward Pass
    pred, ae_in, ae_out = model(inputs)

    # 2. Calculate Losses
    loss_task = task_criterion(pred, targets)
    loss_ae = ae_criterion(ae_out, ae_in)  # The "Real" AE part

    # 3. Combine Losses (you can weight these)
    total_loss = loss_task + loss_ae

    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1} | Task Loss: {loss_task.item():.4f} | AE Loss: {loss_ae.item():.4f}"
        )
