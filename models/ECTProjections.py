import torch
import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=8):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, **kwargs):
        # x shape: (batch_size, set_size, in_channels)
        # print(f'Started deep sets forward: {x.shape}')
        x = self.phi(x)  # Apply phi to each element
        # print(f'Done deep sets phi forward: {x.shape}')
        x = x.sum(dim=1)  # Sum over the set (dimension 1)
        # print(f'Done deep sets phi sum: {x.shape}')
        x = self.rho(x)  # Apply rho to pooled representation
        # print(f'Done deep sets {x.shape}')
        return x


class LinearProj(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten ECT before linear projection
            nn.Linear(input_size, output_size),
        )

    def forward(self, x, **kwargs):
        return self.layer(x)


class SetTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        d_model=16,
        nhead=1,
        num_layers=1,
        dim_feedforward=32,
        out_dim=10,
        pooling="sum",
        pe_dim=0,
    ):
        super().__init__()

        self.pe = bool(pe_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.input_proj = nn.Linear(feature_dim + pe_dim, d_model)
        self.pooling = pooling
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, out_dim)
        )

    def forward(self, x, directions=None, **kwargs):
        # x shape: (batch_size, num_elements, feature_dim)
        if self.pe:
            x = torch.cat([x, directions.expand(x.shape[0], -1, -1)], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x)

        # Pool over set dimension
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "sum":
            x = x.sum(dim=1)
        elif self.pooling == "max":
            x = x.max(dim=1).values
        else:
            raise ValueError("Invalid pooling method")

        return self.output_mlp(x)


class Conv1dEquivariant(nn.Module):
    def __init__(self, num_thetas, num_steps, output_size, hidden_size=8):
        super().__init__()

        d_model = 16
        self.equivariant = nn.Sequential(
            nn.Conv1d(num_thetas, hidden_size, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, d_model, 1),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(num_steps, num_steps),
            nn.ReLU(),
            nn.Linear(num_steps, output_size),
        )

    def forward(self, x, **kwargs):
        x = self.equivariant(x)
        x = x.mean(dim=1)
        x = self.output_mlp(x)
        return x
