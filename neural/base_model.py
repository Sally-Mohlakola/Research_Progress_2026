import torch
import torch.nn as nn

class Model_T(nn.Module):
    """
    P(T | omega_i): a single scalar per incoming direction (3-7-7-1, sigmoid),
    matching Soh & Montazeri 2024 §5.2. The output layer must stay 1-wide:
    prepare_transmittance_data() emits targets of shape (N, 1), so a 3-wide
    output silently broadcasts against them during MSE -- training three
    channels to one scalar -- and then eval_model_t() flattens 3N values for
    N rays.
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.sigmoid
        self.sequential = nn.Sequential(
            nn.Linear(3, 7), nn.PReLU(7),
            nn.Linear(7, 7), nn.PReLU(7),
            nn.Linear(7, 1)
        )

    def forward(self, x):
        return self.activation(self.sequential(x))

class Model_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.exp
        self.sequential = nn.Sequential(
            nn.Linear( 6, 21), nn.PReLU(21),
            nn.Linear(21, 21), nn.PReLU(21),
            nn.Linear(21, 21), nn.PReLU(21),
            nn.Linear(21,  3)
        )

    def forward(self, x):
        return self.activation(self.sequential(x))