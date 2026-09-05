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

def encode_directions(x, bands):
    """
    Sinusoidal (Fourier) feature encoding of the six direction components.

    Layout, matching neural.drjit_wrapper.encode_drjit exactly:

        [ x , sin(f0*pi*x) , cos(f0*pi*x) , sin(f1*pi*x) , cos(f1*pi*x) , ... ]

    with f_b = 2**b, so the output width is D * (1 + 2*bands).

    Why this is here at all: an MLP fed raw coordinates is spectrally biased
    toward smooth functions (Tancik et al. 2020), and the function being
    fitted here is a diamond's exit distribution, whose lobes are a few
    degrees wide. Without an encoding the network cannot represent that
    structure at any width, so a resolution sweep would plateau on the
    network rather than on the method -- which would answer the wrong
    question.
    """
    out = [x]
    for b in range(bands):
        f = (2.0 ** b) * torch.pi
        out.append(torch.sin(f * x))
        out.append(torch.cos(f * x))
    return torch.cat(out, dim=-1)


class Model_M(nn.Module):
    """
    Multi-scatter lobe: f(wi, wo) -> RGB, trained on the RDM's depth>=3 term.

    The output nonlinearity is torch.exp, so `sequential` emits a log-radiance
    and `forward_pre` is what a log-space loss should regress. Keep
    `sequential` made of Linear and PReLU only: MiModelWrapper rebuilds the
    render-time network by walking those two layer types and rejects anything
    else, which is why the encoding is applied here in `forward` rather than
    being a module inside `sequential`.
    """

    IN_DIM = 6

    # Defaults are the render-safe configuration. `bands=6, width=128` fits
    # the RDM far better (median relative error 0.047 against 0.555) but
    # cannot be evaluated by neural/drjit_wrapper.py, which unrolls every
    # weight per lane and exhausts memory above roughly 5K weights. Raise
    # them only once MiModelWrapper is ported to drjit.nn (CoopVec).
    def __init__(self, width=21, bands=0):
        super().__init__()
        self.activation = torch.exp
        self.ENCODE_BANDS = int(bands)
        self.width = int(width)
        enc_dim = self.IN_DIM * (1 + 2 * self.ENCODE_BANDS)
        self.sequential = nn.Sequential(
            nn.Linear(enc_dim, width), nn.PReLU(width),
            nn.Linear(width, width), nn.PReLU(width),
            nn.Linear(width, width), nn.PReLU(width),
            nn.Linear(width, 3)
        )

    def arch(self):
        return {'width': self.width, 'bands': self.ENCODE_BANDS}

    def forward_pre(self, x):
        """Log-radiance, before the exp. The target of a log-space loss."""
        return self.sequential(encode_directions(x, self.ENCODE_BANDS))

    def forward(self, x):
        return self.activation(self.forward_pre(x))