from config import device, variant

import mitsuba as mi
import drjit as dr
# Respect a variant the caller has already chosen (e.g. a scalar_spectral
# test harness) instead of forcing the configured one.
if mi.variant() is None:
    mi.set_variant(variant)

import torch
import torch.nn as nn
    
# Adapted from https://github.com/mitsuba-renderer/mitsuba3/discussions/579#discussioncomment-5652452
def matmul(a: mi.TensorXf, b: mi.TensorXf) -> mi.TensorXf:
    # Check conditions
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[0]

    # Matrix sizes
    N = a.shape[0]
    M = b.shape[1]
    K = b.shape[0]

    # Indices of the final matrix c repeat K times
    i, j = dr.arange(mi.UInt, N), dr.arange(mi.UInt, M)
    i, j = dr.meshgrid(i, j, indexing='ij')
    i, j = dr.repeat(i, K), dr.repeat(j, K)

    # [0, 1, ..., K - 1] repeated N * M times
    offset = dr.tile(dr.arange(mi.UInt, K), N * M)

    # Compute [a[0][0] * b[0][0], a[0][1] * b[1][0], ... ]
    tmp = dr.gather(mi.Float, a.array, i * K + offset) * dr.gather(mi.Float, b.array, offset * M + j)

    # Compute [c[0][0], c[0][1], ... ]
    c = dr.zeros(mi.TensorXf, shape=(N, M))
    dr.scatter_reduce(dr.ReduceOp.Add, c.array, tmp, dr.repeat(dr.arange(mi.UInt, N * M), K))
    return c

class Linear:
    """
    y[o, n] = sum_i W[o, i] * x[i, n] + b[o]

    KNOWN LIMIT: `matmul` above forms the full out x in x lanes outer product
    as one temporary before reducing it. At 768x768x8spp that is 2.4 GB for a
    21x6 layer -- which works, and is the only reason the historical
    6-21-21-21-3 network was renderable -- and 188 GB for a 128x78 one, which
    is not. An accumulating rewrite that scaled as (out + in) x lanes instead
    was tried and is *not* the answer either: it matched torch at 4K lanes but
    segfaulted at 4.7M. The real fix is drjit.nn (CoopVec), which Dr.Jit 1.5
    ships for exactly this and which packs the weights instead of unrolling
    them per lane. Until that port happens, keep Model_M at its default width
    with no encoding.
    """

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return matmul(self.weight, x) + self.bias

class PReLU:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):
        y = dr.maximum(0.0, x) + self.weight * dr.minimum(0.0, x)
        return y
    
def encode_drjit(x, bands):
    """
    Dr.Jit twin of neural.base_model.encode_directions.

    Input and output are (features, N) -- the layout Linear.forward expects --
    where the torch side is (N, features). The row order below must stay
    identical to the torch concatenation order, or training and rendering
    silently optimise different functions. MiModelWrapper.test() compares the
    two end to end and will catch a divergence, so do not skip it after
    touching either.
    """
    d = x.shape[0]
    n = x.shape[1]
    out = dr.zeros(mi.TensorXf, shape=(d * (1 + 2 * bands), n))

    # A TensorXf row cannot be read with x[i] (that yields a tensor, and
    # tensor-to-row assignment is not a supported scatter). The buffer is
    # row-major (d, n), so gather row i out as a flat Float instead.
    col = dr.arange(mi.UInt32, n)
    rows = [dr.gather(mi.Float, x.array, mi.UInt32(i) * n + col)
            for i in range(d)]

    for i in range(d):
        out[i] = rows[i]

    r = d
    for b in range(bands):
        f = (2.0 ** b) * dr.pi
        for i in range(d):
            out[r] = dr.sin(f * rows[i]); r += 1
        for i in range(d):
            out[r] = dr.cos(f * rows[i]); r += 1
    return out


class MiModelWrapper():
    def __init__(self, torch_model, activation):
        self.torch_model = torch_model
        # Models that encode their inputs advertise it; 0 means feed raw.
        self.encode_bands = int(getattr(torch_model, 'ENCODE_BANDS', 0))

        self.activation = activation
        self.layers = []

        for layer in torch_model.sequential:
            if type(layer) == torch.nn.modules.linear.Linear:
                state_dict = layer.state_dict()
                weight, bias = state_dict['weight'], state_dict['bias']
                weight, bias = mi.TensorXf(weight.cpu().numpy()), mi.TensorXf(bias.unsqueeze(1).cpu().numpy())
                self.layers.append(Linear(weight, bias))

            elif type(layer) == torch.nn.modules.activation.PReLU:
                weight = layer.state_dict()['weight']
                weight = mi.TensorXf(weight.unsqueeze(1).cpu().numpy())
                self.layers.append(PReLU(weight))

            else:
                raise NotImplementedError(f"Layer type not supported: {type(layer)}")
            
        # This evaluation path unrolls every layer into lane-width arithmetic,
        # so its cost grows with the total weight count and it has no way to
        # stream. Measured on this project: a 6-21-21-21-3 net (~1.2K weights)
        # renders 768x768x8spp fine; the encoded 78-128-128-128-3 net (~44K)
        # exhausted system memory at 256x256 after ten minutes. Warn loudly
        # rather than let a long render take the machine down with it.
        n_weights = sum(int(l.weight.shape[0]) * int(l.weight.shape[1])
                        for l in self.layers if isinstance(l, Linear))
        if n_weights > 5000:
            print(f"  [warn] Model has {n_weights} weights. This Dr.Jit path "
                  f"unrolls them per lane and does not stream; above roughly "
                  f"5K it exhausts memory before finishing a frame. Port "
                  f"MiModelWrapper to drjit.nn (CoopVec) before rendering "
                  f"a network this size.")

        self.test()

    def forward(self, x):
        if self.encode_bands:
            x = encode_drjit(x, self.encode_bands)
        for layer in self.layers:
            x = layer.forward(x)
        return self.activation(x)
    
    def test(self, samples=42900):
        # With an encoding the first Linear is wider than the model's input,
        # so the probe has to be built at the *raw* input width.
        in_dim = int(getattr(self.torch_model, 'IN_DIM',
                             self.layers[0].weight.shape[1]))
        nn_in = torch.randn([samples, in_dim], device=device)

        torch_out = self.torch_model(nn_in)
        drjit_out = self.forward(mi.TensorXf(nn_in.cpu().numpy().T)).torch().T

        assert torch.allclose(torch_out, drjit_out, rtol=1e-03, atol=1e-03), f'{torch_out}\n{drjit_out}'
    


