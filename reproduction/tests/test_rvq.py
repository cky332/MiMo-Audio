import torch

from mimo_repro.rvq import ResidualVQ


def test_rvq_shapes_and_roundtrip():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=16, codebook_sizes=(64, 64, 16, 16))
    rvq.eval()
    x = torch.randn(100, 16)
    codes = rvq.encode(x)
    assert codes.shape == (4, 100)
    assert codes[0].max() < 64 and codes[2].max() < 16
    decoded = rvq.decode(codes)
    assert decoded.shape == x.shape
    # encode->decode->encode must be a fixed point
    codes2 = rvq.encode(decoded + (x - decoded))  # same x
    assert torch.equal(codes, codes2)


def test_rvq_residual_reduces_error():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=8, codebook_sizes=(32,) * 6)
    rvq.eval()
    x = torch.randn(200, 8)
    errs = []
    for n_q in (1, 3, 6):
        codes = rvq.encode(x, n_q=n_q)
        errs.append((rvq.decode(codes) - x).pow(2).mean().item())
    assert errs[0] > errs[1] > errs[2], f"residual stages should refine: {errs}"


def test_ema_updates_move_codebook():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=8, codebook_sizes=(16,))
    before = rvq.layers[0].embed.clone()
    rvq.train()
    for _ in range(5):
        rvq(torch.randn(64, 8))
    assert not torch.allclose(before, rvq.layers[0].embed)


def test_commit_loss_positive_and_grad_flows():
    torch.manual_seed(0)
    rvq = ResidualVQ(dim=8, codebook_sizes=(16, 16))
    rvq.train()
    lin = torch.nn.Linear(8, 8)
    x = lin(torch.randn(32, 8))
    quantized, codes, commit = rvq(x)
    assert commit.item() > 0
    # straight-through: gradient reaches the input projection
    quantized.sum().backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.abs().sum() > 0
