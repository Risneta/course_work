"""Test model (tự bỏ qua nếu chưa cài torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import build_model, count_params  # noqa: E402

CFG = {"transformer": {"d_model": 32, "nhead": 4, "num_layers": 2, "dim_ff": 64, "dropout": 0.1, "pooling": "cls"},
       "lstm": {"hidden": 32, "num_layers": 2, "dropout": 0.1}}


@pytest.mark.parametrize("name", ["transformer", "lstm"])
def test_forward_shape_and_backward(name):
    m = build_model(name, CFG, n_features=7)
    x = torch.randn(5, 36, 7)
    out = m(x)
    assert out.shape == (5,)
    out.pow(2).mean().backward()
    assert count_params(m) > 0


def test_attention_shapes_rows_sum_to_one_and_matches_forward():
    m = build_model("transformer", CFG, n_features=7).eval()
    x = torch.randn(4, 36, 7)
    pred, attn = m.forward_with_attention(x)
    assert len(attn) == 2
    for a in attn:
        assert a.shape == (4, 4, 37, 37)  # (B, heads, L+1 vì có CLS, L+1)
        assert torch.allclose(a.sum(-1), torch.ones(4, 4, 37), atol=1e-4)
    assert torch.allclose(pred, m(x), atol=1e-5)
    assert all(not l.store_attn and l.last_attn is None for l in m.layers)  # đã dọn trạng thái


@pytest.mark.parametrize("pool", ["cls", "mean", "last"])
def test_pooling_variants(pool):
    cfg = {"transformer": {**CFG["transformer"], "pooling": pool}}
    m = build_model("transformer", cfg, n_features=7).eval()
    assert m(torch.randn(3, 36, 7)).shape == (3,)
    _, attn = m.forward_with_attention(torch.randn(3, 36, 7))
    assert attn[0].shape[-1] == (37 if pool == "cls" else 36)
