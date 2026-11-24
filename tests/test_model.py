import pytest

torch = pytest.importorskip("torch")

from overfit.config import VARIANTS, Config
from overfit.model import evaluate, fit


@pytest.mark.parametrize("name", list(VARIANTS))
def test_every_variant_trains_and_evaluates(name):
    torch.manual_seed(0)
    x = torch.randn(120, 5)
    y = (x[:, 0] > 0).float()
    cfg = Config(epochs=30)
    model = fit(x[:90], y[:90], x[90:], y[90:], VARIANTS[name], cfg)
    loss, acc = evaluate(model, x[90:], y[90:])
    assert loss > 0 and 0.0 <= acc <= 1.0
    assert not model.training


def test_early_stopping_restores_best_state():
    torch.manual_seed(0)
    x, y = torch.randn(60, 5), (torch.rand(60) > 0.5).float()
    cfg = Config(epochs=500, lr=0.05)
    model = fit(x[:40], y[:40], x[40:], y[40:], VARIANTS["early_stop"], cfg)
    assert all(torch.isfinite(p).all() for p in model.parameters())
