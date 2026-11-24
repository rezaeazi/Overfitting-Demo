import numpy as np
import pandas as pd
import pytest

from overfit.config import Config
from overfit.report import summarize
from overfit.validate import walk_forward


def test_walk_forward_windows_are_chronological_and_disjoint():
    folds = list(walk_forward(n=100, train_size=40, test_size=10))
    assert len(folds) == 6
    tests = [(t.start, t.stop) for _, t in folds]
    for (train, test), (prev_start, prev_stop) in zip(folds, tests):
        assert train.stop == test.start and train.stop - train.start == 40
    assert all(a_stop <= b_start for (_, a_stop), (b_start, _) in zip(tests, tests[1:]))
    assert tests[-1][1] <= 100


def test_walk_forward_empty_when_data_too_short():
    assert list(walk_forward(n=30, train_size=40, test_size=10)) == []


def test_summarize_gap():
    results = pd.DataFrame(
        {
            "variant": ["a", "a", "b", "b"],
            "train_acc": [1.0, 0.8, 0.6, 0.6],
            "test_acc": [0.5, 0.5, 0.55, 0.55],
            "baseline_acc": [0.5] * 4,
            "train_loss": [0.1] * 4,
            "test_loss": [0.9] * 4,
        }
    )
    summary = summarize(results)
    assert list(summary.index) == ["a", "b"]
    assert summary.loc["a", "gap"] == pytest.approx(0.4)
    assert summary.loc["b", "gap"] == pytest.approx(0.05)


def test_run_end_to_end_on_synthetic_data():
    pytest.importorskip("torch")
    from overfit.validate import run

    rng = np.random.default_rng(0)
    cfg = Config(train_size=80, test_size=20, epochs=20)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = (rng.random(200) > 0.5).astype(np.float32)
    results = run(x, y, cfg)
    n_folds = len(list(walk_forward(200, 80, 20)))
    assert len(results) == n_folds * 6
    assert results[["train_acc", "test_acc"]].apply(lambda c: c.between(0, 1).all()).all()


def test_run_rejects_short_data():
    pytest.importorskip("torch")
    from overfit.validate import run

    with pytest.raises(ValueError):
        run(np.zeros((10, 5), np.float32), np.zeros(10, np.float32), Config())
