import argparse

import pytest

from overfit.__main__ import _resolve_run_options, positive_int
from overfit.config import Config


def test_quick_mode_uses_one_seed_and_fewer_epochs():
    cfg, seeds = _resolve_run_options(Config(), quick=True, seeds=None)

    assert seeds == (42,)
    assert cfg.epochs == 20


def test_quick_mode_accepts_one_custom_seed():
    _, seeds = _resolve_run_options(Config(), quick=True, seeds=[7])

    assert seeds == (7,)


def test_quick_mode_rejects_multiple_seeds():
    with pytest.raises(ValueError, match="exactly one seed"):
        _resolve_run_options(Config(), quick=True, seeds=[7, 8])


def test_normal_mode_keeps_five_seed_default():
    cfg, seeds = _resolve_run_options(Config(), quick=False, seeds=None)

    assert seeds == (42, 43, 44, 45, 46)
    assert cfg.epochs == Config.epochs


@pytest.mark.parametrize("value", ["0", "-1"])
def test_thread_count_must_be_positive(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_int(value)
