import pytest

from overfit.runtime import RunAlreadyActiveError, exclusive_run_lock


def test_report_directory_lock_prevents_concurrent_runs(tmp_path):
    lock_path = tmp_path / "reports" / ".run.lock"

    with exclusive_run_lock(lock_path):
        with pytest.raises(RunAlreadyActiveError, match="already running"):
            with exclusive_run_lock(lock_path):
                pytest.fail("the second run must not acquire the lock")
