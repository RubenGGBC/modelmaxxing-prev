import sys

import pytest

# macOS es POSIX pero no admite RLIMIT_AS: este modo es solo para VMs Linux desechables.
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="MM_SANDBOX=subprocess solo en Linux (Kaggle/Colab)")


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.setenv("MM_SANDBOX", "subprocess")


def test_pass_fail_and_timeout():
    from model_maxxing.evaluation.runner import run_tests
    assert run_tests("x = 1", {"test_list": ["assert x == 1"]})[0]
    assert not run_tests("x = 1", {"test_list": ["assert x == 2"]})[0]
    assert run_tests("while True: pass", {"test_list": []}, timeout=2) == (False, "timeout")


def test_memory_limit():
    from model_maxxing.evaluation.runner import run_tests
    ok, err = run_tests("x = bytearray(3 * 1024**3)", {"test_list": []})
    assert not ok and "MemoryError" in err
