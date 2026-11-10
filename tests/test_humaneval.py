import sys

import pytest

from model_maxxing.evaluation import humaneval as he

EX = {"task_id": "toy/0", "prompt": "def f(x):\n    \"\"\"suma uno\"\"\"\n", "entry_point": "f",
      "test": "\n\ndef check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n"}


def test_extract_stops_at_column_zero_constructs():
    assert he.extract_code("    return x + 1\n\ndef g():\n    pass") == "    return x + 1\n"
    assert he.extract_code("    return x + 1\n# fin") == "    return x + 1"
    assert he.extract_code("    y = x\n    # comentario indentado\n    return y + 1") == \
        "    y = x\n    # comentario indentado\n    return y + 1"


def test_exec_spec_builds_check_call():
    code, spec = he.exec_spec(EX, "    return x + 1\n")
    assert code.startswith("def f(x):") and spec["test_list"][-1] == "check(f)"


@pytest.mark.skipif(sys.platform != "darwin", reason="usa sandbox-exec (macOS); en Linux lo cubre la validación de Kaggle")
def test_run_pass_and_fail_in_sandbox():
    assert he.run(EX, "    return x + 1\n")[0]
    assert not he.run(EX, "    return x\n")[0]
