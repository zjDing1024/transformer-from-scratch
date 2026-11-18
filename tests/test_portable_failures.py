from unittest.mock import patch

import pytest

from transformer_diagnostics import cli, proc_memory


@pytest.mark.parametrize("error", [PermissionError(), FileNotFoundError(), UnicodeError()])
def test_missing_cpu_info_is_not_parity_or_memory_fatal(error):
    with (
        patch.object(cli.platform, "processor", return_value=""),
        patch.object(cli.sys, "platform", "linux"),
        patch.object(cli.Path, "read_text", side_effect=error),
    ):
        assert cli.cpu_description() == "unreported"


def test_non_utf8_proc_status_is_explicitly_unavailable():
    with (
        patch.object(proc_memory.sys, "platform", "linux"),
        patch.object(proc_memory.Path, "read_text", side_effect=UnicodeError),
    ):
        result = proc_memory.observe()
    assert result["VmRSS"]["status"] == "unavailable"
    assert result["VmHWM"]["status"] == "unavailable"
    assert result["VmRSS"]["reason"] == "UnicodeError"


def test_worker_timeout_is_bounded_without_inference():
    with patch.object(cli.subprocess, "run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "{}"
        assert cli.invoke("parity") == {}
        assert run.call_args.kwargs["timeout"] == 120
