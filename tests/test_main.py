import pytest
import sys

from gamutrf.__main__ import scan


sys.argv.append("-h")


def test_main_scan():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        scan()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
