"""Config file for pytest"""

import pytest

# handle slow tests skipping
def pytest_addoption(parser):
    parser.addoption("--skipslow", action="store_true",
                     default=False, help="skip slow tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipslow"):
        # --runslow given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="--skipslow option was used")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)