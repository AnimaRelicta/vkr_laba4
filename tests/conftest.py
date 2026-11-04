import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vkr.lab4.config import Config  # noqa: E402
from vkr.lab4.data import load_observations, split_frames  # noqa: E402
from vkr.lab4.demo_data import make_demo  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def demo_df(tmp_path_factory):
    d = make_demo(tmp_path_factory.mktemp("demo") / "out")
    return load_observations(d)


@pytest.fixture(scope="session")
def splits(demo_df):
    return split_frames(demo_df)
