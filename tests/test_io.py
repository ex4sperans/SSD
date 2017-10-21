import os

import numpy as np
import pandas as pd
import pytest

import io_ops

DIRNAME = os.path.dirname(__file__)


def test_parse_annotation():
    """Test annotation parsing"""

    annotation = os.path.join(DIRNAME, "data", "000010.xml")
    bboxes = io_ops.parse_annotation(annotation)

    assert (bboxes.index == ["horse", "person"]).all()
    assert (bboxes.x_max == [258, 245]).all()


def test_find_files():

    files = list(io_ops.find_files(os.path.join(DIRNAME, "data"), "*.xml"))

    assert len(files) == 1
