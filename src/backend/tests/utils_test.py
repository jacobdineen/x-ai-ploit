# -*- coding: utf-8 -*-
# test_utils.py
from src.backend import utils


def test_add_numbers():
    assert utils.add_numbers(2, 3) == 5
    assert utils.add_numbers(-1, 1) == 0
