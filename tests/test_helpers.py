from time import sleep

import helpers


def test_percentage():
    assert helpers.percentage(100, 100) == 100
    assert helpers.percentage(25, 100) == 25
    assert helpers.percentage(25, 101) == 25


def test_timeit():
    assert helpers.timeit(lambda: sleep(0.1)) == 0.1
