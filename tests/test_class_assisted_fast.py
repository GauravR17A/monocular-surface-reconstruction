import importlib.util
from pathlib import Path
import time

import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/train_class_assisted_height_fast.py'
spec=importlib.util.spec_from_file_location('fast_height',PATH)
fast=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)


def test_ordered_prefetch_preserves_order_and_bounds():
    def load(i):
        time.sleep((5-i)*.002)
        return i*2
    loader=fast.OrderedPrefetch(range(6),load,workers=3,depth=3)
    try:
        assert len(loader.queue) <= 3
        assert [loader.pop() for _ in range(6)] == [(i,i*2) for i in range(6)]
        with pytest.raises(StopIteration):
            loader.pop()
    finally:
        loader.close()


def test_worker_failure_is_not_silently_skipped():
    def bad(_):
        raise ValueError('invalid input')
    loader=fast.OrderedPrefetch([1],bad,workers=1,depth=1)
    try:
        with pytest.raises(ValueError,match='invalid input'):
            loader.pop()
    finally:
        loader.close()
