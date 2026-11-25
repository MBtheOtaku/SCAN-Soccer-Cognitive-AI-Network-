"""Reuse image registration without reusing a candidate or another recording."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.ball_tracking.detector_confirmed_recovery import FrameAlignmentCache


class Verifier:
    def __init__(self):
        self.calls = 0

    def align(self, image):
        self.calls += 1
        return np.eye(3), {'inlier_current_points': [[1.0, 2.0]], 'status': 'registered'}


def test_only_identical_frame_pixels_reuse_registration_and_results_are_isolated():
    verifier = Verifier()
    cache = FrameAlignmentCache(verifier)
    image = np.zeros((4, 5, 3), np.uint8)
    original = deepcopy(image)
    matrix, info = cache.align(7, image)
    matrix[0, 0] = 99
    info['inlier_current_points'][0][0] = 99
    matrix, info = cache.align(7, image.copy())
    assert verifier.calls == 1
    assert matrix[0, 0] == 1 and info['inlier_current_points'] == [[1, 2]]
    matrix[0, 0] = 98
    info['status'] = 'modified'
    assert cache.align(7, image)[1]['status'] == 'registered'
    np.testing.assert_array_equal(image, original)
    image[0, 0, 0] = 1
    cache.align(7, image)
    cache.align(8, image)
    cache.align(8, image.astype(np.uint16))
    cache.align(8, image.reshape((5, 4, 3)))
    assert verifier.calls == 5
    assert cache.configuration()['hits'] == 2
    assert cache.configuration()['misses'] == 5


def test_a_new_run_cannot_reuse_registration_from_another_verifier():
    image = np.zeros((4, 5, 3), np.uint8)
    first, second = Verifier(), Verifier()
    FrameAlignmentCache(first).align(7, image)
    FrameAlignmentCache(second).align(7, image)
    assert first.calls == second.calls == 1


def test_cache_is_bounded_and_discards_least_recently_used_registration():
    verifier = Verifier()
    cache = FrameAlignmentCache(verifier, maximum_entries=2)
    image = np.zeros((4, 5, 3), np.uint8)
    cache.align(1, image)
    cache.align(2, image)
    cache.align(1, image)
    cache.align(3, image)
    cache.align(2, image)
    assert verifier.calls == 4
    assert cache.configuration()['entries'] == 2


def test_registration_exception_is_not_cached_or_replaced_with_another_frame():
    class Failing:
        calls = 0
        def align(self, image):
            self.calls += 1
            raise RuntimeError('registration failed')
    verifier = Failing()
    cache = FrameAlignmentCache(verifier)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cache.align(7, np.zeros((4, 5, 3), np.uint8))
    assert verifier.calls == 2 and cache.configuration()['entries'] == 0


@pytest.mark.parametrize('limit', [0, 385, True, 2.5])
def test_unbounded_or_invalid_cache_is_rejected(limit):
    with pytest.raises(ValueError):
        FrameAlignmentCache(Verifier(), maximum_entries=limit)
