#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

from copy import deepcopy

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def detect_corners(image, shi_tomasi_params, cur_corners=None):
    if cur_corners is None:
        shi_tomasi_params["maxCorners"] = 1000
        new_corners = FrameCorners(np.array([]), np.array([]), np.array([]))
        mask = None
    else:
        new_corners = cur_corners
        shi_tomasi_params["maxCorners"] = max(1000 - len(cur_corners.ids), 0)
        mask = np.ones_like(image) * 255
        for p in cur_corners.points:
            cv2.circle(mask, np.int0(p), shi_tomasi_params["blockSize"], 0, -1)

    points = cv2.goodFeaturesToTrack(image, **shi_tomasi_params, mask=mask)
    if points is not None:
        new_corners.add_new_points(points, shi_tomasi_params["blockSize"])

    return new_corners


def track_corners(prev_image, cur_image, lukas_kanade_params, corners):
    (new_points, status, error) = cv2.calcOpticalFlowPyrLK(prev_image, cur_image, corners.points,
                                                           cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, **lukas_kanade_params)
    corners.update_points(new_points, status.flatten())
    return corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    shi_tomasi_params = dict(
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    lukas_kanade_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.01),
        minEigThreshold=5 * 10 ** (-4)
    )

    prev_image = np.uint8(frame_sequence[0] * 255.0)
    corners = detect_corners(prev_image, shi_tomasi_params)
    builder.set_corners_at_frame(0, deepcopy(corners))
    for frame, cur_image in enumerate(frame_sequence[1:], 1):
        cur_image = np.uint8(cur_image * 255.0)
        corners = track_corners(prev_image, cur_image, lukas_kanade_params, corners)
        if frame % 5 == 0:
            corners = detect_corners(cur_image, shi_tomasi_params, corners)

        builder.set_corners_at_frame(frame, deepcopy(corners))
        prev_image = cur_image


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
