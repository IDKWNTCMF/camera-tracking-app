#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    to_camera_center
)

import cv2

def triangulate_and_add_points(intrinsic_mat, corner_storage, view_mats, frame_1, frame_2, point_cloud_builder=None):
    triangulation_parameters = TriangulationParameters(
        max_reprojection_error=5.0,
        min_triangulation_angle_deg=1.0,
        min_depth=0.01
    )

    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    points3d, ids, _ = triangulate_correspondences(
        correspondences,
        view_mats[frame_1],
        view_mats[frame_2],
        intrinsic_mat,
        triangulation_parameters
    )

    if point_cloud_builder is None:
        return PointCloudBuilder(points=points3d, ids=ids)
    point_cloud_builder.add_points(points=points3d, ids=ids)
    point_cloud_builder._sort_data()
    return point_cloud_builder

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count

    frame_1 = known_view_1[0]
    frame_2 = known_view_2[0]
    view_mats[frame_1] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[frame_2] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = triangulate_and_add_points(intrinsic_mat, corner_storage, view_mats, frame_1, frame_2, None)

    frames_with_computed_camera_poses = set()
    frames_with_computed_camera_poses.add(frame_1)
    frames_with_computed_camera_poses.add(frame_2)
    frames_to_process = set()
    if frame_1 - 1 >= 0:
        frames_to_process.add(frame_1 - 1)
    if frame_1 + 1 < frame_count:
        frames_to_process.add(frame_1 + 1)
    if frame_2 - 1 >= 0:
        frames_to_process.add(frame_2 - 1)
    if frame_2 + 1 < frame_count:
        frames_to_process.add(frame_2 + 1)

    print(f"Size of point cloud - {len(point_cloud_builder.ids)}")
    while len(frames_to_process) > 0:
        max_intersection_ids = 0
        selected_frame = 0
        frames_to_remove = set()
        for cur_frame in frames_to_process:
            if cur_frame in frames_with_computed_camera_poses:
                frames_to_remove.add(cur_frame)
                continue
            ids = np.intersect1d(point_cloud_builder.ids, corner_storage[cur_frame].ids)
            if len(ids) > max_intersection_ids:
                max_intersection_ids = len(ids)
                selected_frame = cur_frame
        for cur_frame in frames_to_remove:
            frames_to_process.remove(cur_frame)
        if len(frames_to_process) == 0:
            break
        print(f"---------------------------------")
        print(f"Processing frame {selected_frame}")
        frames_to_process.remove(selected_frame)
        if selected_frame - 1 >= 0:
            frames_to_process.add(selected_frame - 1)
        if selected_frame + 1 < frame_count:
            frames_to_process.add(selected_frame + 1)
        ids, point_builder_indices, corners_indices = \
            np.intersect1d(point_cloud_builder.ids, corner_storage[selected_frame].ids, return_indices=True)
        points3d = point_cloud_builder.points[point_builder_indices]
        points2d = corner_storage[selected_frame].points[corners_indices]
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3d,
            imagePoints=points2d,
            cameraMatrix=intrinsic_mat,
            distCoeffs=np.array([]),
            reprojectionError=5.0
        )

        if retval:
            print(f"Frame {selected_frame} processed successfully, {len(inliers)} inliers found")
            view_mats[selected_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            outliers = np.delete(ids, inliers)
            indices_to_remove = []
            for i in range(len(point_cloud_builder.ids)):
                if point_cloud_builder.ids[i] in outliers:
                    indices_to_remove.append(i)
            point_cloud_builder.delete_points(indices_to_remove)
            print(f"{len(outliers)} outliers have been filtered")

            good_frames = 0
            for i in range(frame_count):
                frame = selected_frame - i
                if frame >= 0 and frame in frames_with_computed_camera_poses and np.linalg.norm(to_camera_center(view_mats[frame]) - to_camera_center(view_mats[selected_frame])) >= 0.2:
                    point_cloud_builder = triangulate_and_add_points(
                        intrinsic_mat,
                        corner_storage,
                        view_mats,
                        selected_frame,
                        frame,
                        point_cloud_builder
                    )
                    good_frames += 1

                frame = selected_frame + i
                if frame < frame_count and frame in frames_with_computed_camera_poses and np.linalg.norm(to_camera_center(view_mats[frame]) - to_camera_center(view_mats[selected_frame])) >= 0.2:
                    point_cloud_builder = triangulate_and_add_points(
                        intrinsic_mat,
                        corner_storage,
                        view_mats,
                        selected_frame,
                        frame,
                        point_cloud_builder
                    )
                    good_frames += 1

                if good_frames >= 20:
                    break

            frames_with_computed_camera_poses.add(selected_frame)
            print(f"Size of point cloud - {len(point_cloud_builder.ids)}")

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
