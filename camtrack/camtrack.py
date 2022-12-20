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
    rodrigues_and_translation_to_view_mat3x4
)

import cv2

def compute_poses(corner_storage, intrinsic_mat, frame_1, frame_2):
    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    if correspondences.ids.shape[0] < 5:
        return False, None, None, 0

    E, inliers_mask = cv2.findEssentialMat(
        points1=correspondences.points_1,
        points2=correspondences.points_2,
        cameraMatrix=intrinsic_mat,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1,
        maxIters=2000
    )

    _, R, t, mask = cv2.recoverPose(
        E=E,
        points1=correspondences.points_1,
        points2=correspondences.points_2,
        cameraMatrix=intrinsic_mat,
        mask=inliers_mask
    )

    pose_1 = Pose(r_mat=np.eye(3, ), t_vec=np.zeros(3, ))
    pose_2 = view_mat3x4_to_pose(
        rodrigues_and_translation_to_view_mat3x4(cv2.Rodrigues(R)[0], t)
    )
    return True, pose_1, pose_2, mask

def find_initial_frames(corner_storage, intrinsic_mat):
    frame_count = len(corner_storage)
    frame_pairs = []
    cloud_sizes = []
    ratios = []

    step = 10 if frame_count >= 100 else 5
    diff = 30 if frame_count >= 100 else 5
    max_diff = 60 if frame_count >= 100 else 30
    max_corners = 0
    for frame in range(0, frame_count):
        max_corners = max(max_corners, corner_storage[frame].ids.shape[0])
    for frame_1 in range(0, min(120, frame_count), step):
        for frame_2 in range(frame_1 + diff, min(frame_count, frame_1 + max_diff), step):
            found, pose_1, pose_2, mask = compute_poses(corner_storage, intrinsic_mat, frame_1, frame_2)
            if not found:
                continue

            _, ids, median_cos, mean_cos = do_triangulate_correspondences(
                intrinsic_mat,
                corner_storage,
                pose_to_view_mat3x4(pose_1),
                pose_to_view_mat3x4(pose_2),
                frame_1,
                frame_2
            )

            # ratio = ids.shape[0] / max_corners # min(corner_storage[frame_1].ids.shape[0], corner_storage[frame_2].ids.shape[0])
            ratio = np.count_nonzero(mask) / max_corners
            print(f"Frame 1: {frame_1}, frame 2: {frame_2}, median cos: {median_cos}, mean cos: {mean_cos}, cnt: {ids.shape[0]}, ratio: {ratio}")
            if (frame_count >= 100 and mean_cos < 0.999 and ratio > 0.3) or (frame_count < 100 and mean_cos < 0.9995 and ratio > 0.3): # or (frame_count < 100 and median_cos < 0.9995 and ratio > 0.4):
                return frame_1, frame_2, pose_1, pose_2

            if (frame_count < 100 and median_cos < 0.995) or (frame_count >= 100 and median_cos < 0.999):
                frame_pairs.append((frame_1, frame_2))
                cloud_sizes.append(ids.shape[0])
                ratios.append(ratio)

    frame_1, frame_2 = frame_pairs[np.argmax(cloud_sizes)]
    _, pose_1, pose_2, _ = compute_poses(corner_storage, intrinsic_mat, frame_1, frame_2)

    return frame_1, frame_2, pose_1, pose_2

def add_frames_to_dicts(point_id_to_frames, point_id_to_projections, counters, corners, frame, frame_count):
    ids_to_retriangulate = set()
    for idx in range(corners.ids.shape[0]):
        corner_id = corners.ids[idx][0]
        if corner_id not in point_id_to_frames.keys():
            point_id_to_frames[corner_id] = [frame]
            point_id_to_projections[corner_id] = [corners.points[idx]]
            counters[corner_id] = 1
        else:
            counters[corner_id] += 1
            point_id_to_frames[corner_id].append(frame)
            point_id_to_projections[corner_id].append(corners.points[idx])
            if frame_count < 100 or counters[corner_id] < 10 or counters[corner_id] % 5 == 0:
                ids_to_retriangulate.add(corner_id)
    return point_id_to_frames, point_id_to_projections, counters, ids_to_retriangulate

def do_triangulate_correspondences(intrinsic_mat, corner_storage, view_mat_1, view_mat_2, frame_1, frame_2):
    triangulation_parameters = TriangulationParameters(
        max_reprojection_error=5.0,
        min_triangulation_angle_deg=0.7,
        min_depth=0.7
    )

    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])

    return triangulate_correspondences(
        correspondences,
        view_mat_1,
        view_mat_2,
        intrinsic_mat,
        triangulation_parameters
    )

def select_frame(frames_to_process, frames_with_computed_camera_poses, point_cloud_builder, corner_storage):
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
    return selected_frame, frames_to_process

def retriangulate_point_by_several_frames(point_projections, frames, view_mats, intrinsic_mat):
    equations = []
    for idx, point_2d in enumerate(point_projections):
        mat = intrinsic_mat @ view_mats[frames[idx]]
        equations.append(mat[2] * point_2d[0] - mat[0])
        equations.append(mat[2] * point_2d[1] - mat[1])
    equations = np.array(equations)
    a = equations[:, :3]
    b = (-1) * equations[:, 3]
    return np.linalg.lstsq(a, b, rcond=None)[0]

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        selected_frame_1, selected_frame_2, pose_1, pose_2 = find_initial_frames(corner_storage, intrinsic_mat)
        print(selected_frame_1, selected_frame_2)
        known_view_1 = (selected_frame_1, pose_1)
        known_view_2 = (selected_frame_2, pose_2)

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    point_id_to_frames, point_id_to_projections, counters = dict(), dict(), dict()

    frame_1 = known_view_1[0]
    frame_2 = known_view_2[0]
    view_mats[frame_1] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[frame_2] = pose_to_view_mat3x4(known_view_2[1])
    point_id_to_frames, point_id_to_projections, counters, _ = add_frames_to_dicts(
        point_id_to_frames,
        point_id_to_projections,
        counters,
        corner_storage[frame_1],
        frame_1,
        frame_count
    )
    point_id_to_frames, point_id_to_projections, counters, _ = add_frames_to_dicts(
        point_id_to_frames,
        point_id_to_projections,
        counters,
        corner_storage[frame_2],
        frame_2,
        frame_count
    )

    points3d, ids, _, _ = do_triangulate_correspondences(
        intrinsic_mat,
        corner_storage,
        view_mats[frame_1],
        view_mats[frame_2],
        frame_1,
        frame_2
    )
    point_cloud_builder = PointCloudBuilder(points=points3d, ids=ids)

    frames_with_computed_camera_poses, frames_to_process = set(), set()
    frames_with_computed_camera_poses.add(frame_1)
    frames_with_computed_camera_poses.add(frame_2)

    print(f"Size of point cloud - {len(point_cloud_builder.ids)}")
    last_len = 0
    while len(frames_with_computed_camera_poses) < frame_count:
        frames_to_process = set(np.arange(frame_count))
        if last_len == len(frames_with_computed_camera_poses):
            print("---------------------------------")
            selected_frame, frames_to_process = select_frame(
                frames_to_process,
                frames_with_computed_camera_poses,
                point_cloud_builder,
                corner_storage
            )
            print(f"Selecting frame to copy view mat for frame {selected_frame}")
            frame_to_copy = frame_1
            for frame in frames_with_computed_camera_poses:
                if abs(frame - selected_frame) < abs(frame_to_copy - selected_frame):
                    frame_to_copy = frame
            print(f"Copy view mat for frame {selected_frame} from view mat for frame {frame_to_copy}")
            view_mats[selected_frame] = view_mats[frame_to_copy]
            frames_with_computed_camera_poses.add(selected_frame)
            continue

        last_len = len(frames_with_computed_camera_poses)

        while len(frames_to_process) > 0:
            selected_frame, frames_to_process = select_frame(
                frames_to_process,
                frames_with_computed_camera_poses,
                point_cloud_builder,
                corner_storage
            )
            print("---------------------------------")

            if len(frames_to_process) == 0:
                break

            print(f"Processing frame {selected_frame}")
            frames_to_process.remove(selected_frame)
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
                _, new_rvec, new_tvec = cv2.solvePnP(
                    objectPoints=points3d[inliers],
                    imagePoints=points2d[inliers],
                    cameraMatrix=intrinsic_mat,
                    rvec=rvec,
                    tvec=tvec,
                    distCoeffs=np.array([]),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                view_mats[selected_frame] = rodrigues_and_translation_to_view_mat3x4(new_rvec, new_tvec)
                point_id_to_frames, point_id_to_projections, counters, ids_to_retriangulate = add_frames_to_dicts(
                    point_id_to_frames,
                    point_id_to_projections,
                    counters,
                    corner_storage[selected_frame],
                    selected_frame,
                    frame_count
                )
                ids_to_update, points_3d_to_update = [], []
                ids_to_add, points_3d_to_add = [], []
                for point_id in ids_to_retriangulate:
                    point_3d = retriangulate_point_by_several_frames(
                        point_id_to_projections[point_id],
                        point_id_to_frames[point_id],
                        view_mats,
                        intrinsic_mat
                    )
                    if point_id in point_cloud_builder.ids:
                        ids_to_update.append(point_id)
                        points_3d_to_update.append(point_3d)
                    else:
                        ids_to_add.append(point_id)
                        points_3d_to_add.append(point_3d)
                if len(ids_to_add) > 0:
                    point_cloud_builder.add_points(ids=np.array(ids_to_add), points=np.array(points_3d_to_add))
                    print(f"{len(ids_to_add)} points added to cloud")
                if len(ids_to_update) > 0:
                    point_cloud_builder.update_points(ids=np.array(ids_to_update), points=np.array(points_3d_to_update))
                    print(f"{len(ids_to_update)} points updated")
                frames_with_computed_camera_poses.add(selected_frame)
                print(f"Size of point cloud - {len(point_cloud_builder.ids)}")
            else:
                print("Failed to solve PnP Ransac")

    print("All frames processed successfully")
    print("Begin final reprocessing of all frames without point cloud recalculation")
    for frame in range(frame_count):
        print("---------------------------------")
        ids, point_builder_indices, corners_indices = \
            np.intersect1d(point_cloud_builder.ids, corner_storage[frame].ids, return_indices=True)
        points3d = point_cloud_builder.points[point_builder_indices]
        points2d = corner_storage[frame].points[corners_indices]
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3d,
            imagePoints=points2d,
            cameraMatrix=intrinsic_mat,
            distCoeffs=np.array([]),
            reprojectionError=5.0
        )

        if retval:
            print(f"View mat for frame {frame} was adjusted")
            _, new_rvec, new_tvec = cv2.solvePnP(
                objectPoints=points3d[inliers],
                imagePoints=points2d[inliers],
                cameraMatrix=intrinsic_mat,
                rvec=rvec,
                tvec=tvec,
                distCoeffs=np.array([]),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(new_rvec, new_tvec)
        else:
            print(f"View mat for frame {frame} was not adjusted")

    print("End of camera tracking")

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
