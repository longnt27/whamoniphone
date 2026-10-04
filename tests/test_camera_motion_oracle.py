"""Regression tests for WHAM-compatible 3DPW camera motion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "evaluation"))

from camera_motion_oracle import (
    camera_angular_velocity,
    camera_angular_velocity_with_validity,
    first_frame_aligned_root_errors,
    first_frame_aligned_world_joint_error,
    masked_first_valid_root_errors,
    rollout_root_translation,
)


class CameraMotionOracleTests(unittest.TestCase):
    def test_stationary_camera_has_zero_signal(self) -> None:
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        result = camera_angular_velocity(cameras)
        self.assertEqual(result.shape, (2, 6))
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_ninety_degree_z_rotation_matches_official_wham_convention(self) -> None:
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        cameras[1, :3, :3] = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
        )
        cameras[1, :3, 3] = [100, -20, 3]
        result = camera_angular_velocity(cameras)
        np.testing.assert_allclose(
            result[0], np.array([-30, 30, 0, -30, -30, 0]), atol=1e-5
        )

    def test_rejects_wrong_pose_count_and_nonfinite_rotation(self) -> None:
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        with self.assertRaisesRegex(ValueError, "expected 3 camera poses"):
            camera_angular_velocity(cameras, recurrent_frames=2)
        cameras[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            camera_angular_velocity(cameras)

    def test_full_track_retains_every_recurrent_frame_with_invalid_transition_zeroed(self) -> None:
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
        cameras[1, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        cameras[2, :3, :3] = np.nan  # Invalid frame must not contaminate output.
        frames = np.array([4, 5, 6, 7])
        valid = np.array([1, 1, 0, 1], dtype=bool)
        angular, usable = camera_angular_velocity_with_validity(cameras, frames, valid)
        self.assertEqual(angular.shape, (3, 6))
        np.testing.assert_array_equal(usable, [True, False, False])
        np.testing.assert_allclose(angular[0], [-30, 30, 0, -30, -30, 0])
        np.testing.assert_allclose(angular[1:], 0)

    def test_full_track_marks_frame_gaps_without_dropping_other_frames(self) -> None:
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
        angular, usable = camera_angular_velocity_with_validity(
            cameras, np.array([10, 11, 14, 15]), np.ones(4, bool)
        )
        np.testing.assert_array_equal(usable, [True, False, True])
        self.assertEqual(len(angular), 3)

    def test_world_root_scoring_aligns_at_first_valid_frame_and_masks_invalid(self) -> None:
        identity = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        gt = np.array([[100, 0, 0], [10, 0, 0], [11, 0, 0], [99, 0, 0]], dtype=np.float32)
        angular, displacement = masked_first_valid_root_errors(
            identity, pred, identity, gt, np.array([False, True, True, False])
        )
        np.testing.assert_allclose(angular, [0, 0], atol=1e-4)
        np.testing.assert_allclose(displacement, [0, 0], atol=1e-4)

    def test_world_joint_error_scores_rigid_first_frame_alignment_without_rescaling(self) -> None:
        identity = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
        quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        gt_root = np.repeat(quarter_turn[None], 2, axis=0)
        pred_joints = np.array([
            [[0, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [2, 0, 0]],
        ], dtype=np.float32)
        gt_joints = np.array([
            [[10, 20, 0], [10, 21, 0]],
            [[10, 21, 0], [10, 22, 0]],
        ], dtype=np.float32)
        errors = first_frame_aligned_world_joint_error(
            pred_joints, gt_joints, identity, gt_root, np.ones(2, bool)
        )
        np.testing.assert_allclose(errors, [0, 0], atol=1e-4)
        # A 0.5-m body-size error must remain after rigid alignment.
        pred_joints[1, 1, 0] += 0.5
        errors = first_frame_aligned_world_joint_error(
            pred_joints, gt_joints, identity, gt_root, np.ones(2, bool)
        )
        np.testing.assert_allclose(errors, [0, 250], atol=1e-4)

    def test_root_metrics_remove_only_initial_coordinate_offset(self) -> None:
        identity = np.eye(3, dtype=np.float32)
        quarter_turn = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
        )
        prediction_rotations = np.stack([identity, identity])
        target_rotations = np.stack([quarter_turn, quarter_turn])
        prediction_translation = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float32)
        target_translation = np.array([[10, 20, 30], [10, 21, 30]], dtype=np.float32)
        angle, displacement = first_frame_aligned_root_errors(
            prediction_rotations,
            prediction_translation,
            target_rotations,
            target_translation,
        )
        np.testing.assert_allclose(angle, [0, 0], atol=1e-4)
        np.testing.assert_allclose(displacement, [0, 500], atol=1e-4)

    def test_rollout_uses_previous_root_rotation_for_local_velocity(self) -> None:
        identity = np.eye(3, dtype=np.float32)
        quarter_turn = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
        )
        future_rotations = np.stack([quarter_turn, quarter_turn])
        velocity = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        result = rollout_root_translation(identity, future_rotations, velocity)
        np.testing.assert_allclose(result, [[1, 0, 0], [1, 1, 0]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
