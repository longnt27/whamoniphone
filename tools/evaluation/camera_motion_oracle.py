"""Convert 3DPW camera poses to WHAM's 30-fps camera-motion input.

This reproduces the official WHAM evaluation convention for datasets with
known camera rotations. It is an oracle signal, not a simulated phone gyro.
"""

from __future__ import annotations

import numpy as np


IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def camera_angular_velocity(
    camera_poses: np.ndarray,
    recurrent_frames: int | None = None,
    fps: float = 30.0,
) -> np.ndarray:
    """Return [N-1, 6] WHAM angular input from N 3DPW camera poses."""

    poses = np.asarray(camera_poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected camera poses [N,4,4], got {poses.shape}")
    if recurrent_frames is not None and len(poses) != recurrent_frames + 1:
        raise ValueError(
            f"expected {recurrent_frames + 1} camera poses, got {len(poses)}"
        )
    if len(poses) < 2:
        raise ValueError("expected at least 2 camera poses")
    rotations = poses[:, :3, :3]
    if not np.isfinite(rotations).all():
        raise ValueError("camera rotations contain non-finite values")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")

    relative = rotations[:-1] @ np.swapaxes(rotations[1:], -1, -2)
    six_d = relative[:, :2, :].reshape(-1, 6)
    return ((six_d - IDENTITY_6D) * np.float32(fps)).astype(np.float32)


def camera_angular_velocity_with_validity(
    camera_poses: np.ndarray,
    frame_ids: np.ndarray,
    camera_valid: np.ndarray,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep every transition; zero only those without two adjacent valid poses.

    ``camera_valid`` here is the validity mask for the supplied poses, not the
    full raw-sequence mask. The returned Boolean mask identifies transitions
    for which the 3DPW camera rotation was actually supplied to WHAM.
    """

    poses = np.asarray(camera_poses, dtype=np.float32)
    ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    valid = np.asarray(camera_valid, dtype=bool).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected camera poses [N,4,4], got {poses.shape}")
    if len(poses) < 2 or len(poses) != len(ids) or len(ids) != len(valid):
        raise ValueError("camera poses, frame IDs and validity must share N >= 2")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    usable = valid[:-1] & valid[1:] & (ids[1:] == ids[:-1] + 1)
    angular = np.zeros((len(poses) - 1, 6), dtype=np.float32)
    if usable.any():
        pair = np.flatnonzero(usable)
        rotations = poses[:, :3, :3]
        if not np.isfinite(rotations[pair]).all() or not np.isfinite(rotations[pair + 1]).all():
            raise ValueError("valid camera rotations contain non-finite values")
        relative = rotations[pair] @ np.swapaxes(rotations[pair + 1], -1, -2)
        angular[pair] = (relative[:, :2, :].reshape(-1, 6) - IDENTITY_6D) * np.float32(fps)
    return angular, usable


def first_frame_aligned_root_errors(
    predicted_rotations: np.ndarray,
    predicted_translation: np.ndarray,
    target_rotations: np.ndarray,
    target_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Root rotation degrees and translation millimetres after first-frame SE(3) alignment.

    This diagnostic is not WHAM's official world-MPJPE. Alignment is used only
    for scoring, never as an inference input.
    """

    pred_r = np.asarray(predicted_rotations, dtype=np.float64)
    pred_t = np.asarray(predicted_translation, dtype=np.float64)
    gt_r = np.asarray(target_rotations, dtype=np.float64)
    gt_t = np.asarray(target_translation, dtype=np.float64)
    if pred_r.ndim != 3 or pred_r.shape[1:] != (3, 3):
        raise ValueError(f"predicted root rotations must be [N,3,3], got {pred_r.shape}")
    if gt_r.shape != pred_r.shape or pred_t.shape != (len(pred_r), 3) or gt_t.shape != pred_t.shape:
        raise ValueError("predicted and target root trajectories must have matching lengths")
    if len(pred_r) == 0 or not all(np.isfinite(a).all() for a in (pred_r, pred_t, gt_r, gt_t)):
        raise ValueError("root trajectories must be nonempty and finite")

    alignment = gt_r[0] @ pred_r[0].T
    aligned_rotations = alignment @ pred_r
    relative = aligned_rotations @ np.swapaxes(gt_r, -1, -2)
    cosines = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    angular_error_degrees = np.rad2deg(np.arccos(cosines)).astype(np.float32)
    aligned_displacement = (alignment @ (pred_t - pred_t[0]).T).T
    target_displacement = gt_t - gt_t[0]
    translation_error_mm = (
        np.linalg.norm(aligned_displacement - target_displacement, axis=-1) * 1000.0
    ).astype(np.float32)
    return angular_error_degrees, translation_error_mm


def masked_first_valid_root_errors(
    predicted_rotations: np.ndarray,
    predicted_translation: np.ndarray,
    target_rotations: np.ndarray,
    target_translation: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every valid world-root frame after first-valid-frame alignment."""

    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if len(mask) != len(predicted_rotations):
        raise ValueError("world-root validity mask length mismatch")
    if not mask.any():
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    first = int(np.flatnonzero(mask)[0])
    # Invalid labels may contain non-finite values; only feed valid rows to the
    # diagnostic. Alignment remains anchored to the first valid world frame.
    selected = np.flatnonzero(mask[first:]) + first
    return first_frame_aligned_root_errors(
        np.asarray(predicted_rotations)[selected],
        np.asarray(predicted_translation)[selected],
        np.asarray(target_rotations)[selected],
        np.asarray(target_translation)[selected],
    )


def first_frame_aligned_world_joint_error(
    predicted_joints: np.ndarray,
    target_joints: np.ndarray,
    predicted_root_rotations: np.ndarray,
    target_root_rotations: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Per-frame 14-joint world error, after one rigid first-valid-frame alignment.

    Rotation aligns root axes; translation aligns first-frame joint centroids.
    No scale adjustment or per-frame pose fitting is allowed. This diagnostic
    differs from WHAM's official 100-frame W-MPJPE protocol.
    """

    pred = np.asarray(predicted_joints, dtype=np.float64)
    gt = np.asarray(target_joints, dtype=np.float64)
    pred_r = np.asarray(predicted_root_rotations, dtype=np.float64)
    gt_r = np.asarray(target_root_rotations, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if pred.ndim != 3 or pred.shape[-1] != 3 or gt.shape != pred.shape:
        raise ValueError("predicted and target world joints must match [N,J,3]")
    if pred_r.shape != (len(pred), 3, 3) or gt_r.shape != pred_r.shape or len(mask) != len(pred):
        raise ValueError("world-joint root rotations and validity must match N")
    selected = np.flatnonzero(mask)
    if not len(selected):
        return np.empty(0, dtype=np.float32)
    if not all(np.isfinite(value[selected]).all() for value in (pred, gt, pred_r, gt_r)):
        raise ValueError("valid world joints or root rotations contain non-finite values")
    anchor = int(selected[0])
    alignment = gt_r[anchor] @ pred_r[anchor].T
    offset = gt[anchor].mean(axis=0) - alignment @ pred[anchor].mean(axis=0)
    aligned = np.einsum("ab,njb->nja", alignment, pred[selected]) + offset
    return (np.linalg.norm(aligned - gt[selected], axis=-1).mean(axis=-1) * 1000.0).astype(np.float32)


def rollout_root_translation(
    initial_root_rotation: np.ndarray,
    predicted_root_rotations: np.ndarray,
    local_velocity: np.ndarray,
) -> np.ndarray:
    """Integrate WHAM local root velocities with the preceding root rotation."""

    initial = np.asarray(initial_root_rotation, dtype=np.float64)
    roots = np.asarray(predicted_root_rotations, dtype=np.float64)
    velocity = np.asarray(local_velocity, dtype=np.float64)
    if initial.shape != (3, 3) or roots.ndim != 3 or roots.shape[1:] != (3, 3):
        raise ValueError("root rotation shapes must be [3,3] and [N,3,3]")
    if velocity.shape != (len(roots), 3):
        raise ValueError("root velocity must be [N,3]")
    if not all(np.isfinite(a).all() for a in (initial, roots, velocity)):
        raise ValueError("root rollout inputs contain non-finite values")
    preceding = np.concatenate((initial[None], roots[:-1]), axis=0)
    world_velocity = (preceding @ velocity[..., None]).squeeze(-1)
    return np.cumsum(world_velocity, axis=0).astype(np.float32)
