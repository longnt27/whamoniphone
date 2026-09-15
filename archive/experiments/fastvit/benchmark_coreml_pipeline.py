#!/usr/bin/env python3
"""Run the iOS Core ML workload on the bundled real-image mini set.

This mirrors the Benchmark tab closely enough to validate model contracts and
produce a host baseline. Host timings are not substitutes for iPhone timings.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypeVar

import coremltools as ct
import numpy as np
from PIL import Image


T = TypeVar("T")


def timed(samples: dict[str, list[float]], stage: str, operation: Callable[[], T]) -> T:
    started = time.perf_counter_ns()
    result = operation()
    samples[stage].append((time.perf_counter_ns() - started) / 1_000_000)
    return result


def load_model(path: Path, samples: dict[str, list[float]], stage: str) -> ct.models.MLModel:
    return timed(
        samples,
        stage,
        lambda: ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL),
    )


def detect_person(model: ct.models.MLModel, image: Image.Image) -> dict[str, np.ndarray] | None:
    resized = image.resize((640, 640), Image.Resampling.BILINEAR)
    prediction = model.predict({"image": resized})

    if "var_1033" in prediction:
        # YOLOv8 pose: [1, 56, 8400], xywh + score + 17 x (x, y, score).
        raw = np.asarray(prediction["var_1033"])[0]
        index = int(np.argmax(raw[4]))
        confidence = float(raw[4, index])
        cx, cy, width, height = raw[:4, index] / 640.0
        box = np.asarray(
            [cx - width / 2, cy - height / 2, width, height], dtype=np.float32
        )
        keypoints = np.stack(
            [raw[5::3, index] / 640.0, raw[6::3, index] / 640.0], axis=-1
        ).astype(np.float32)
        keypoint_confidence = raw[7::3, index].astype(np.float32)
    elif "var_1573" in prediction:
        # YOLO26 end-to-end pose: [1, 300, 57], xyxy + score + class
        # + 17 x (x, y, score). The exported model already performs NMS.
        raw = np.asarray(prediction["var_1573"])[0]
        index = int(np.argmax(raw[:, 4]))
        confidence = float(raw[index, 4])
        x1, y1, x2, y2 = raw[index, :4] / 640.0
        box = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        keypoints = np.stack(
            [raw[index, 6::3] / 640.0, raw[index, 7::3] / 640.0], axis=-1
        ).astype(np.float32)
        keypoint_confidence = raw[index, 8::3].astype(np.float32)
    else:
        raise RuntimeError(
            f"Unsupported pose output(s): {sorted(prediction)}; expected var_1033 or var_1573"
        )

    if confidence <= 0.5:
        return None
    return {
        "box": box,
        "keypoints": keypoints,
        "keypoint_confidence": keypoint_confidence,
        "confidence": np.asarray(confidence, dtype=np.float32),
    }


def preprocess_observation(
    image: Image.Image, detection: dict[str, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray, Image.Image | None]:
    if detection is None:
        return (
            np.zeros((1, 1, 37), dtype=np.float32),
            np.ones((1, 1, 17), dtype=np.float32),
            None,
        )

    image_width, image_height = image.size
    keypoint_confidence = detection["keypoint_confidence"]
    valid = keypoint_confidence >= 0.3
    pixel_keypoints = detection["keypoints"] * np.asarray(
        [image_width, image_height], dtype=np.float32
    )
    if int(valid.sum()) >= 7:
        box_points = pixel_keypoints[valid]
        minimum = box_points.min(axis=0)
        maximum = box_points.max(axis=0)
        center_x, center_y = (minimum + maximum) / 2
        side = max(float(np.max(maximum - minimum)) * 1.2, 1)
    else:
        x, y, width, height = detection["box"] * np.asarray(
            [image_width, image_height, image_width, image_height], dtype=np.float32
        )
        center_x = x + width / 2
        center_y = y + height / 2
        side = max(float(max(width, height)) * 1.05, 1)

    coordinates = 2 * (
        pixel_keypoints - np.asarray([center_x, center_y], dtype=np.float32)
    ) / side
    longest = max(image_width, image_height)
    location = np.asarray(
        [
            2 * center_x / longest - image_width / longest,
            2 * center_y / longest - image_height / longest,
            side / longest,
        ],
        dtype=np.float32,
    )
    keypoints = np.concatenate((coordinates.reshape(34), location)).reshape(1, 1, 37)
    mask = (keypoint_confidence < 0.3).astype(np.float32).reshape(1, 1, 17)
    bounds = (
        int(round(center_x - side / 2)),
        int(round(center_y - side / 2)),
        int(round(center_x + side / 2)),
        int(round(center_y + side / 2)),
    )
    crop = image.crop(bounds).resize((256, 256), Image.Resampling.BILINEAR)
    return keypoints.astype(np.float32), mask, crop


def neutral_pose() -> np.ndarray:
    pose = np.zeros((1, 1, 144), dtype=np.float32)
    for joint in range(24):
        pose[0, 0, joint * 6] = 1
        pose[0, 0, joint * 6 + 4] = 1
    return pose


def root_identity() -> np.ndarray:
    rotation = np.zeros((1, 1, 6), dtype=np.float32)
    rotation[0, 0, (0, 4)] = 1
    return rotation


def init_keypoints(first_frame_keypoints: np.ndarray) -> np.ndarray:
    value = np.zeros((1, 1, 88), dtype=np.float32)
    value[..., 51:] = first_frame_keypoints
    return value


def step_input(
    keypoints: np.ndarray,
    mask: np.ndarray,
    feature: np.ndarray,
    feature_valid: bool,
    previous_keypoints: np.ndarray,
    previous_root: np.ndarray,
    previous_pose: np.ndarray,
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "x_step": keypoints,
        "keypoint_mask_step": mask,
        "image_feature_step": feature,
        "image_feature_valid_step": np.asarray([[[float(feature_valid)]]], dtype=np.float32),
        "cam_a_step": np.zeros((1, 1, 6), dtype=np.float32),
        "prev_kp3d": previous_keypoints,
        "prev_root": previous_root,
        "prev_pose": previous_pose,
        "h_enc_in": state["h_enc"],
        "c_enc_in": state["c_enc"],
        "h_traj_in": state["h_traj"],
        "c_traj_in": state["c_traj"],
        "h_dec_in": state["h_dec"],
        "c_dec_in": state["c_dec"],
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("WhamApp/WhamApp/BenchmarkFrames"))
    parser.add_argument("--models", type=Path, default=Path("WhamApp/WhamApp"))
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("yolov8n-pose.mlpackage"),
        help="Pose model path, relative to --models unless absolute",
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame_paths = sorted(args.frames.glob("frame_*.jpg"))
    if len(frame_paths) < 2:
        parser.error("--frames must contain at least two frame_*.jpg files")
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    samples: dict[str, list[float]] = defaultdict(list)

    yolo_path = args.yolo_model
    if not yolo_path.is_absolute():
        yolo_path = args.models / yolo_path
    yolo = load_model(yolo_path, samples, "load_yolo")
    fastvit = load_model(args.models / "FastViTNormalized.mlpackage", samples, "load_fastvit")
    initializer = load_model(args.models / "WHAM_I.mlpackage", samples, "load_wham_init")
    step = load_model(args.models / "WHAM_ImageStep.mlpackage", samples, "load_wham_image_step")

    # Warm all providers.
    warm_detection = detect_person(yolo, images[0])
    if warm_detection is None:
        raise RuntimeError("No person detected in the first sample frame")
    warm_keypoints, warm_mask, warm_crop = preprocess_observation(images[0], warm_detection)
    assert warm_crop is not None
    warm_feature = fastvit.predict({"image_input": warm_crop})["features_1024"].reshape(1, 1, 1024)
    pose = neutral_pose()
    for _ in range(3):
        state = initializer.predict(
            {"init_kp": init_keypoints(warm_keypoints), "init_smpl": pose}
        )
        step.predict(
            step_input(
                warm_keypoints,
                warm_mask,
                warm_feature,
                True,
                np.zeros((1, 1, 51), dtype=np.float32),
                root_identity(),
                pose,
                state,
            )
        )

    detection_count = 0
    comparison_input: dict[str, np.ndarray] | None = None
    last_valid_feature = warm_feature
    passes = max(args.passes, 1)

    for _ in range(passes):
        clip_started = time.perf_counter_ns()
        state: dict[str, np.ndarray] | None = None
        previous_keypoints = np.zeros((1, 1, 51), dtype=np.float32)
        previous_root = root_identity()
        previous_pose = pose

        for frame_index, image in enumerate(images):
            detection = timed(samples, "yolo_pose_and_parse", lambda image=image: detect_person(yolo, image))
            detection_count += detection is not None
            keypoints, mask, crop = timed(
                samples,
                "crop_and_keypoint_normalization",
                lambda image=image, detection=detection: preprocess_observation(image, detection),
            )
            if crop is not None:
                feature = timed(
                    samples,
                    "fastvit_normalized",
                    lambda crop=crop: fastvit.predict({"image_input": crop})["features_1024"].reshape(1, 1, 1024),
                )
                feature_valid = True
                last_valid_feature = feature
            else:
                feature = np.zeros((1, 1, 1024), dtype=np.float32)
                feature_valid = False

            if frame_index == 0:
                state = timed(
                    samples,
                    "wham_init",
                    lambda: initializer.predict(
                        {"init_kp": init_keypoints(keypoints), "init_smpl": pose}
                    ),
                )
                continue
            assert state is not None
            model_input = step_input(
                keypoints,
                mask,
                feature,
                feature_valid,
                previous_keypoints,
                previous_root,
                previous_pose,
                state,
            )
            output = timed(samples, "wham_image_step", lambda: step.predict(model_input))
            previous_keypoints = output["pred_kp3d"]
            previous_root = output["pred_root"]
            previous_pose = output["pred_pose"]
            for name in ("h_enc", "c_enc", "h_traj", "c_traj", "h_dec", "c_dec"):
                state[name] = output[f"{name}_out"]
            comparison_input = model_input

        clip_ms = (time.perf_counter_ns() - clip_started) / 1_000_000
        samples["full_8_frame_clip"].append(clip_ms)
        samples["amortized_source_frame"].append(clip_ms / len(images))

    assert comparison_input is not None
    connected_input = dict(comparison_input)
    connected_input["image_feature_step"] = last_valid_feature
    connected_input["image_feature_valid_step"] = np.ones((1, 1, 1), dtype=np.float32)
    ablated_input = dict(comparison_input)
    ablated_input["image_feature_step"] = np.zeros((1, 1, 1024), dtype=np.float32)
    ablated_input["image_feature_valid_step"] = np.zeros((1, 1, 1), dtype=np.float32)
    connected = step.predict(connected_input)["pred_pose"]
    ablated = step.predict(ablated_input)["pred_pose"]
    feature_delta = float(np.max(np.abs(connected - ablated)))

    report = {
        "schema_version": 2,
        "host": platform.platform(),
        "pose_detector": yolo_path.stem,
        "pose_detector_bytes": sum(
            path.stat().st_size for path in yolo_path.rglob("*") if path.is_file()
        ),
        "sample_set": "8 frames sampled from official WHAM IMG_9730 example",
        "frames": len(images),
        "passes": passes,
        "person_detections": int(detection_count),
        "possible_detections": len(images) * passes,
        "initializer_mode": "WHAM_I with real first-frame 2D/crop input and neutral 3D/SMPL fallback; the available FastViT distillation failed HMR2 feature-space validation",
        "feature_connection_pose_max_delta": feature_delta,
        "feature_connection_passed": feature_delta > 1e-5,
        "timings": {name: summarize(values) for name, values in samples.items()},
        "timing_note": "Host Core ML baseline only; run the Benchmark tab in Release on iPhone for device figures.",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
