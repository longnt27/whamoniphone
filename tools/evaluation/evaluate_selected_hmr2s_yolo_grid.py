"""Evaluate the validation-locked HMR2-S/YOLO/WHAM grid winner on 3DPW test."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import evaluate_full_pipeline_tradeoff as reference_eval
import evaluate_wham_feature_substitution as wham_eval
import joblib
import numpy as np
import torch
import train_hmr2s_wham_adaptation as adaptation
from hmr2s_frozen import (
    HMR2S_CHECKPOINT_SHA256,
    HMR2S_COMMIT,
    FrozenHMR2S,
    checkpoint_smpl_buffers,
)
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test-parsed", type=Path, required=True)
    parser.add_argument("--three-dpw-root", type=Path, required=True)
    parser.add_argument("--wham-repo", type=Path, required=True)
    parser.add_argument("--wham-checkpoint", type=Path, required=True)
    parser.add_argument("--hmr2s-repo", type=Path, required=True)
    parser.add_argument("--hmr2s-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo26-weights", type=Path, required=True)
    parser.add_argument("--yolo-variant", required=True)
    parser.add_argument("--expected-yolo-sha256", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--tuned-wham-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--grid-validation-report", type=Path, required=True)
    parser.add_argument("--smpl-model-directory", type=Path, required=True)
    parser.add_argument("--h36m-joint-regressor", type=Path, required=True)
    parser.add_argument("--wham-joint-regressor", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--pose-batch-size", type=int, default=24)
    parser.add_argument("--hmr-batch-size", type=int, default=24)
    parser.add_argument("--smpl-batch-size", type=int, default=192)
    return parser.parse_args()


def require_files(args: argparse.Namespace) -> None:
    files = (
        args.test_parsed,
        args.wham_checkpoint,
        args.hmr2s_checkpoint,
        args.yolo26_weights,
        args.adapter_checkpoint,
        args.tuned_wham_checkpoint,
        args.grid_validation_report,
        args.h36m_joint_regressor,
        args.wham_joint_regressor,
        args.wham_repo / "lib/models/wham.py",
        args.hmr2s_repo / "4D-Humans/hmr2/models/backbones/vit.py",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing final-evaluation inputs: " + ", ".join(missing)
        )
    for gender in ("NEUTRAL", "MALE", "FEMALE"):
        path = args.smpl_model_directory / f"SMPL_{gender}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint dictionary in {path}")
    return payload


def main() -> None:
    args = parse_args()
    require_files(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for final evaluation")
    wham_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.wham_repo, text=True
    ).strip()
    hmr2s_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.hmr2s_repo, text=True
    ).strip()
    if wham_commit != wham_eval.WHAM_COMMIT:
        raise RuntimeError(f"Wrong WHAM commit: {wham_commit}")
    if hmr2s_commit != HMR2S_COMMIT:
        raise RuntimeError(f"Wrong HMR2-S commit: {hmr2s_commit}")
    if adaptation.sha256(args.hmr2s_checkpoint) != HMR2S_CHECKPOINT_SHA256:
        raise RuntimeError("Wrong HMR2-S checkpoint")
    if adaptation.sha256(args.yolo26_weights) != args.expected_yolo_sha256:
        raise RuntimeError(f"Wrong {args.yolo_variant} checkpoint")

    grid_report = json.loads(args.grid_validation_report.read_text(encoding="utf-8"))
    locked = grid_report["locked_winner"]
    if locked["yolo_variant"] != args.yolo_variant:
        raise RuntimeError("YOLO argument does not match the validation-locked winner")
    if locked["candidate"] != args.selected_candidate:
        raise RuntimeError(
            "Candidate argument does not match the validation-locked winner"
        )

    device = torch.device("cuda")
    released_network = wham_eval.load_wham_core(
        args.wham_repo, args.wham_checkpoint, device
    )
    adapter_payload = load_payload(args.adapter_checkpoint)
    adapter = adaptation.TokenAdapter().to(device)
    adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    adapter.eval()
    tuned_payload = load_payload(args.tuned_wham_checkpoint)
    tuned_network = wham_eval.load_wham_core(
        args.wham_repo, args.wham_checkpoint, device
    )
    tuned_network.load_state_dict(tuned_payload["wham_state_dict"], strict=True)
    tuned_network.eval()

    hmr2s = FrozenHMR2S(args.hmr2s_repo, args.hmr2s_checkpoint).to(device).eval()
    pose_model = YOLO(str(args.yolo26_weights))
    initializer = (
        adaptation.frozen_eval.FrozenSMPLInitializer(
            checkpoint_smpl_buffers(args.hmr2s_checkpoint),
            torch.from_numpy(np.load(args.wham_joint_regressor)).float(),
        )
        .to(device)
        .eval()
    )
    smpl_models = reference_eval.load_smpl_models(args.smpl_model_directory, device)
    h36m_regressor = (
        torch.from_numpy(np.load(args.h36m_joint_regressor)[adaptation.H36M_TO_J14])
        .float()
        .unsqueeze(0)
        .to(device)
    )

    # This is deliberately the first read of 3DPW test in the grid workflow.
    test_labels = joblib.load(args.test_parsed)
    image_root = wham_eval.locate_image_root(args.three_dpw_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report = adaptation.final_test_evaluation(
        test_labels,
        image_root,
        args.wham_repo,
        released_network,
        tuned_network,
        adapter,
        bool(adapter_payload["uses_learned_adapter"]),
        pose_model,
        hmr2s,
        initializer,
        smpl_models,
        h36m_regressor,
        device,
        args.pose_batch_size,
        args.hmr_batch_size,
        args.smpl_batch_size,
        args.output_json,
        args.output_csv,
        yolo_variant=args.yolo_variant,
    )
    report["grid_selection"] = {
        "selection_split": "3DPW validation",
        "test_opened_after_lock": True,
        "locked_winner": locked,
        "reported_deployment_variant": args.selected_candidate,
        "grid_validation_report_sha256": adaptation.sha256(args.grid_validation_report),
        "adapter_checkpoint_sha256": adaptation.sha256(args.adapter_checkpoint),
        "tuned_wham_checkpoint_sha256": adaptation.sha256(args.tuned_wham_checkpoint),
        "yolo_checkpoint_sha256": adaptation.sha256(args.yolo26_weights),
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    adaptation.json_log(
        completed=True,
        locked_winner=locked,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
