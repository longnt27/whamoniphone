#!/usr/bin/env python3
"""Extract the fixed SMPL triangle topology from a licensed checkpoint.

The generated binary is an application resource, but it remains ignored by
Git because its contents are derived from the user's licensed SMPL model.
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
from pathlib import Path

import torch


MAGIC = b"SMPLFACE"
VERSION = 1
VERTEX_COUNT = 6_890
TRIANGLE_COUNT = 13_776
INDICES_PER_TRIANGLE = 3


def find_faces(payload: object) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint is not a dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint state_dict is not a dictionary")

    matches = [
        value
        for key, value in state.items()
        if key.endswith(("smpl.faces_tensor", "smpl.faces", "faces_tensor"))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one SMPL faces tensor, found {len(matches)}"
        )
    faces = matches[0]
    if not isinstance(faces, torch.Tensor):
        raise RuntimeError("SMPL faces entry is not a tensor")
    return faces.detach().cpu().to(torch.int64)


def encode_topology(faces: torch.Tensor) -> bytes:
    if tuple(faces.shape) != (TRIANGLE_COUNT, INDICES_PER_TRIANGLE):
        raise RuntimeError(
            "Unexpected SMPL faces shape "
            f"{tuple(faces.shape)}; expected {(TRIANGLE_COUNT, INDICES_PER_TRIANGLE)}"
        )
    minimum = int(faces.min().item())
    maximum = int(faces.max().item())
    if minimum < 0 or maximum >= VERTEX_COUNT:
        raise RuntimeError(
            f"SMPL face index range [{minimum}, {maximum}] is invalid"
        )

    indices = [int(value) for value in faces.reshape(-1).tolist()]
    header = struct.pack(
        "<8sIIII",
        MAGIC,
        VERSION,
        VERTEX_COUNT,
        TRIANGLE_COUNT,
        INDICES_PER_TRIANGLE,
    )
    return header + struct.pack(f"<{len(indices)}H", *indices)


def atomic_write(output: Path, data: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    faces = find_faces(payload)
    encoded = encode_topology(faces)
    atomic_write(args.output.resolve(), encoded)
    print(
        f"Wrote {args.output.resolve()} with "
        f"{TRIANGLE_COUNT:,} triangles ({len(encoded):,} bytes)"
    )


if __name__ == "__main__":
    main()
