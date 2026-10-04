"""Checks shared by both arms of a paired world-space evaluation."""

from __future__ import annotations

from collections.abc import Mapping


def verify_paired_population(
    expected: Mapping[str, int],
    original: Mapping[str, int],
    phone: Mapping[str, int],
) -> int:
    """Reject a comparison with even one missing or shortened person track."""

    reference = dict(expected)
    for name, observed in (("original", original), ("phone", phone)):
        if dict(observed) != reference:
            raise ValueError(
                f"{name} 3DPW track/frame population differs from locked test: "
                f"observed={dict(observed)}, expected={reference}"
            )
    return sum(reference.values())
