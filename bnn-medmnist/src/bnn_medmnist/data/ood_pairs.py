"""First-class OOD pair definitions.

An :class:`OODPair` couples an in-distribution dataset with one or more
out-of-distribution datasets used to evaluate OOD detection performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OODPair:
    """Pairing of an in-distribution dataset with OOD evaluation sources.

    Attributes:
        name: Short identifier, e.g. "pneumonia_vs_blood_path".
        in_dist: Flag of the in-distribution MedMNIST dataset (e.g. "pneumoniamnist").
        ood: Flags of OOD datasets to score against.
        notes: Free-form description (channel mismatch, size mismatch, etc.).
    """

    name: str
    in_dist: str
    ood: tuple[str, ...]
    notes: str = ""


# Registry of canonical OOD pairs used in the project.
# TODO: extend as new experiments are added.
PNEUMONIA_VS_BLOOD_PATH = OODPair(
    name="pneumonia_vs_blood_path",
    in_dist="pneumoniamnist",
    ood=("bloodmnist", "pathmnist"),
    notes="Grayscale ID vs. RGB OOD; channel adaptation required.",
)

REGISTRY: dict[str, OODPair] = {
    PNEUMONIA_VS_BLOOD_PATH.name: PNEUMONIA_VS_BLOOD_PATH,
}


def get_pair(name: str) -> OODPair:
    """Look up an OOD pair by name."""
    # TODO: implement — error message listing available pairs.
    raise NotImplementedError
