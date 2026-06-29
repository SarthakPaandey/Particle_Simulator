"""Lattice construction: FODO cells and ring assembly."""
from __future__ import annotations

import numpy as np
from typing import List, Sequence

from .elements import (
    BPM,
    Corrector,
    Drift,
    Element,
    ErrorKick,
    Quadrupole,
)


class Lattice:
    """An ordered sequence of accelerator elements.

    Convenience helpers expose BPM and corrector lists along with their
    cumulative longitudinal positions.
    """

    def __init__(self, elements: Sequence[Element]):
        self.elements: List[Element] = list(elements)
        self._assign_longitudinal_positions()
        self._index_monitors_and_correctors()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _assign_longitudinal_positions(self) -> None:
        s = 0.0
        for el in self.elements:
            el.s = s
            # Treat BPMs/Correctors/Errors as zero-length markers placed at s.
            s += el.length

    def _index_monitors_and_correctors(self) -> None:
        for i, el in enumerate(self.elements):
            if isinstance(el, BPM):
                el.index = sum(1 for e in self.elements[: i + 1] if isinstance(e, BPM)) - 1
            elif isinstance(el, Corrector):
                el.index = sum(1 for e in self.elements[: i + 1] if isinstance(e, Corrector)) - 1

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def bpms(self) -> List[BPM]:
        return [e for e in self.elements if isinstance(e, BPM)]

    @property
    def correctors(self) -> List[Corrector]:
        return [e for e in self.elements if isinstance(e, Corrector)]

    @property
    def s_positions(self) -> np.ndarray:
        return np.array([e.s for e in self.elements])

    def total_length(self) -> float:
        return float(sum(e.length for e in self.elements))

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx: int) -> Element:
        return self.elements[idx]


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
def build_fodo_lattice(
    n_cells: int = 12,
    drift_length: float = 1.0,
    quad_focal_length: float = 8.0,
    bpms_per_cell: int = 1,
    correctors_per_cell: int = 1,
    half_cell: bool = False,
) -> Lattice:
    """Build a FODO lattice.

    Cell template: DRIFT - QF - DRIFT/(BPM/CORRECTOR) - QD - DRIFT.

    With ``bpms_per_cell`` and ``correctors_per_cell`` set to 1, the BPM
    and corrector are inserted after the QF (centre of the focusing
    half-cell).  Multiple BPMs/correctors distribute around the cell.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if drift_length <= 0:
        raise ValueError("drift_length must be positive")
    if quad_focal_length <= 0:
        raise ValueError("quad_focal_length must be positive")

    # Distribute BPMs and correctors uniformly around the cell by
    # splitting the drift_length into the required number of sub-drifts.
    # We use 2 drift sections per cell (one per half-cell).
    n_drifts = 2
    drift_per_section = drift_length / n_drifts

    # Distribute monitors and correct across the 2 half-cells.
    # BPM placement: evenly spaced at the beginning of each drift.
    # Corrector placement: at the centre of each half-cell (after QF or QD).

    elements: List[Element] = []

    bpm_indices_per_cell = _spread_indices(n_drifts, bpms_per_cell)
    corr_indices_per_cell = _spread_indices(n_drifts, correctors_per_cell)

    for cell_idx in range(n_cells):
        # --- First half: DRIFT (with possible BPM) - QF - DRIFT (with possible CORRECTOR) ---
        # Drift A (may contain BPMs)
        n_bpm_a = sum(1 for i in bpm_indices_per_cell if i == 0)
        sub_drifts_a = max(1, n_bpm_a + 1)
        sub_len_a = drift_per_section / sub_drifts_a
        # Pre-drift
        elements.append(Drift(name=f"Drift_{cell_idx}_A0", length=sub_len_a))
        for k in range(n_bpm_a):
            elements.append(BPM(name=f"BPM_{cell_idx}_A{k}"))
            elements.append(Drift(name=f"Drift_{cell_idx}_A{k+1}", length=sub_len_a))
        # Focusing quad (zero-length, thin lens)
        elements.append(Quadrupole(name=f"QF_{cell_idx}", focal_length=quad_focal_length, focusing=True))
        # Drift B (may contain correctors)
        n_corr_a = sum(1 for i in corr_indices_per_cell if i == 0)
        sub_drifts_b = max(1, n_corr_a + 1)
        sub_len_b = drift_per_section / sub_drifts_b
        elements.append(Drift(name=f"Drift_{cell_idx}_B0", length=sub_len_b))
        for k in range(n_corr_a):
            elements.append(Corrector(name=f"COR_{cell_idx}_A{k}"))
            elements.append(Drift(name=f"Drift_{cell_idx}_B{k+1}", length=sub_len_b))

        # --- Second half: DRIFT (with possible BPM) - QD - DRIFT (with possible CORRECTOR) ---
        # Drift C
        n_bpm_b = sum(1 for i in bpm_indices_per_cell if i == 1)
        sub_drifts_c = max(1, n_bpm_b + 1)
        sub_len_c = drift_per_section / sub_drifts_c
        elements.append(Drift(name=f"Drift_{cell_idx}_C0", length=sub_len_c))
        for k in range(n_bpm_b):
            elements.append(BPM(name=f"BPM_{cell_idx}_B{k}"))
            elements.append(Drift(name=f"Drift_{cell_idx}_C{k+1}", length=sub_len_c))
        # Defocusing quad
        elements.append(Quadrupole(name=f"QD_{cell_idx}", focal_length=quad_focal_length, focusing=False))
        # Drift D
        n_corr_b = sum(1 for i in corr_indices_per_cell if i == 1)
        sub_drifts_d = max(1, n_corr_b + 1)
        sub_len_d = drift_per_section / sub_drifts_d
        elements.append(Drift(name=f"Drift_{cell_idx}_D0", length=sub_len_d))
        for k in range(n_corr_b):
            elements.append(Corrector(name=f"COR_{cell_idx}_B{k}"))
            elements.append(Drift(name=f"Drift_{cell_idx}_D{k+1}", length=sub_len_d))

    return Lattice(elements)


def _spread_indices(total_groups: int, count: int) -> List[int]:
    """Spread ``count`` items across ``total_groups`` groups, as evenly as possible."""
    if count <= 0:
        return []
    if total_groups <= 0:
        raise ValueError("total_groups must be >= 1")
    out: List[int] = []
    for k in range(count):
        out.append(k % total_groups)
    return out


__all__ = ["Lattice", "build_fodo_lattice"]
