"""Unit tests for beam tracking."""
import numpy as np
import pytest

from src.beam import BeamState
from src.elements import Drift, BPM, Corrector, Quadrupole
from src.lattice import Lattice, build_fodo_lattice
from src.tracking import track_beam, compute_response_matrix


class TestTrackBeam:
    def _simple_lattice(self):
        elements = [
            Drift("d1", length=1.0),
            BPM("b1"),
            Drift("d2", length=1.0),
            Corrector("c1"),
            Drift("d3", length=1.0),
            BPM("b2"),
        ]
        return Lattice(elements)

    def test_zero_state_stays_zero(self):
        lattice = self._simple_lattice()
        traj = track_beam(lattice, BeamState(), None, None, False, 0.0)
        assert np.allclose(traj.x, 0.0)
        assert np.allclose(traj.xp, 0.0)

    def test_drift_propagation(self):
        lattice = self._simple_lattice()
        state = BeamState(x=1e-3, xp=0.0)
        traj = track_beam(lattice, state, None, None, False, 0.0)
        assert np.isclose(traj.x[0], 1e-3)

    def test_corrector_changes_angle(self):
        lattice = self._simple_lattice()
        c = np.array([0.1e-3])
        traj = track_beam(lattice, BeamState(), c, None, False, 0.0)
        angles_after_corrector = traj.xp
        assert angles_after_corrector.sum() != 0.0

    def test_bpm_readings_recorded(self):
        lattice = self._simple_lattice()
        state = BeamState(x=2e-3, xp=0.0)
        traj = track_beam(lattice, state, None, None, False, 0.0)
        assert len(traj.bpm_readings) == 2

    def test_bpm_noise(self):
        lattice = self._simple_lattice()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        traj_no_noise = track_beam(lattice, BeamState(x=1e-3), None, None, False, 0.0, rng1)
        traj_noise = track_beam(lattice, BeamState(x=1e-3), None, None, True, 0.1e-3, rng2)
        assert not np.allclose(traj_no_noise.bpm_x_positions, traj_noise.bpm_x_positions)

    def test_error_kicks(self):
        lattice = self._simple_lattice()
        kicks = np.zeros(len(lattice))
        kicks[0] = 0.1e-3
        traj = track_beam(lattice, BeamState(), None, kicks, False, 0.0)
        assert traj.x[-1] != 0.0 or traj.xp[-1] != 0.0


class TestResponseMatrix:
    def test_shape(self):
        lattice = build_fodo_lattice(n_cells=4, bpms_per_cell=1, correctors_per_cell=1)
        R = compute_response_matrix(lattice, delta_theta=1e-4)
        n_bpm = len(lattice.bpms)
        n_corr = len(lattice.correctors)
        assert R.shape == (n_bpm, n_corr)

    def test_nonzero(self):
        lattice = build_fodo_lattice(n_cells=4)
        R = compute_response_matrix(lattice, delta_theta=1e-4)
        assert np.any(R != 0.0)

    def test_empty_bpm(self):
        elements = [Drift("d1", length=1.0)]
        lattice = Lattice(elements)
        R = compute_response_matrix(lattice)
        assert R.shape == (0, 0)


class TestLatticeErrorKicks:
    def test_quads_only_kicks(self):
        from src.errors import generate_lattice_error_kicks
        from src.elements import Quadrupole
        lattice = build_fodo_lattice(n_cells=4)
        kicks = generate_lattice_error_kicks(lattice, error_sigma=1e-3, kick_type="quads", seed=42)
        assert len(kicks) == len(lattice)
        
        # Verify non-quad elements have exactly 0 error kick
        for idx, el in enumerate(lattice.elements):
            if not isinstance(el, Quadrupole):
                assert kicks[idx] == 0.0
            else:
                # Quadrupoles should have random kicks (non-zero with high probability)
                assert kicks[idx] != 0.0 or el.focal_length == 0.0

    def test_all_elements_kicks(self):
        from src.errors import generate_lattice_error_kicks
        lattice = build_fodo_lattice(n_cells=4)
        kicks = generate_lattice_error_kicks(lattice, error_sigma=1e-3, kick_type="all", seed=42)
        assert len(kicks) == len(lattice)
        # With high probability, drifts (which make up a large portion of elements) are non-zero
        assert np.any(kicks != 0.0)

