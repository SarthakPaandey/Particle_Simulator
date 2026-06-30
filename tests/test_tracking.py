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
            Corrector("c1", plane="x"),
            Drift("d3", length=1.0),
            BPM("b2"),
        ]
        return Lattice(elements)

    def test_zero_state_stays_zero(self):
        lattice = self._simple_lattice()
        traj = track_beam(lattice, BeamState(), None, None, False, 0.0)
        assert np.allclose(traj.x, 0.0)
        assert np.allclose(traj.xp, 0.0)
        assert np.allclose(traj.y, 0.0)
        assert np.allclose(traj.yp, 0.0)

    def test_drift_propagation(self):
        lattice = self._simple_lattice()
        state = BeamState(x=1e-3, xp=0.0, y=2e-3, yp=0.0)
        traj = track_beam(lattice, state, None, None, False, 0.0)
        assert np.isclose(traj.x[0], 1e-3)
        assert np.isclose(traj.y[0], 2e-3)

    def test_corrector_changes_angle(self):
        lattice = self._simple_lattice()
        c = np.array([0.1e-3])
        traj = track_beam(lattice, BeamState(), c, None, False, 0.0)
        angles_after_corrector = traj.xp
        assert angles_after_corrector.sum() != 0.0

    def test_bpm_readings_recorded(self):
        lattice = self._simple_lattice()
        state = BeamState(x=2e-3, xp=0.0, y=1e-3, yp=0.0)
        traj = track_beam(lattice, state, None, None, False, 0.0)
        assert len(traj.bpm_readings_x) == 2
        assert len(traj.bpm_readings_y) == 2

    def test_bpm_noise(self):
        lattice = self._simple_lattice()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        traj_no_noise = track_beam(lattice, BeamState(x=1e-3, y=1e-3), None, None, False, 0.0, rng1)
        traj_noise = track_beam(lattice, BeamState(x=1e-3, y=1e-3), None, None, True, 0.1e-3, rng2)
        assert not np.allclose(traj_no_noise.bpm_x_positions, traj_noise.bpm_x_positions)
        assert not np.allclose(traj_no_noise.bpm_y_positions, traj_noise.bpm_y_positions)

    def test_error_kicks(self):
        lattice = self._simple_lattice()
        kicks = np.zeros((2, len(lattice)))
        kicks[0, 0] = 0.1e-3
        kicks[1, 0] = 0.05e-3
        traj = track_beam(lattice, BeamState(), None, kicks, False, 0.0)
        assert traj.x[-1] != 0.0 or traj.xp[-1] != 0.0
        assert traj.y[-1] != 0.0 or traj.yp[-1] != 0.0

    def test_quadrupole_displacements(self):
        # Displaced quadrupole should act as a dipole steering kick
        q = Quadrupole("qf", focal_length=1.0, focusing=True, dx=1e-3)
        d = Drift("d", length=0.0)
        lattice = Lattice([q, d])
        traj = track_beam(lattice, BeamState(), None, None, False, 0.0)
        # xp_kick = dx/f = 1 mrad
        assert np.isclose(traj.xp[-1], 1e-3)

    def test_bpm_gain_offset(self):
        b = BPM("bpm", dx=0.5e-3, gain_x=2.0)
        lattice = Lattice([b])
        # Beam starting with x = 1.0mm
        state = BeamState(x=1e-3)
        traj = track_beam(lattice, state, None, None, False, 0.0)
        # Measured: gain_x * (x_true - dx) = 2.0 * (1mm - 0.5mm) = 1.0mm
        assert np.isclose(traj.bpm_x_positions[0], 1.0e-3)


class TestResponseMatrix:
    def test_shape(self):
        lattice = build_fodo_lattice(n_cells=4, bpms_per_cell=1, correctors_per_cell=1)
        R = compute_response_matrix(lattice, delta_theta=1e-4)
        n_bpm = len(lattice.bpms)
        n_corr = len(lattice.correctors)
        assert R.shape == (2 * n_bpm, n_corr)

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
        assert kicks.shape == (2, len(lattice))
        
        # Verify non-quad elements have exactly 0 error kick
        for idx, el in enumerate(lattice.elements):
            if not isinstance(el, Quadrupole):
                assert kicks[0, idx] == 0.0
                assert kicks[1, idx] == 0.0
            else:
                # Quadrupoles should have random kicks (non-zero with high probability)
                assert kicks[0, idx] != 0.0 or el.focal_length == 0.0
                assert kicks[1, idx] != 0.0 or el.focal_length == 0.0

    def test_all_elements_kicks(self):
        from src.errors import generate_lattice_error_kicks
        lattice = build_fodo_lattice(n_cells=4)
        kicks = generate_lattice_error_kicks(lattice, error_sigma=1e-3, kick_type="all", seed=42)
        assert kicks.shape == (2, len(lattice))
        assert np.any(kicks[0] != 0.0)
        assert np.any(kicks[1] != 0.0)

