"""Unit tests for accelerator elements."""
import numpy as np
import pytest

from src.elements import Drift, Quadrupole, BPM, Corrector, ErrorKick


class TestDrift:
    def test_matrix_shape(self):
        d = Drift("d1", length=2.0)
        M = d.matrix()
        assert M.shape == (4, 4)

    def test_propagation(self):
        d = Drift("d1", length=2.0)
        M = d.matrix()
        # x=1mm, x'=1mrad, y=1mm, y'=1mrad
        state = np.array([[1e-3], [1e-3], [1e-3], [1e-3]])
        new_state = M @ state
        assert np.isclose(new_state[0, 0], 3e-3)  # x_new = 1mm + 2m*1mrad = 3mm
        assert np.isclose(new_state[1, 0], 1e-3)   # xp unchanged
        assert np.isclose(new_state[2, 0], 3e-3)  # y_new = 1mm + 2m*1mrad = 3mm
        assert np.isclose(new_state[3, 0], 1e-3)   # yp unchanged

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            Drift("d1", length=-1.0)

    def test_identity_drift(self):
        d = Drift("d1", length=0.0)
        M = d.matrix()
        assert np.allclose(M, np.eye(4))


class TestQuadrupole:
    def test_focusing_matrix(self):
        q = Quadrupole("qf", focal_length=8.0, focusing=True)
        M = q.matrix()
        assert M.shape == (4, 4)
        assert np.isclose(M[0, 0], 1.0)
        assert np.isclose(M[0, 1], 0.0)
        assert np.isclose(M[1, 0], -1.0 / 8.0) # Horizontal focuses
        assert np.isclose(M[1, 1], 1.0)
        assert np.isclose(M[2, 2], 1.0)
        assert np.isclose(M[3, 2], 1.0 / 8.0)  # Vertical defocuses
        assert np.isclose(M[3, 3], 1.0)

    def test_defocusing_matrix(self):
        q = Quadrupole("qd", focal_length=8.0, focusing=False)
        M = q.matrix()
        assert np.isclose(M[1, 0], 1.0 / 8.0)  # Horizontal defocuses
        assert np.isclose(M[3, 2], -1.0 / 8.0) # Vertical focuses

    def test_zero_focal_length_raises(self):
        with pytest.raises(ValueError):
            Quadrupole("q", focal_length=0.0)

    def test_rolled_quadrupole_coupling(self):
        q = Quadrupole("q_skew", focal_length=1.0, focusing=True, tilt=np.pi / 4.0)
        M = q.matrix()
        # Verify off-diagonal cross-plane coupling terms are non-zero
        assert not np.isclose(M[1, 2], 0.0)
        assert not np.isclose(M[3, 0], 0.0)


class TestBPM:
    def test_identity_matrix(self):
        b = BPM("bpm1")
        assert np.allclose(b.matrix(), np.eye(4))


class TestCorrector:
    def test_kick_x(self):
        c = Corrector("corr1", strength=0.2e-3, plane="x")
        state = np.array([[0.0], [0.1e-3], [0.0], [0.0]])
        new = c.apply_kick(state)
        assert np.isclose(new[1, 0], 0.3e-3)  # x' += 0.2 mrad
        assert np.isclose(new[3, 0], 0.0)

    def test_kick_y(self):
        c = Corrector("corr2", strength=0.2e-3, plane="y")
        state = np.array([[0.0], [0.0], [0.0], [0.1e-3]])
        new = c.apply_kick(state)
        assert np.isclose(new[3, 0], 0.3e-3)  # y' += 0.2 mrad
        assert np.isclose(new[1, 0], 0.0)

    def test_custom_kick(self):
        c = Corrector("corr1", strength=0.0, plane="x")
        state = np.array([[0.0], [0.0], [0.0], [0.0]])
        new = c.apply_kick(state, kick_angle=0.5e-3)
        assert np.isclose(new[1, 0], 0.5e-3)


class TestErrorKick:
    def test_kick(self):
        e = ErrorKick("err1", kick_x=0.05e-3, kick_y=-0.02e-3)
        state = np.array([[0.0], [0.0], [0.0], [0.0]])
        new = e.apply_kick(state)
        assert np.isclose(new[1, 0], 0.05e-3)
        assert np.isclose(new[3, 0], -0.02e-3)
