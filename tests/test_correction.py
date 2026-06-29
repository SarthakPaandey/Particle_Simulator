"""Unit tests for orbit correction algorithms."""
import numpy as np
import pytest

from src.beam import BeamState
from src.lattice import build_fodo_lattice
from src.tracking import track_beam, compute_response_matrix
from src.correction import least_squares_correction, svd_correction, iterative_correction
from src.errors import generate_error_kicks
from src.metrics import rms_error


@pytest.fixture
def sim_setup():
    lattice = build_fodo_lattice(n_cells=4, bpms_per_cell=1, correctors_per_cell=1)
    rng = np.random.default_rng(42)
    kicks = generate_error_kicks(len(lattice), error_sigma=5e-5, seed=42)
    traj = track_beam(lattice, BeamState(x=2e-3, xp=0.1e-3), None, kicks, False, 0.0, rng)
    R = compute_response_matrix(lattice, delta_theta=1e-4)
    return lattice, traj, R, kicks, rng


class TestLeastSquaresCorrection:
    def test_improves_rms(self, sim_setup):
        lattice, traj, R, kicks, rng = sim_setup
        bpm_error = traj.bpm_x_positions
        c, _, _, _ = least_squares_correction(R, bpm_error)

        # Apply correction
        traj_corr = track_beam(
            lattice, BeamState(x=2e-3, xp=0.1e-3),
            corrector_strengths=c,
            error_kicks=kicks,
            add_noise=False, bpm_noise_sigma=0.0, rng=rng,
        )
        rms_before = rms_error(bpm_error * 1000)
        rms_after = rms_error(traj_corr.bpm_x_positions * 1000)
        assert rms_after < rms_before or rms_before < 0.01  # trivially small case

    def test_output_shape(self, sim_setup):
        _, _, R, _, _ = sim_setup
        n_corr = R.shape[1]
        c, _, _, _ = least_squares_correction(R, np.zeros(R.shape[0]))
        assert c.shape == (n_corr,)


class TestSVDCorrection:
    def test_improves_rms(self, sim_setup):
        lattice, traj, R, kicks, rng = sim_setup
        bpm_error = traj.bpm_x_positions
        c, U, s, Vt = svd_correction(R, bpm_error, cutoff=1e-4)

        traj_corr = track_beam(
            lattice, BeamState(x=2e-3, xp=0.1e-3),
            corrector_strengths=c,
            error_kicks=kicks,
            add_noise=False, bpm_noise_sigma=0.0, rng=rng,
        )
        rms_before = rms_error(bpm_error * 1000)
        rms_after = rms_error(traj_corr.bpm_x_positions * 1000)
        assert rms_after < rms_before or rms_before < 0.01

    def test_truncation_changes_result(self, sim_setup):
        _, _, R, _, _ = sim_setup
        bpm_error = np.random.default_rng(0).normal(0, 1e-3, size=R.shape[0])
        c_full, _, s, _ = svd_correction(R, bpm_error, cutoff=1e-10)
        c_trunc, _, _, _ = svd_correction(R, bpm_error, n_singular=2)
        assert not np.allclose(c_full, c_trunc)

    def test_singular_values_positive(self, sim_setup):
        _, _, R, _, _ = sim_setup
        _, _, s, _ = svd_correction(R, np.zeros(R.shape[0]))
        assert np.all(s >= 0)


class TestIterativeCorrection:
    def test_runs(self, sim_setup):
        lattice, traj, R, kicks, rng = sim_setup
        c, rms_hist, niters = iterative_correction(
            lattice,
            BeamState(x=2e-3, xp=0.1e-3),
            R,
            method="svd",
            gain=0.8,
            max_iterations=3,
            tolerance_mm=0.001,
            error_kicks=kicks,
            add_noise=False,
            rng=rng,
        )
        assert niters >= 1
        assert len(rms_hist) >= 2

    def test_limited_correctors(self, sim_setup):
        lattice, traj, R, kicks, rng = sim_setup
        c, rms_hist, _ = iterative_correction(
            lattice,
            BeamState(x=2e-3, xp=0.1e-3),
            R,
            method="svd",
            gain=0.8,
            max_iterations=3,
            tolerance_mm=0.001,
            corrector_limit=5e-3,
            apply_limits=True,
            error_kicks=kicks,
            add_noise=False,
            rng=rng,
        )
        assert np.all(np.abs(c) <= 5e-3 + 1e-10)


class TestDirectCorrectionLimits:
    def test_lsq_limited_correctors(self, sim_setup):
        lattice, traj, R, kicks, rng = sim_setup
        bpm_error = traj.bpm_x_positions
        limit_val = 1e-6 # Choose a tiny limit so clipping definitely occurs
        c_unlim, _, _, _ = least_squares_correction(R, bpm_error)
        c_lim, _, _, _ = least_squares_correction(R, bpm_error, limit=limit_val)
        
        # Verify limits are respected
        assert np.all(np.abs(c_lim) <= limit_val + 1e-15)
        # Verify it actually clipped some values (if they were larger than limit)
        if np.any(np.abs(c_unlim) > limit_val):
            assert not np.allclose(c_unlim, c_lim)

    def test_svd_limited_correctors(self, sim_setup):
        lattice, traj, R, kicks, rng = sim_setup
        bpm_error = traj.bpm_x_positions
        limit_val = 1e-6
        c_unlim, _, _, _ = svd_correction(R, bpm_error, cutoff=1e-4)
        c_lim, _, _, _ = svd_correction(R, bpm_error, cutoff=1e-4, limit=limit_val)
        
        assert np.all(np.abs(c_lim) <= limit_val + 1e-15)
        if np.any(np.abs(c_unlim) > limit_val):
            assert not np.allclose(c_unlim, c_lim)

