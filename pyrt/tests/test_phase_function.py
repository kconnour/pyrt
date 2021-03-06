import numpy as np
import pytest
from pyrt import decompose, construct_hg, decompose_hg, fit_asymmetry_parameter


class TestDecompose:
    def test_negative_phase_function_raises_value_error(self):
        pf = [-1, 1]
        sa = [10, 20]
        with pytest.raises(ValueError):
            decompose(pf, sa, 5)

    def test_different_shaped_phase_function_scattering_angles_raises_value_error(self):
        pf = [10, 5]
        sa = [0, 10, 20]
        with pytest.raises(ValueError):
            decompose(pf, sa, 5)

    def test_large_n_moments_raises_value_error(self):
        pf = np.linspace(10, 5, num=50)
        sa = np.linspace(0, 100, num=50)
        with pytest.raises(ValueError):
            decompose(pf, sa, 51)


class TestConstructHG:
    @pytest.fixture
    def g(self) -> np.ndarray:
        yield np.array([[0.5, -0.6], [1, 0.2]])

    @pytest.fixture
    def sa(self) -> np.ndarray:
        yield np.linspace(0, 180, num=181)

    def test_function_gives_known_values(self, sa):
        g = 0
        pf = construct_hg(g, sa)
        assert np.all(pf == 1 / (4 * np.pi))

    def test_g_greater_than_1_raises_value_error(self, sa):
        with pytest.raises(ValueError):
            construct_hg(np.nextafter(1, 2), sa)

    def test_g_less_than_negative_1_raises_value_error(self, sa):
        with pytest.raises(ValueError):
            construct_hg(np.nextafter(-1, -2), sa)

    def test_shape_is_scattering_angles_then_g(self, g, sa):
        pf = construct_hg(g, sa)
        assert pf.shape == sa.shape + g.shape


class TestDecomposeHG:
    def test_g_greater_than_1_raises_value_error(self):
        with pytest.raises(ValueError):
            decompose_hg(np.nextafter(1, 2), 129)

    def test_g_less_than_negative_1_raises_value_error(self):
        with pytest.raises(ValueError):
            decompose_hg(np.nextafter(-1, -2), 129)

    def test_moment_0_is_1(self):
        g = np.linspace(-1, 1, num=101)
        assert np.all(decompose_hg(g, 129)[0] == 1)


class TestFitAsymmetryParameter:
    def test_negative_phase_function_raises_value_error(self):
        pf = [-1, 1]
        sa = [10, 20]
        with pytest.raises(ValueError):
            fit_asymmetry_parameter(pf, sa)

    def test_hg_decomposition_gives_known_result(self):
        g = 0.8
        sa = np.linspace(0, 180, num=181000)
        pf = construct_hg(g, sa) * 4 * np.pi
        fit_g = fit_asymmetry_parameter(pf, sa)
        assert np.isclose(g, fit_g, rtol=0.001)
