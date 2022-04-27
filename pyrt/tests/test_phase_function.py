import numpy as np
import pytest
from pyrt import decompose, construct_hg


class TestDecompose:
    @pytest.fixture
    def angles(self) -> np.ndarray:
        yield np.array([0, 10, 70])

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