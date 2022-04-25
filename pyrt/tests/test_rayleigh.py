import numpy as np
import pytest
from pyrt import rayleigh_legendre


class TestRayleighLegendre:
    def test_float_layers_raises_type_error(self):
        with pytest.raises(TypeError):
            rayleigh_legendre(3.5, 10)

    def test_float_wavelengths_raises_type_error(self):
        with pytest.raises(TypeError):
            rayleigh_legendre(10, 3.5)

    def test_moment_0_is_always_1(self):
        rleg = rayleigh_legendre(100, 100)
        assert np.all(rleg[0] == 1)

    def test_moment_2_is_always_05(self):
        rleg = rayleigh_legendre(100, 100)
        assert np.all(rleg[2] == 0.5)
