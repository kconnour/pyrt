import numpy as np
import pytest
from pyrt import rayleigh_legendre, rayleigh_co2_optical_depth


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


class TestRayleighCO2OpticalDepth:
    @pytest.fixture
    def wavelengths(self):
        yield np.linspace(0.2, 40, num=50)

    @pytest.fixture
    def column_density(self):
        yield np.linspace(10**25, 10**26, num=10)

    def test_optical_depth_monotonically_decreases(self, column_density, wavelengths):
        od = rayleigh_co2_optical_depth(column_density, wavelengths)
        column_od = np.sum(od, axis=0)
        assert np.all(np.diff(column_od) < 0)

    def test_int_colden_raises_type_error(self, wavelengths):
        with pytest.raises(TypeError):
            rayleigh_co2_optical_depth(10**25, wavelengths)

    def test_too_small_wavelength_raises_value_error(self, column_density):
        with pytest.raises(ValueError):
            rayleigh_co2_optical_depth(column_density, 0.05)

    def test_too_large_wavelength_raises_value_error(self, column_density):
        with pytest.raises(ValueError):
            rayleigh_co2_optical_depth(column_density, 51)
