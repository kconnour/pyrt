import numpy as np
from numpy.typing import ArrayLike


def regrid(array, particle_size_grid, wavelength_grid,
        particle_sizes: ArrayLike,
           wavelengths: ArrayLike) \
        -> np.ndarray:
    """Regrid the input array onto a new particle size and wavelength grid
    using nearest neighbor 'interpolation'.

    Parameters
    ----------
    particle_sizes: ArrayLike
        The particle sizes to regrid the array on to.
    wavelengths: ArrayLike
        The wavelengths to regrid the array on to.

    Returns
    -------
    np.ndarray
        Regridded array of shape (..., particle_sizes, wavelengths)

    """
    psizes = _ParticleSizes(particle_sizes, 'particle_sizes')
    #wavs = _Wavelengths(wavelengths, 'wavelengths')
    reff_indices = _get_nearest_indices(particle_size_grid, psizes)
    wav_indices = _get_nearest_indices(wavelength_grid, wavelengths)
    return np.take(np.take(array, reff_indices, axis=-2),
                   wav_indices, axis=-1)


def _get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
        -> np.ndarray:
    # grid should be 1D; values can be ND
    return np.abs(np.subtract.outer(grid, values)).argmin(0)


class _ParticleSizes(np.ndarray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array).view(cls)
        obj.name = name
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def _make_array(obj: np.ndarray):
        try:
            obj = np.asarray(obj)
            obj.astype(float)
        except TypeError as te:
            message = 'The particle size must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The particle sizes must be numeric.'
            raise ValueError(message) from ve
        return obj

    @staticmethod
    def _validate(array):
        if not np.all(array > 0):
            message = 'The particle sizes must be positive.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The particle sizes must be 1-dimensional.'
            raise ValueError(message)


class _Wavelengths(np.ndarray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array).view(cls)
        obj.name = name
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def _make_array(value):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = 'The wavelengths must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The wavelengths must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all((0.1 <= array) & (array <= 50)):
            message = 'The wavelengths must be between 0.1 and 50 microns.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The wavelengths must be 1-dimensional.'
            raise ValueError(message)