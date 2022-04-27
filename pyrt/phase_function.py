import numpy as np
from numpy.typing import ArrayLike


class _ScatteringAngles(np.ndarray):
    def __new__(cls, array: ArrayLike):
        obj = cls._make_array(array).view(cls)
        cls._validate(obj)
        return obj

    @staticmethod
    def _make_array(value):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = 'The scattering angles must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The scattering angles must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all((0 <= array) & (array <= 180)):
            message = 'scattering_angles must be between 0 and 180 degrees.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The scattering angles must be 1-dimensional.'
            raise ValueError(message)


class _FiniteNumericArray(np.ndarray):
    def __new__(cls, array: ArrayLike):
        obj = cls._make_array(array).view(cls)
        cls._validate(obj)
        return obj

    @staticmethod
    def _make_array(value):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = 'The array must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The array must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all(np.isfinite(array)):
            message = 'The array must be finite.'
            raise ValueError(message)


class _PhaseFunctionND(_FiniteNumericArray):
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.all(array >= 0):
            message = 'The phase function must be non-negative'
            raise ValueError(message)


class _PhaseFunction1D(_PhaseFunctionND):
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.ndim(array) == 1:
            message = 'The phase function must be 1-dimensional.'
            raise ValueError(message)


class _AsymmetryParameter(_FiniteNumericArray):
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.all((array >= -1) & (array <= 1)):
            message = 'The asymmetry parameter must be between -1 and 1.'
            raise ValueError(message)


def _validate_samples(samples: int) -> int:
    try:
        return int(samples)
    except TypeError as te:
        message = 'samples must be an int.'
        raise TypeError(message) from te
    except ValueError as ve:
        message = 'samples cannot be converted to an int.'
        raise ValueError(message) from ve


def _validate_moments(moments: int, scattering_angles: np.ndarray = None) \
        -> int:
    try:
        moments = int(moments)
    except TypeError as te:
        message = 'moments must be an int.'
        raise TypeError(message) from te
    except ValueError as ve:
        message = 'moments cannot be converted to an int.'
        raise ValueError(message) from ve
    if moments <= 0:
        message = 'moments must be positive.'
        raise ValueError(message)
    if scattering_angles is not None:
        samples = len(scattering_angles)
        if moments > samples:
            message = f'moments ({moments}) cannot be larger than the ' \
                      f'number of samples ({samples}).'
            raise ValueError(message)
    return moments


def _validate_scattering_angle_dimension(
        phase_function: np.ndarray, scattering_angles: np.ndarray) -> None:
    if phase_function.shape[0] != scattering_angles.shape[0]:
        message = f'Axis 0 of phase_function ({phase_function.shape[0]},) ' \
                  f'must have the same length as ' \
                  f'scattering_angles {scattering_angles.shape}.'
        raise ValueError(message)


def decompose(phase_function: ArrayLike,
              scattering_angles: ArrayLike,
              n_moments: int) -> np.ndarray:
    """Decompose a phase function into Legendre coefficients.

    .. warning::
       This is not vectorized and can only handle 1-dimensional arrays.

    Parameters
    ----------
    phase_function: ArrayLike
        1-dimensional array of phase functions.
    scattering_angles: ArrayLike
        1-dimensional array of the scattering angles [degrees]. This array must
        have the same shape as ``phase_function``.
    n_moments: int
        The number of moments to decompose the phase function into. This
        value must be smaller than the number of points in the phase
        function.

    Returns
    -------
    np.ndarray
        1-dimensional array of Legendre coefficients of the decomposed phase
        function with a shape of ``(moments,)``.

    Raises
    ------
    TypeError
        Raised if the `phase_function` or `scattering_angles` cannot be cast
        into an ndarray.
    ValueError
        Raised if `phase_function` or `scattering_angles` do not have the same
        shape or if `n_moments` is not larger than the length of
        `scattering_angles`.

    """

    def _make_legendre_polynomials(scat_angles, n_mom) -> np.ndarray:
        """Make an array of Legendre polynomials at the scattering angles.

        Notes
        -----
        This returns a 2D array. The 0th index is the i+1 polynomial and the
        1st index is the angle. So index [2, 6] will be the 3rd Legendre
        polynomial (L3) evaluated at the 6th angle

        """
        ones = np.ones((n_mom, scat_angles.shape[0]))

        # This creates an MxN array with 1s on the diagonal and 0s elsewhere
        diag_mask = np.triu(ones) + np.tril(ones) - 1

        # Evaluate the polynomials at the input angles. I don't know why
        return np.polynomial.legendre.legval(
            np.cos(scat_angles), diag_mask)[1:n_mom, :]

    def _make_normal_matrix(phase_func, legendre_poly: np.ndarray) -> np.ndarray:
        return np.sum(
            legendre_poly[:, None, :] * legendre_poly[None, :, :] / phase_func ** 2,
            axis=-1)

    def _make_normal_vector(phase_func, legendre_poly: np.ndarray) -> np.ndarray:
        return np.sum(legendre_poly / phase_func, axis=-1)

    pf = _PhaseFunction1D(phase_function)
    sa = _ScatteringAngles(scattering_angles)
    _validate_scattering_angle_dimension(pf, sa)
    n_moments = _validate_moments(n_moments, sa)
    sa = np.radians(sa)
    try:
        # Subtract 1 since I'm forcing c0 = 1 in the equation
        # P(x) = c0 + c1*L1(x) + ... for DISORT
        pf -= 1
        lpoly = _make_legendre_polynomials(sa, n_moments)
        normal_matrix = _make_normal_matrix(pf, lpoly)
        normal_vector = _make_normal_vector(pf, lpoly)
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        return coeff
    except np.linalg.LinAlgError as lae:
        message = 'The inputs did not make a positive definite matrix.'
        raise ValueError(message) from lae
