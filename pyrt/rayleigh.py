import numpy as np
from pyrt.spectral import wavenumber


def rayleigh_legendre(n_layers: int, n_wavelengths: int) -> np.ndarray:
    r"""Make the generic Legendre decomposition of the Rayleigh scattering
    phase function.

    The Rayleigh scattering phase function is independent of the layer and
    wavelength, and it can be decomposed into 3 moments.

    Parameters
    ----------
    n_layers: int
        The number of layers to make the phase function for.
    n_wavelengths: int
        The number of wavelengths to make

    Returns
    -------
    np.ndarray
        3-dimensional array of the Legendre decomposition of the phase
        function with a shape of ``(3, n_layers, n_wavelengths)``.

    Raises
    ------
    TypeError
        Raised if the inputs are not ints.
    ValueError
        Raised if the inputs cannot be cast into an int.

    Notes
    -----
    Moment 0 is always 1 and moment 2 is always 0.5. The Rayleigh scattering
    phase function is given by

    .. math::
       P(\theta) = \frac{3}{4} (1 + \cos^2(\theta)).

    Since :math:`P_0(x) = 1` and :math:`P_2(x) = \frac{3x^2 - 1}{2},
    :math:`P(\mu) = P_0(\mu) + 0.5 P_2(\mu).

    Examples
    --------
    Make the Rayleigh scattering phase function for a model with 15 layers
    and 5 wavelengths.

    >>> import pyrt
    >>> rayleigh_pf = pyrt.rayleigh_legendre(15, 5)
    >>> rayleigh_pf.shape
    (3, 15, 5)

    """
    try:
        pf = np.zeros((3, n_layers, n_wavelengths))
        pf[0, :] = 1
        pf[2, :] = 0.5
        return pf
    except TypeError as te:
        message = 'The inputs must be ints.'
        raise TypeError(message) from te
    except ValueError as ve:
        message = 'The inputs cannot be cast to ints.'
        raise ValueError(message) from ve
