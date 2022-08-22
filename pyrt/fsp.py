import numpy as np
from pyrt.grid import regrid


def extinction_ratio_grid(extinction_cross_section, particle_size_grid, wavelength_grid, wavelength_reference: float) -> np.ndarray:
    """Make a grid of extinction cross section ratios.

    This is the extinction cross section at the input wavelengths divided by
    the extinction cross section at the reference wavelength.

    Parameters
    ----------
    extinction_cross_section
    particle_size_grid
    wavelength_gri
    wavelength_reference

    Returns
    -------

    """
    cext_slice = np.squeeze(regrid(extinction_cross_section, particle_size_grid, wavelength_grid, particle_size_grid, wavelength_reference))
    return (extinction_cross_section.T / cext_slice).T


def optical_depth(q_prof, column_density, extinction_ratio, column_integrated_od):
    """Make the optical depth in each layer.

    Parameters
    ----------
    q_prof
    column_density
    extinction_ratio
    column_integrated_od

    Returns
    -------

    """
    normalization = np.sum(q_prof * column_density)
    profile = q_prof * column_density * column_integrated_od / normalization
    return (profile * extinction_ratio.T).T