# Type Hinting Modules
import numpy.typing as npt
from typing import Callable

# Calculation Modules
import numpy as np
import scipy.signal as dsp
from numpy.polynomial.chebyshev import chebval, chebfromroots

# Plotting Modules
import os
import subprocess
import matplotlib
matplotlib.use("Agg")  # flake8 supression to deal with matplotlib's usage of non-interative backend
import matplotlib.pyplot as plt  # noqa: E402


def plot_mag_func(z_transform_coefs: npt.NDArray[np.float64],
                  spectal_mag_sq_roots: npt.NDArray[np.float64 | np.complex128]) -> None:

    [omega, h_omega] = dsp.freqz(z_transform_coefs)

    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": np.sqrt(chebval(np.cos(omega), np.real(chebfromroots(-1*np.array(spectal_mag_sq_roots))))),
              "Actual": np.abs(h_omega)}  # noqa: E127

    freq = omega/(2*np.pi)
    png_filename = "actual.png"
    [fig, axes] = plt.subplots(len(sq_magnitudes))
    for ax, title in zip(axes, list(sq_magnitudes.keys())):
        ax.plot(freq, sq_magnitudes[title])
        ax.set_xlim([0, 0.5])
        ax.grid()
        ax.set_title(title)
    fig.supxlabel("Digital Frequency (Hz)")
    fig.set_size_inches((8, 10))
    fig.set_layout_engine("tight")
    fig.savefig(png_filename)
    plt.close()
    subprocess.run(["feh", png_filename])
    os.remove(png_filename)


def calculate_z_coefs_from_complex_spectral_root(gamma: np.complex128) -> npt.NDArray[np.float64]: 

    gamma_abs_sq = np.square(np.abs(gamma))
    gamma_angle = np.angle(gamma)

    eta = gamma_abs_sq + np.sqrt(np.square(gamma_abs_sq) - 2*np.cos(2*gamma_angle)*gamma_abs_sq + 1)
    rho_abs = np.sqrt(eta - np.sqrt(np.square(eta) - 1))
    rho_angle = np.arctan(np.tan(gamma_angle)/np.tanh(np.log(rho_abs))) + np.pi*np.heaviside(gamma_angle - (np.pi/2), 0)

    z_coefs = (1/(2*rho_abs))*np.array([1, 2*rho_abs*np.cos(rho_angle), np.square(rho_abs)])

    return z_coefs


if (__name__ == "__main__"):

    gamma_abs = 2
    gamma_angle_deg = 45
    gamma = gamma_abs*np.exp(1j*(np.pi/180)*gamma_angle_deg)

    cheb_roots = np.array([gamma, np.conjugate(gamma)])

    z_coefs = calculate_z_coefs_from_complex_spectral_root(cheb_roots[0])

    plot_mag_func(z_coefs, cheb_roots)
