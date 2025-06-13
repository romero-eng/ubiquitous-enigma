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
                  spectral_roots: npt.NDArray[np.complex128]) -> None:

    [omega, h_omega] = dsp.freqz(z_transform_coefs)

    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": np.sqrt(chebval(np.cos(omega), np.pow(-1, np.sum(np.angle(spectral_roots) == 0))*np.real(chebfromroots(spectral_roots)))),
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


def calculate_complex_spectral_root(gamma: np.complex128) -> npt.NDArray[np.float64]: 

    gamma_abs_sq = np.square(np.abs(gamma))
    gamma_angle = np.angle(gamma)

    eta = gamma_abs_sq + np.sqrt(np.square(gamma_abs_sq) - 2*np.cos(2*gamma_angle)*gamma_abs_sq + 1)
    rho_abs = np.sqrt(eta - np.sqrt(np.square(eta) - 1))
    rho_angle = np.arctan(np.tan(gamma_angle)/np.tanh(np.log(rho_abs))) + np.pi*np.heaviside(gamma_angle - (np.pi/2), 0)

    z_coefs: npt.NDArray[np.float64] = np.array([1, -2*rho_abs*np.cos(rho_angle), np.square(rho_abs)])/(2*rho_abs)

    return z_coefs


def calculate_real_spectral_root(gamma: np.float64) -> npt.NDArray[np.float64]:

    rho_abs = np.abs(gamma) - np.sqrt(np.square(gamma) - 1)
    rho = np.sign(gamma)*rho_abs

    z_coefs: npt.NDArray[np.float64] = np.array([1, -rho])/np.sqrt(2*rho_abs)

    return z_coefs


if (__name__ == "__main__"):

    gammas = \
        np.array([2*np.exp(1j*(np.pi/180)*45),
                  -1.33611,
                  3])

    roots: npt.NDArray[np.complex128] = np.array([gammas[0], np.conjugate(gammas[0]), gammas[1], gammas[2]])

    z_coefs_1 = calculate_complex_spectral_root(roots[0])
    z_coefs_2 = calculate_real_spectral_root(np.float64(np.real(roots[2])))
    z_coefs_3 = calculate_real_spectral_root(np.float64(np.real(roots[3])))

    z_coefs: npt.NDArray[np.float64] = np.array([np.float64(num) for num in np.convolve(np.convolve(z_coefs_1, z_coefs_2), z_coefs_3)])

    plot_mag_func(z_coefs, roots)

