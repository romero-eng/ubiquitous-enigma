# Type Hinting Modules
import numpy.typing as npt

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


def calculate_spectral_roots(spectral_roots: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:

    z_coefs: npt.NDArray[np.float64] = np.array([1])

    for gamma in spectral_roots[np.angle(spectral_roots) >= 0]:
        z_coefs = \
            np.convolve(z_coefs,
                        calculate_real_spectral_root(np.float64(np.real(gamma))) if np.angle(gamma) in [0, np.pi] else calculate_complex_spectral_root(gamma)).astype(np.float64)  # noqa: E501

    return z_coefs


def plot_spectral_roots(spectral_roots: npt.NDArray[np.complex128]) -> None:

    [omega, h_omega] = dsp.freqz(calculate_spectral_roots(spectral_roots))

    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": 5*np.log10(chebval(np.cos(omega), np.pow(-1, np.sum(np.angle(spectral_roots) == 0))*np.real(chebfromroots(spectral_roots)))),  # noqa: E501
              "Actual": 10*np.log10(np.abs(h_omega))}  # noqa: E127

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


if (__name__ == "__main__"):

    roots: npt.NDArray[np.complex128] = \
        np.array([2*np.exp( 1j*(np.pi/180)*45),  # noqa: E201
                  2*np.exp(-1j*(np.pi/180)*45),
                  1.33611,
                  -3])

    plot_spectral_roots(roots)
