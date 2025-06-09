# Type Hinting Modules
import numpy.typing as npt
from typing import Callable

# Calculation Modules
import numpy as np
import scipy.signal as dsp
from numpy.polynomial.chebyshev import chebval, cheb2poly

# Plotting Modules
import os
import subprocess
import matplotlib
matplotlib.use("Agg")  # flake8 supression to deal with matplotlib's usage of non-interative backend
import matplotlib.pyplot as plt  # noqa: E402


def plot_mag_func(b: list[float],
                  mag_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:

    [w, h] = dsp.freqz(b)

    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": mag_func(w),
              "Actual": np.abs(h)}  # noqa: E127

    freq = w/(2*np.pi)
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


def plot_irreducible_quadratic_mag_func(gamma_abs: float,
                                        gamma_angle_deg: float) -> None: 
 
    gamma_abs_sq = np.square(gamma_abs)
    gamma_angle = (np.pi/180)*gamma_angle_deg
    gamma = gamma_abs*np.exp(1j*gamma_angle)

    eta = gamma_abs_sq + np.sqrt(np.square(gamma_abs_sq) - 2*np.cos(2*gamma_angle)*gamma_abs_sq + 1)
    rho_abs = np.sqrt(eta - np.sqrt(np.square(eta) - 1))
    rho_angle = (1/2)*np.arccos(2*gamma_abs_sq - eta) 

    K = 2*rho_abs

    plot_mag_func([1, 2*rho_abs*np.cos(rho_angle), np.square(rho_abs)],
                  lambda w: K*np.sqrt(np.abs((np.cos(w) + gamma)*(np.cos(w) + np.conjugate(gamma)))))


if (__name__ == "__main__"):

    plot_irreducible_quadratic_mag_func(2,
                                        35)
