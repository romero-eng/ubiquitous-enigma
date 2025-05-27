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


def plot_irreducible_quadratic_mag_func(mag_rho: float,
                                        angle_deg_rho: float) -> None:

    K = 2*mag_rho
    angle_rho = (np.pi/180)*angle_deg_rho
    gamma_r = np.cos(angle_rho)*np.cosh(np.log(mag_rho))
    gamma_i = np.sin(angle_rho)*np.sinh(np.log(mag_rho))

    b = [1, 2*mag_rho*np.cos(angle_rho), np.square(mag_rho)]

    irreducible_quadratic_mag_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = \
        lambda w: K*np.sqrt(np.square(np.cos(w) + gamma_r) + np.square(gamma_i))

    plot_mag_func(b, irreducible_quadratic_mag_func)


if (__name__ == "__main__"):

    """
    b = [2.1, -4, 5, 3]

    N = len(b) - 1
    R_cc = np.correlate(b, b, "full")[N:]

    simple_mag_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = \
        lambda w: np.sqrt(R_cc[0] + 2*chebval(np.cos(w), np.insert(R_cc[1:], 0, 0)))

    expanded_mag_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = \
        lambda w: np.sqrt(R_cc[0] + 2*np.sum([coef*np.power(np.cos(w), n) for n, coef in enumerate(np.matmul(np.transpose(np.array([np.array(list(cheb2poly(n*[0] + [1])) + (N - n)*[0]) for n in range(N, 0, -1)])), R_cc[:0:-1]))], 0))  # noqa: E501
    """

    plot_irreducible_quadratic_mag_func(0.6432, 15)
