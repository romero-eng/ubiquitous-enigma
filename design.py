# Type Hinting Modules
import numpy.typing as npt

# Calculation Modules
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebfromroots, cheb2poly

# Plotting Modules
import os
import subprocess
import matplotlib
matplotlib.use("Agg")  # flake8 suppression to deal with matplotlib's useage of non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402


def plot_designed_spectrum(MA_spectral_roots: list[np.complex128] = [],
                           AR_spectral_roots: list[np.complex128] = []) -> None:

    def calculate_spectral_magnitude(spectral_roots: npt.NDArray[np.complex128],
                                     delta_freq) -> npt.NDArray[np.float64]:

        return 5*np.log10(chebval(np.cos(2*np.pi*np.arange(0, 0.5 + delta_f, delta_f)),
                                  np.pow(-1, np.sum(np.angle(spectral_roots) == 0))*np.real(chebfromroots(spectral_roots))))

    delta_f = 0.001

    png_filename = "actual.png"
    [fig, ax] = plt.subplots(1)
    ax.plot(np.arange(0, 0.5 + delta_f, delta_f),
            calculate_spectral_magnitude(np.array(MA_spectral_roots), delta_f) - \
            calculate_spectral_magnitude(np.array(AR_spectral_roots), delta_f))
    ax.set_xlim((0, 0.5))
    ax.grid()
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("Digital Frequency (Hz)")
    fig.set_size_inches((8, 5))
    fig.set_layout_engine("tight")
    fig.savefig(png_filename)
    plt.close()
    subprocess.run(["feh", png_filename])
    os.remove(png_filename)

def add_trough(f_e: np.float64,
               target_offset_dB: np.float64,
               roots: list[np.complex128],
               order: int = 1) -> list[np.complex128]:

    M = np.pow(10, target_offset_dB/10)
    cos_omega_e = np.cos(2*np.pi*f_e)

    gamma_mag = np.sqrt(np.square(M) + np.square(cos_omega_e))
    quant = np.arctan(M/cos_omega_e)
    gamma_angle = quant if f_e < 0.25 else np.pi - quant
    gamma = gamma_mag*np.exp(1j*gamma_angle)

    roots += [gamma, np.conjugate(gamma)]*order

    return roots


if (__name__=="__main__"):

    """
    ma_roots: list[np.complex128] = []
    ma_roots = add_trough(0.27 , 0, ma_roots, order = 4)
    ma_roots = add_trough(0.27 , 0, ma_roots, order = 4)

    ar_roots: list[np.complex128] = []
    ar_roots = add_trough(0.24, 0, ar_roots, order = 5)
    ar_roots = add_trough(0.05, 0, ar_roots)

    plot_designed_spectrum(ma_roots, ar_roots)
    """

    ar_roots = []
    #ar_roots = add_trough(0.15, -4, ar_roots)
    ar_roots = add_trough(0.2, -3, ar_roots, 3)
    ar_roots = add_trough(0.25, -12, ar_roots, 2)

    ma_roots = []
    ma_roots = add_trough(0.45, -8, ma_roots, 2)
    ma_roots = add_trough(0.4, -6, ma_roots, 2)
    ma_roots = add_trough(0.35, -3, ma_roots, 2)
    ma_roots = add_trough(0.3, -10, ma_roots, 2)

    plot_designed_spectrum(ma_roots, ar_roots)
