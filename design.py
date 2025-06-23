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

    offset = np.pow(10, target_offset_dB/5) + np.square(np.cos(2*np.pi*f_e))
    root = np.sqrt(offset)*np.exp(1j*np.arccos(np.cos(2*np.pi*f_e)/np.sqrt(offset)))

    roots += [root, np.conjugate(root)]*order

    return roots


if (__name__=="__main__"):

    ma_roots: list[np.complex128] = []
    ma_roots = add_trough(0.27 , 0, ma_roots, order = 4)

    ar_roots: list[np.complex128] = []
    ar_roots = add_trough(0.24, 0, ar_roots, order = 5)
    ar_roots = add_trough(0.05, 0, ar_roots)

    plot_designed_spectrum(ma_roots, ar_roots) 
