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


def plot_designed_spectrum(MA_spectral_roots: npt.NDArray[np.complex128]) -> None:

    AR_spectral_roots = np.array([])

    delta_f = 0.001
    f = np.arange(0, 0.5 + delta_f, delta_f)

    w = 2*np.pi*f
    MA_sign = np.pow(-1, np.sum(np.angle(MA_spectral_roots) == 0))
    AR_sign = np.pow(-1, np.sum(np.angle(AR_spectral_roots) == 0))
    MA_cheb_coefs = MA_sign*np.real(chebfromroots(MA_spectral_roots))
    AR_cheb_coefs = AR_sign*np.real(chebfromroots(AR_spectral_roots))
    mag_dB = \
        5*(np.log10(chebval(np.cos(w), MA_cheb_coefs)) -
           np.log10(chebval(np.cos(w), AR_cheb_coefs)))

    png_filename = "actual.png"
    [fig, ax] = plt.subplots(1)
    ax.plot(f, mag_dB)
    ax.set_xlim([0, 0.5])
    ax.grid()
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("Digital Frequency (Hz)")
    fig.set_size_inches((8, 5))
    fig.set_layout_engine("tight")
    fig.savefig(png_filename)
    plt.close()
    subprocess.run(["feh", png_filename])
    os.remove(png_filename)


if (__name__=="__main__"):

    f_e = 0.2
    target_offset_dB = -8

    offset = np.pow(10, target_offset_dB/5) + np.square(np.cos(2*np.pi*f_e))
    root = np.sqrt(offset)*np.exp(1j*np.arccos(np.cos(2*np.pi*f_e)/np.sqrt(offset)))

    roots: npt.NDArray[np.complex128] = \
        np.array([root, np.conjugate(root)])

    plot_designed_spectrum(roots) 
