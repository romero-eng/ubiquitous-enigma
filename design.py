# Type Hinting Modules
import numpy.typing as npt

# Calculation Modules
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebfromroots

# Plotting Modules
import os
import subprocess
import matplotlib
matplotlib.use("Agg")  # flake8 suppression to deal with matplotlib's useage of non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402


def plot_designed_spectrum(spectral_roots: npt.NDArray[np.complex128]) -> None:

    delta_f = 0.01
    f = np.arange(0, 0.5 + delta_f, delta_f)

    png_filename = "actual.png"
    [fig, ax] = plt.subplots(1)
    ax.plot(f, 5*np.log10(chebval(np.cos(2*np.pi*f), np.pow(-1, np.sum(np.angle(spectral_roots) == 0))*np.real(chebfromroots(spectral_roots))))) 
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

    offset = 1
    f_e = 0.2

    root = np.sqrt(offset)*np.exp(1j*np.arccos(np.cos(2*np.pi*f_e)/np.sqrt(offset)))

    roots: npt.NDArray[np.complex128] = \
        np.array([root, np.conjugate(root)])

    plot_designed_spectrum(roots) 
