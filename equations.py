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


if (__name__ == "__main__"):

    b = [2.1, -4, 5, 3]
    [w, h] = dsp.freqz(b)
    freq = w/(2*np.pi)

    N = len(b) - 1
    R_cc = np.correlate(b, b, "full")[N:]

    # The "simpler" code with Chebyshev series:
    #   np.sqrt(R_cc[0] + 2*chebval(np.cos(w), np.insert(R_cc[1:], 0, 0)))
    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": np.sqrt(R_cc[0] + 2*np.sum([coef*np.power(np.cos(w), n) for n, coef in enumerate(np.matmul(np.transpose(np.array([np.array(list(cheb2poly(n*[0] + [1])) + (N - n)*[0]) for n in range(N, 0, -1)])), R_cc[:0:-1]))], 0)),
              "Actual": np.abs(h)}  # noqa: E127

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
