# Type Hinting Modules
import numpy.typing as npt
from typing import Callable

# Calculation Modules
import numpy as np
import scipy.signal as dsp
from numpy.polynomial.chebyshev import chebval

# Plotting Modules
import os
import subprocess
import matplotlib
matplotlib.use("Agg")  # flake8 supression to deal with matplotlib's usage of non-interative backend
import matplotlib.pyplot as plt  # noqa: E402


if (__name__ == "__main__"):

    mag_: Callable[[list[float], npt.NDArray[np.float64]], npt.NDArray[np.float64]] = \
        lambda c, w: np.sqrt(np.dot(c, c) + 2*chebval(np.cos(w), np.insert(np.correlate(c, c, "full")[len(c):], 0, 0)))

    b = [2.1, -4, 5, 3]
    [w, h] = dsp.freqz(b)
    freq = w/(2*np.pi)

    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": mag_(b, w),
              "Actual": np.abs(h)}  # noqa: E127

    png_filename = "actual.png"
    [fig, axes] = plt.subplots(len(sq_magnitudes))
    for ax, title in zip(axes, list(sq_magnitudes.keys())):
        ax.plot(freq, sq_magnitudes[title])
        ax.set_xlim([0, 0.5])
        # ax.set_ylim(bottom=0)
        ax.grid()
        ax.set_title(title)
    fig.supxlabel("Digital Frequency (Hz)")
    fig.set_size_inches((8, 10))
    fig.set_layout_engine("tight")
    fig.savefig(png_filename)
    plt.close()
    subprocess.run(["feh", png_filename])
    os.remove(png_filename)
