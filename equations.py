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
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Added a flake8 supression to deal with matplotlib's usage of non-interative backend  # noqa: E402, E501


if (__name__ == "__main__"):

    U_: Callable[[int, int], npt.NDArray[np.float64]] = \
        lambda N, n: np.concatenate((np.concatenate((np.zeros((N - n, n)), np.identity(N - n)),   axis=1),
                                     np.concatenate((np.zeros((    n, n)), np.zeros((n, N - n))), axis=1)),  # noqa: E201, E501
                                    axis=0)

    mag_: Callable[[list[float], npt.NDArray[np.float64]], npt.NDArray[np.float64]] = \
        lambda c, w: np.sqrt(np.dot(c, c) + 2*chebval(np.cos(w), [np.dot(c, np.matmul(U_(len(c), n), c)) if n > 0 else 0 for n in range(len(c))]))  # noqa: E501

    b = [2.1, -4, 5, 3]
    [w, h] = dsp.freqz(b)
    freq = w/(2*np.pi)

    sq_magnitudes: dict[str, npt.NDArray[np.float64]] = \
        {"Theoretical": mag_(b, w),
              "Actual": np.abs(h)}                              # noqa: E127

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
