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


class Spectrum:

    def __init__(self):

        self._roots: list[np.complex128] = []
        self._gain_dB: float = 0
        self._overall_order: float = 1

    def set_gain_dB(self,
                    gain_dB: float) -> None:

        self._gain_dB = gain_dB

    def set_overall_order(self,
                          overall_order: float) -> None:

        self._overall_order = overall_order

    def add_downward_curvature(self,
                               target_offset_dB: float,
                               order: int = 1) -> None:

        gamma = -(1 + np.pow(10, target_offset_dB/5))

        self._roots += [gamma]*order

    def add_upward_curvature(self,
                             target_offset_dB: float,
                             order: int = 1) -> list[np.complex128]:

        gamma = 1 + np.pow(10, target_offset_dB/5)

        self._roots += [gamma]*order

    def add_trough(self,
                   f_e: np.float64,
                   target_offset_dB: np.float64,
                   order: int = 1) -> list[np.complex128]:

        M = np.pow(10, target_offset_dB/10)
        cos_omega_e = np.cos(2*np.pi*f_e)

        gamma_mag = np.sqrt(np.square(M) + np.square(cos_omega_e))
        quant = np.arctan(M/cos_omega_e)
        gamma_angle = quant if f_e < 0.25 else np.pi - quant
        gamma = gamma_mag*np.exp(1j*gamma_angle)

        self._roots += [gamma, np.conjugate(gamma)]*order

    def plot(self) -> None: 

        delta_f = 0.001
        f = np.arange(0, 0.5 + delta_f, delta_f)

        spectral_roots: npt.NDArray[np.complex128] = np.array(self._roots)

        png_filename = "actual.png"
        [fig, ax] = plt.subplots(1)
        ax.plot(f, self._overall_order*(self._gain_dB + 5*np.log10(chebval(np.cos(2*np.pi*f), np.pow(-1, np.sum(np.angle(spectral_roots) == 0))*np.real(chebfromroots(spectral_roots))))))
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


if (__name__=="__main__"):

    upward_curves = \
        [(-10, 1),
         ( -5, 1),
         ( -1, 1)]

    troughs = \
        [(0.475, -26.00, 1),
         (0.45 , -25.00, 1),
         (0.425, -24.00, 1),
         (0.4  , -23.00, 1),
         (0.375, -22.00, 1),
         (0.35 , -21.00, 1),
         (0.325, -20.00, 1),
         (0.3  , -10.00, 3),
         (0.25 ,   9.00, 1),
         (0.2  ,  -6.00, 2),
         (0.18 ,   2.00, 1),
         (0.15 ,  -6.50, 2),
         (0.125,   1.00, 1),
         (0.1  ,  -8.00, 1),
         (0.05 , -11.00, 2),
         (0.025,  -0.35, 2)]

    downward_curves = \
        [(-20, 1)]

    ma_spectrum: Spectrum = Spectrum()

    for (target_offset_dB, order) in upward_curves:
        ma_spectrum.add_upward_curvature(target_offset_dB, order)

    for (f_e, target_offset_dB, order) in troughs:
        ma_spectrum.add_trough(f_e, target_offset_dB, order)

    for (target_offset_dB, order) in downward_curves:
        ma_spectrum.add_downward_curvature(target_offset_dB, order)

    ma_spectrum.set_gain_dB(18)
    ma_spectrum.set_overall_order(3)

    ma_spectrum.plot()
