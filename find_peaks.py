# find_peaks.py

import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def plot_spectrum_with_peaks(x, y, sensitivity=0.5, min_distance=20, show_peaks=True):
    """
    Detekuje peaky v daném spektru a vykresluje graf se zobrazením detekovaných peaků.

    Parameters:
        x (array-like): Hodnoty na ose X (např. wavenumber).
        y (array-like): Hodnoty na ose Y (intenzity).
        sensitivity (float): Práh pro detekci peaků (parametr height).
        min_distance (int): Minimální vzdálenost mezi peakami.
        show_peaks (bool): Pokud True, vykreslí textové popisky pro peaky.
    """
    # Detekce peaků
    peaks, properties = find_peaks(y, height=sensitivity, distance=min_distance)
    peak_positions = x[peaks]

    # Vypíšeme nalezené hodnoty
    print(f"Peaks at positions (wavenumbers):\n{peak_positions}")
    print(f"Peak intensities: {y[peaks]}")

    # Vykreslení spektra
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Spectrum")

    if show_peaks:
        for peak in peaks:
            plt.text(x[peak], y[peak],
                     f'x={x[peak]:.2f}',
                     color='red', rotation=90, ha='center', va='bottom', fontsize=14)

    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")
    plt.title("Spectrum with Peaks")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()
