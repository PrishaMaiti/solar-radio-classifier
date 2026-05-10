"""
Noise utilities for generating background "quiet sun" noise in solar radio data.
"""

import numpy as np


def generate_quiet_sun_noise(freq_bins=128, time_bins=256, mean=0.0, std=0.2):
    """
    Generate background quiet sun noise.

    Returns:
        array: Quiet sun noise waveform
    """
    return np.random.normal(mean, std, (freq_bins, time_bins))


def add_noise_to_signal(signal, noise_level=1.0):
    """
    Add quiet sun noise to a signal.

    Args:
        signal: Input signal
        noise_level: Amplitude scaling factor for noise

    Returns:
        array: Signal with added noise
    """
    noise = np.random.normal(0.0, 0.2 * noise_level, signal.shape)
    return signal + noise
