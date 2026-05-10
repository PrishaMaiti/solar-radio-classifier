"""
Synthesizer module for generating solar radio burst shapes.
"""

import numpy as np


def _blank_canvas(freq_bins, time_bins, background_mean=0.0, background_std=0.2):
    return np.random.normal(background_mean, background_std, (freq_bins, time_bins))


def _add_gaussian_blob(canvas, center_freq, center_time, freq_sigma, time_sigma, amplitude):
    freq_idx, time_idx = np.indices(canvas.shape)
    blob = amplitude * np.exp(
        -(((freq_idx - center_freq) ** 2) / (2.0 * freq_sigma**2)
          + ((time_idx - center_time) ** 2) / (2.0 * time_sigma**2))
    )
    canvas += blob


def _draw_line(canvas, start_freq, start_time, end_freq, end_time, thickness, amplitude):
    freq_idx, time_idx = np.indices(canvas.shape)
    line_vec = np.array([end_freq - start_freq, end_time - start_time], dtype=float)
    line_length = np.hypot(line_vec[0], line_vec[1])
    if line_length == 0:
        return

    rel_freq = freq_idx - start_freq
    rel_time = time_idx - start_time
    projection = (rel_freq * line_vec[0] + rel_time * line_vec[1]) / line_length
    closest_freq = start_freq + projection * line_vec[0] / line_length
    closest_time = start_time + projection * line_vec[1] / line_length
    distance = np.sqrt((freq_idx - closest_freq) ** 2 + (time_idx - closest_time) ** 2)
    canvas += amplitude * np.exp(-(distance**2) / (2.0 * thickness**2))


def generate_no_burst(freq_bins=128, time_bins=256, background_mean=0.0, background_std=0.2):
    """
    Generate the quiet-sun background baseline.

    Returns:
        array: Background noise spectrogram
    """
    return _blank_canvas(freq_bins, time_bins, background_mean, background_std)


def generate_type_1(freq_bins=128, time_bins=256, num_spikes=50, band=None, spike_amplitude=3.0):
    """
    Generate a Type 1 solar radio burst shape.

    Returns:
        array: Type 1 burst spectrogram
    """
    canvas = _blank_canvas(freq_bins, time_bins)
    if band is None:
        band_start = int(freq_bins * 0.25)
        band_end = int(freq_bins * 0.7)
    else:
        band_start, band_end = band

    band_start = max(0, min(freq_bins - 1, band_start))
    band_end = max(band_start + 1, min(freq_bins, band_end))

    spike_rows = np.random.randint(band_start, band_end, size=num_spikes)
    spike_cols = np.random.randint(0, time_bins, size=num_spikes)
    canvas[spike_rows, spike_cols] += spike_amplitude * (0.7 + 0.6 * np.random.random(num_spikes))
    return canvas


def generate_type_2(
    freq_bins=128,
    time_bins=256,
    start_freq=None,
    start_time=None,
    drift=None,
    thickness=1.8,
    amplitude=4.0,
    add_harmonic=True,
):
    """
    Generate a Type 2 solar radio burst shape.

    Returns:
        array: Type 2 burst spectrogram
    """
    canvas = _blank_canvas(freq_bins, time_bins)
    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.45, freq_bins * 0.8)
    if start_time is None:
        start_time = np.random.uniform(0, time_bins * 0.25)
    if drift is None:
        drift = np.random.uniform(freq_bins * 0.25, freq_bins * 0.55)

    end_freq = max(0, start_freq - drift)
    end_time = min(time_bins - 1, start_time + np.random.uniform(time_bins * 0.35, time_bins * 0.8))
    _draw_line(canvas, start_freq, start_time, end_freq, end_time, thickness, amplitude)

    if add_harmonic:
        harmonic_start = min(freq_bins - 1, start_freq * 2.0)
        harmonic_end = max(0, harmonic_start - drift * 1.15)
        _draw_line(canvas, harmonic_start, start_time + 3, harmonic_end, end_time + 3, thickness, amplitude * 0.55)

    return canvas


def generate_type_3(
    freq_bins=128,
    time_bins=256,
    start_freq=None,
    start_time=None,
    thickness=1.1,
    amplitude=4.5,
):
    """
    Generate a Type 3 solar radio burst shape.

    Returns:
        array: Type 3 burst spectrogram
    """
    canvas = _blank_canvas(freq_bins, time_bins)
    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.35, freq_bins * 0.85)
    if start_time is None:
        start_time = np.random.uniform(0, time_bins * 0.35)

    end_freq = max(0, start_freq - np.random.uniform(freq_bins * 0.7, freq_bins * 0.95))
    end_time = min(time_bins - 1, start_time + np.random.uniform(time_bins * 0.08, time_bins * 0.2))
    _draw_line(canvas, start_freq, start_time, end_freq, end_time, thickness, amplitude)
    return canvas


def generate_type_4(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=3.5):
    """
    Generate a Type 4 solar radio burst shape.

    Returns:
        array: Type 4 burst spectrogram
    """
    canvas = _blank_canvas(freq_bins, time_bins)
    if center is None:
        center = (
            np.random.uniform(freq_bins * 0.25, freq_bins * 0.75),
            np.random.uniform(time_bins * 0.2, time_bins * 0.7),
        )
    if freq_sigma is None:
        freq_sigma = np.random.uniform(freq_bins * 0.08, freq_bins * 0.18)
    if time_sigma is None:
        time_sigma = np.random.uniform(time_bins * 0.12, time_bins * 0.28)

    _add_gaussian_blob(canvas, center[0], center[1], freq_sigma, time_sigma, amplitude)
    return canvas


def generate_type_5(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=2.8):
    """
    Generate a Type 5 solar radio burst shape.

    Returns:
        array: Type 5 burst spectrogram
    """
    canvas = _blank_canvas(freq_bins, time_bins)
    if center is None:
        center = (
            np.random.uniform(freq_bins * 0.3, freq_bins * 0.8),
            np.random.uniform(time_bins * 0.2, time_bins * 0.8),
        )
    if freq_sigma is None:
        freq_sigma = np.random.uniform(freq_bins * 0.06, freq_bins * 0.14)
    if time_sigma is None:
        time_sigma = np.random.uniform(time_bins * 0.05, time_bins * 0.12)

    _add_gaussian_blob(canvas, center[0], center[1], freq_sigma, time_sigma, amplitude)
    return canvas
