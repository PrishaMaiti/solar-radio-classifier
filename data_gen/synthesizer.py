"""
Synthesizer module for generating solar radio burst shapes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d


'''def _blank_canvas(freq_bins, time_bins, background_mean=0.0, background_std=0.2):
    return np.random.normal(background_mean, background_std, (freq_bins, time_bins))'''

#creates a blank canvas with correlated noise to mimic background noise better than random spots of IID noise
def _blank_canvas(
    freq_bins,
    time_bins,
    background_mean=0.0,
    background_std=0.2
):

    noise = np.random.normal(
        background_mean,
        background_std,
        (freq_bins, time_bins)
    )

    # Correlate nearby pixels
    noise = gaussian_filter(noise, sigma=1.2)

    return noise


def _add_gaussian_blob(canvas, center_freq, center_time, freq_sigma, time_sigma, amplitude):
    freq_idx, time_idx = np.indices(canvas.shape)
    blob = amplitude * np.exp(
        -(((freq_idx - center_freq) ** 2) / (2.0 * freq_sigma**2)
          + ((time_idx - center_time) ** 2) / (2.0 * time_sigma**2))
    )
    canvas += blob


'''def _draw_line(canvas, start_freq, start_time, end_freq, end_time, thickness, amplitude):
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
    canvas += amplitude * np.exp(-(distance**2) / (2.0 * thickness**2))'''

# rather than drawing a specific line, draws a trajectory of points along the line with some thickness, which creates a more natural burst shape
def _draw_trajectory(canvas, freqs, times, thickness=1.5, amplitude=4.0):

    radius = int(thickness * 3)

    for f, t in zip(freqs, times):

        f = int(round(f))
        t = int(round(t))

        fmin = max(0, f - radius)
        fmax = min(canvas.shape[0], f + radius + 1)

        tmin = max(0, t - radius)
        tmax = min(canvas.shape[1], t + radius + 1)

        freq_idx, time_idx = np.indices((fmax - fmin, tmax - tmin))

        freq_idx += fmin
        time_idx += tmin

        blob = amplitude * np.exp(
            -(
                ((freq_idx - f) ** 2)
                + ((time_idx - t) ** 2)
            ) / (2.0 * thickness**2)
        )

        canvas[fmin:fmax, tmin:tmax] += blob

    return canvas



def generate_no_burst(freq_bins=128, time_bins=256, background_mean=0.0, background_std=0.2):
    """
    Generate the quiet-sun background baseline.

    Returns:
        array: Background noise spectrogram
    """
    return _blank_canvas(freq_bins, time_bins, background_mean, background_std)


'''def generate_type_1(freq_bins=128, time_bins=256, num_spikes=50, band=None, spike_amplitude=3.0):
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
    return canvas'''

#type 1 designed to mimic multiple bandwidth bursts rather than independent noise spikes
def generate_type_1(freq_bins=128, time_bins=256, num_clusters=8, spikes_per_cluster=(20, 80), band=None, spike_amplitude=3.0,):
    canvas = _blank_canvas(freq_bins, time_bins)

    if band is None:
        band_start = int(freq_bins * 0.25)
        band_end = int(freq_bins * 0.7)
    else:
        band_start, band_end = band

    band_start = max(0, min(freq_bins - 1, band_start))
    band_end = max(band_start + 1, min(freq_bins, band_end))

    for _ in range(num_clusters):

        # Cluster center
        center_freq = np.random.uniform(band_start, band_end)
        center_time = np.random.uniform(0, time_bins)

        num_spikes = np.random.randint(*spikes_per_cluster)

        # Local spread
        freq_spread = np.random.uniform(2, 8)
        time_spread = np.random.uniform(3, 15)

        spike_freqs = np.random.normal(
            center_freq,
            freq_spread,
            size=num_spikes
        )

        spike_times = np.random.normal(
            center_time,
            time_spread,
            size=num_spikes
        )

        for f, t in zip(spike_freqs, spike_times):
            f = int(round(f))
            t = int(round(t))
            if (0 <= f < freq_bins and 0 <= t < time_bins):
                canvas[f, t] += (
                    spike_amplitude
                    * np.random.uniform(0.5, 1.5)
                )
    return canvas


'''def generate_type_2(
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
    _draw_trajectory(canvas, start_freq, start_time, end_freq, end_time, thickness, amplitude)

    if add_harmonic:
        harmonic_start = min(freq_bins - 1, start_freq * 2.0)
        harmonic_end = max(0, harmonic_start - drift * 1.15)
        _draw_trajectory(canvas, harmonic_start, start_time + 3, harmonic_end, end_time + 3, thickness, amplitude * 0.55)

    return canvas'''

def generate_type_2(freq_bins=128, time_bins=256, start_freq=None, start_time=None, drift=None, thickness=1.8, amplitude=4.0, add_harmonic=True):
    canvas = _blank_canvas(freq_bins, time_bins)

    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.45, freq_bins * 0.8)

    if start_time is None:
        start_time = np.random.uniform(0, time_bins * 0.25)

    if drift is None:
        drift = np.random.uniform(freq_bins * 0.25, freq_bins * 0.55)

    end_time = min(
        time_bins - 1,
        start_time + np.random.uniform(time_bins * 0.35, time_bins * 0.8)
    )

    times = np.arange(start_time, end_time).astype(int)

    freqs = start_freq - drift * (times - start_time)

    # curvature (shock deceleration)
    freqs -= 0.002 * (times - start_time) ** 2

    # jitter
    freqs += np.random.normal(0, 0.8, size=len(times))

    # smooth
    freqs = gaussian_filter1d(freqs, sigma=1)

    # dropout (fragmentation)
    mask = np.random.random(len(times)) > 0.15
    freqs = freqs[mask]
    times = times[mask]

    # main lane
    for i, (f, t) in enumerate(zip(freqs, times)):
        if 0 <= f < freq_bins and 0 <= t < time_bins:
            #canvas[int(f), int(t)] += amplitude
            _add_gaussian_blob(canvas, f, t, freq_sigma=thickness, time_sigma=thickness * 1.5, amplitude=amplitude)

    # harmonic
    if add_harmonic:
        h_times = times + np.random.randint(2, 6)

        h_freqs = (start_freq * 2.0) - drift * 0.9 * (times - start_time)
        h_freqs += np.random.normal(0, 1.2, size=len(times))
        _draw_trajectory(canvas, h_freqs, h_times, thickness * 0.8, amplitude * 0.45)

    return canvas


'''def generate_type_3(
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
    return canvas'''

def generate_type_3(freq_bins=128, time_bins=256, start_freq=None, start_time=None, thickness=1.1, amplitude=4.5):
    canvas = _blank_canvas(freq_bins, time_bins)

    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.35, freq_bins * 0.85)

    if start_time is None:
        start_time = np.random.uniform(0, time_bins * 0.35)

    end_time = min(time_bins - 1, start_time + np.random.uniform(time_bins * 0.08, time_bins * 0.2))

    times = np.arange(start_time, end_time)

    drift_rate = np.random.uniform(freq_bins * 0.7, freq_bins * 1.2)

    freqs = start_freq - drift_rate * (times - start_time)
    freqs += 0.002 * (times - start_time)**2
    freqs += np.random.normal(0, 1.2, size=len(times))

    mask = np.random.random(len(times)) > 0.2
    freqs = freqs[mask]
    times = times[mask]

    for f, t in zip(freqs, times):
        if 0 <= f < freq_bins and 0 <= t < time_bins:
            _add_gaussian_blob(canvas, f, t, thickness*0.8, thickness*1.8, amplitude)

    if np.random.random() < 0.7:
        h_freqs = freqs * 1.9 + np.random.normal(0, 1.5, size=len(freqs))
        h_times = times + np.random.randint(0, 3)
        for f, t in zip(h_freqs, h_times):
            if 0 <= f < freq_bins and 0 <= t < time_bins:
                _add_gaussian_blob(canvas, f, t, thickness*0.7, thickness*1.3, amplitude*0.25)

    return canvas


'''def generate_type_4(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=3.5):
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
    return canvas'''
from scipy.ndimage import gaussian_filter
import numpy as np

def generate_type_4(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=3.5):
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

    times = np.arange(time_bins)

    envelope = 0.3 + 0.7 * np.sin(np.pi * times / time_bins)

    for t in range(time_bins):

        width = freq_sigma * (0.6 + 0.8*np.sin(np.pi * t / time_bins))

        f0 = center[0] - 0.03 * t + np.random.normal(0, 0.5)

        for f in range(freq_bins):

            canvas[f, t] += amplitude * envelope[t] * np.exp(
                -((f - f0) ** 2) / (2 * width**2)
            )

    # internal structure
    canvas += gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=1) * 0.2

    return canvas


'''def generate_type_5(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=2.8):
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
    return canvas'''

from scipy.ndimage import gaussian_filter
import numpy as np

def generate_type_5(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=2.8):
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

    # --- Type III precursor influence ---
    parent_start = np.random.uniform(0, time_bins * 0.4)
    parent_freq = np.random.uniform(freq_bins * 0.4, freq_bins * 0.9)

    times = np.arange(parent_start, parent_start + np.random.randint(30, 80))
    drift = np.random.uniform(freq_bins * 0.6, freq_bins * 1.1)

    freqs = parent_freq - drift * (times - parent_start)
    freqs += np.random.normal(0, 1.5, size=len(times))

    for f, t in zip(freqs, times):
        if 0 <= f < freq_bins and 0 <= t < time_bins:
            _add_gaussian_blob(canvas, f, t, 1.0, 2.0, amplitude * 0.3)

    # --- main Type V continuum ---
    low_band = int(freq_bins * 0.2)
    mid_band = int(freq_bins * 0.6)

    for t in range(time_bins):

        decay = np.exp(-t / (time_bins * 0.6))

        for f in range(low_band, mid_band):

            canvas[f, t] += amplitude * decay * np.random.uniform(0.5, 1.2)

    # --- internal structure ---
    canvas += gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=1.5) * 0.25

    # --- optional hidden Type II ---
    if np.random.random() < 0.4:

        t0 = np.random.randint(0, time_bins // 2)
        t1 = t0 + np.random.randint(40, 100)

        times2 = np.arange(t0, min(t1, time_bins))

        freqs2 = np.random.uniform(freq_bins * 0.5, freq_bins * 0.8) - 0.5 * (times2 - t0)

        for f, t in zip(freqs2, times2):
            if 0 <= f < freq_bins and 0 <= t < time_bins:
                _add_gaussian_blob(canvas, f, t, 1.2, 2.5, amplitude * 0.2)

    # --- base continuum support ---
    _add_gaussian_blob(
        canvas,
        center[0],
        center[1],
        freq_sigma * 1.5,
        time_sigma * 2.0,
        amplitude * 0.5
    )

    return canvas