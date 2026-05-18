"""
Synthesizer module for generating solar radio burst shapes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d


'''def _blank_canvas(freq_bins, time_bins, background_mean=0.0, background_std=0.2):
    return np.random.normal(background_mean, background_std, (freq_bins, time_bins))'''

#creates a blank canvas with correlated noise to mimic background noise better than random spots of IID noise
def _blank_canvas(freq_bins, time_bins, background_mean=0.0, background_std=0.03):
    noise = np.random.normal(
        background_mean,
        background_std,
        (freq_bins, time_bins)
    )

    # Very mild correlation only
    noise = gaussian_filter(noise, sigma=0.6)

    # faint receiver banding
    banding = np.random.normal(0, 0.01, (freq_bins, 1))

    canvas = noise + banding

    return canvas


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

        if fmax <= fmin or tmax <= tmin:
            continue

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



'''def generate_no_burst(freq_bins=128, time_bins=256, background_mean=0.0, background_std=0.2):
    """
    Generate the quiet-sun background baseline.

    Returns:
        array: Background noise spectrogram
    """
    return _blank_canvas(freq_bins, time_bins, background_mean, background_std)'''
def generate_no_burst(freq_bins=128, time_bins=256, background_mean=0.0, background_std=0.04):
    canvas = _blank_canvas(freq_bins, time_bins, background_mean, background_std)

    freq_gradient = np.linspace(0.03, 0.10, freq_bins)[:, None]
    time_variation = 0.03 * np.sin(np.linspace(0, 2*np.pi, time_bins))[None, :]

    canvas += freq_gradient + time_variation
    return canvas


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
def generate_type_1(freq_bins=128, time_bins=256, num_clusters=14, spikes_per_cluster=(80, 180), band=None, spike_amplitude=1.2):
    canvas = _blank_canvas(freq_bins, time_bins)

    if band is None:
        band_start = int(freq_bins * 0.25)
        band_end = int(freq_bins * 0.7)
    else:
        band_start, band_end = band

    band_start = max(0, min(freq_bins - 1, band_start))
    band_end = max(band_start + 1, min(freq_bins, band_end))
    _add_gaussian_blob(canvas, center_freq=(band_start + band_end) / 2, center_time=time_bins / 2, freq_sigma=(band_end - band_start) * 0.45, time_sigma=time_bins * 0.6, amplitude=0.25)

    # faint underlying Type I storm continuum
    storm_center = (band_start + band_end) / 2
    storm_width = (band_end - band_start) * 0.35

    for t in range(time_bins):
        time_env = 0.4 + 0.6 * np.random.random()
        for f in range(band_start, band_end):
            band_env = np.exp(-((f - storm_center) ** 2) / (2 * storm_width**2))
            canvas[f, t] += 0.025 * time_env * band_env

    texture = gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=2)
    canvas[band_start:band_end, :] += 0.08 * texture[band_start:band_end, :]

    num_lanes = np.random.randint(2, 4)

    for _ in range(num_lanes):
        lane_freq = np.random.uniform(band_start, band_end)
        lane_start = np.random.uniform(0, time_bins * 0.3)
        lane_end = np.random.uniform(time_bins * 0.6, time_bins)

        num_clusters_in_lane = np.random.randint(4, 8)
        for center_time in np.linspace(lane_start, lane_end, num_clusters_in_lane):
            #center_freq = lane_freq + np.random.normal(0, 4)
            lane_slope = np.random.uniform(-0.08, 0.08)
            center_freq = lane_freq + lane_slope * (center_time - lane_start) + np.random.normal(0, 4)
            num_spikes = np.random.randint(*spikes_per_cluster)

            freq_spread = np.random.uniform(2, 5)
            time_spread = np.random.uniform(12, 35)

            spike_freqs = np.random.normal(center_freq, freq_spread, size=num_spikes)
            spike_times = np.random.normal(center_time, time_spread, size=num_spikes)

            for f, t in zip(spike_freqs, spike_times):
                f = int(round(f))
                t = int(round(t))
                if 0 <= f < freq_bins and 0 <= t < time_bins:
                    amp = spike_amplitude * np.random.uniform(0.08, 0.25)
                    _add_gaussian_blob(canvas, f, t, np.random.uniform(1.0, 2.2), np.random.uniform(0.4, 1.0), amp)
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

def generate_type_2(freq_bins=128, time_bins=256, start_freq=None, start_time=None, drift=None, thickness=1.3, amplitude=1.9, add_harmonic=True):
    canvas = _blank_canvas(freq_bins, time_bins)

    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.55, freq_bins * 0.82)
    if start_time is None:
        start_time = np.random.uniform(0, time_bins * 0.2)
    if drift is None:
        drift = np.random.uniform(freq_bins * 0.22, freq_bins * 0.48)

    duration = np.random.uniform(time_bins * 0.3, time_bins * 0.65)
    end_time = min(time_bins - 1, start_time + duration)
    times = np.arange(int(start_time), int(end_time))
    if len(times) == 0:
        return canvas

    progress = (times - start_time) / max(end_time - start_time, 1)

    def make_lane(base_start_freq, base_drift, harmonic=False, allow_split=True):
        curve_power = np.random.uniform(0.55, 0.9)
        freqs = base_start_freq - base_drift * (progress ** curve_power)

        wobble = np.random.uniform(1.5, 4.0) * np.sin(progress * np.random.uniform(1.5, 3.5) * np.pi + np.random.uniform(0, 2*np.pi))
        freqs += wobble
        freqs += gaussian_filter1d(np.random.normal(0, 1.6 if not harmonic else 2.0, size=len(times)), sigma=5)

        keep = np.random.random(len(times)) > (0.05 if not harmonic else 0.18)
        lane_freqs = freqs[keep]
        lane_times = times[keep]
        lane_progress = progress[keep]

        split_lanes = [0.0]
        if allow_split and np.random.random() < 0.45:
            split_offset = np.random.uniform(4.0, 9.0)
            split_lanes = [-0.5 * split_offset, 0.5 * split_offset]

        for offset in split_lanes:
            for f, t, p in zip(lane_freqs + offset, lane_times, lane_progress):
                if 0 <= f < freq_bins and 0 <= t < time_bins:
                    cloud_w = thickness * np.random.uniform(2.8, 5.0) * (1.0 + 0.6 * p)
                    cloud_amp = amplitude * np.random.uniform(0.008, 0.022) * (0.65 if harmonic else 1.0)
                    _add_gaussian_blob(canvas, f, t, cloud_w, cloud_w * 2.2, cloud_amp)

            for f, t, p in zip(lane_freqs + offset, lane_times, lane_progress):
                if 0 <= f < freq_bins and 0 <= t < time_bins:
                    lane_w = thickness * np.random.uniform(1.0, 2.0) * (1.0 + 0.35 * p)
                    lane_amp = amplitude * np.random.uniform(0.02, 0.055) * (0.55 if harmonic else 1.0)
                    _add_gaussian_blob(canvas, f, t, lane_w, lane_w * 1.6, lane_amp)

            lane_band = np.zeros((freq_bins, time_bins))
            for f, t in zip(lane_freqs + offset, lane_times):
                if 0 <= f < freq_bins and 0 <= t < time_bins:
                    _add_gaussian_blob(lane_band, f, t, thickness * 4.0, thickness * 6.0, 0.03)

            texture = gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=2)
            power_texture = np.clip(0.75 + 0.35 * texture, 0.2, 1.4)
            canvas[:] += lane_band * power_texture * (0.08 if not harmonic else 0.045)

    # main fundamental lane
    make_lane(start_freq, drift, harmonic=False, allow_split=True)

    # harmonic only sometimes
    if add_harmonic and np.random.random() < 0.55:
        h_start = min(freq_bins - 1, start_freq * np.random.uniform(1.3, 1.65))
        h_drift = drift * np.random.uniform(0.75, 1.05)
        make_lane(h_start, h_drift, harmonic=True, allow_split=True)

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

def generate_type_3(freq_bins=128, time_bins=256, start_freq=None, start_time=None, thickness=1.2, amplitude=1.9):
    canvas = _blank_canvas(freq_bins, time_bins)

    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.55, freq_bins * 0.9)
    if start_time is None:
        start_time = np.random.uniform(0, time_bins * 0.25)

    mode = np.random.choice(["compact", "spread", "storm"], p=[0.35, 0.4, 0.25])

    if mode == "compact":
        n_subbursts = np.random.randint(5, 9)
        time_jitter = (-4, 10)
        duration_scale = (0.08, 0.16)
        drift_scale = (0.45, 0.75)
        envelope_scale = (4.0, 6.0)
    elif mode == "spread":
        n_subbursts = np.random.randint(6, 12)
        time_jitter = (-6, 45)
        duration_scale = (0.10, 0.22)
        drift_scale = (0.4, 0.7)
        envelope_scale = (5.0, 8.0)
    else:  # storm
        n_subbursts = np.random.randint(10, 18)
        time_jitter = (-8, 80)
        duration_scale = (0.08, 0.18)
        drift_scale = (0.35, 0.65)
        envelope_scale = (5.5, 9.0)

    base_duration = np.random.uniform(time_bins * duration_scale[0], time_bins * duration_scale[1])
    base_drift = np.random.uniform(freq_bins * drift_scale[0], freq_bins * drift_scale[1])

    all_freqs = []
    all_times = []

    for _ in range(n_subbursts):
        local_start_time = start_time + np.random.uniform(*time_jitter)
        local_start_freq = start_freq + np.random.normal(0, 8)

        duration = base_duration * np.random.uniform(0.7, 1.25)
        end_time = min(time_bins - 1, local_start_time + duration)

        times = np.arange(int(max(0, local_start_time)), int(end_time))
        if len(times) < 5:
            continue

        progress = (times - local_start_time) / max(end_time - local_start_time, 1)

        total_drift = base_drift * np.random.uniform(0.85, 1.15)
        curve_power = np.random.uniform(0.6, 0.95)
        freqs = local_start_freq - total_drift * (progress ** curve_power)

        freqs += gaussian_filter1d(np.random.normal(0, 1.5, size=len(times)), sigma=2)

        keep_prob = 0.94 if mode != "storm" else 0.88
        keep = np.random.random(len(times)) < keep_prob
        freqs, times, progress = freqs[keep], times[keep], progress[keep]

        all_freqs.extend(freqs)
        all_times.extend(times)

        for f, t, p in zip(freqs, times, progress):
            if 0 <= f < freq_bins and 0 <= t < time_bins:
                halo_w = thickness * np.random.uniform(2.0, 4.5)
                halo_amp = amplitude * np.random.uniform(0.015, 0.045)
                _add_gaussian_blob(canvas, f, t, halo_w, halo_w * 1.8, halo_amp)

        for f, t, p in zip(freqs, times, progress):
            if 0 <= f < freq_bins and 0 <= t < time_bins:
                lane_w = thickness * np.random.uniform(0.9, 1.8)
                lane_amp = amplitude * np.random.uniform(0.04, 0.11)
                _add_gaussian_blob(canvas, f, t, lane_w, lane_w * 1.4, lane_amp)

        if np.random.random() < 0.55:
            h_offset = np.random.uniform(5, 12)
            h_times = times + np.random.randint(0, 3)
            h_freqs = freqs + h_offset + gaussian_filter1d(np.random.normal(0, 1.0, size=len(freqs)), sigma=2)

            for f, t in zip(h_freqs, h_times):
                if 0 <= f < freq_bins and 0 <= t < time_bins:
                    h_w = thickness * np.random.uniform(0.9, 1.7)
                    h_amp = amplitude * np.random.uniform(0.015, 0.05)
                    _add_gaussian_blob(canvas, f, t, h_w, h_w * 1.3, h_amp)

    if len(all_freqs) > 0:
        envelope = np.zeros((freq_bins, time_bins))
        env_f = np.random.uniform(*envelope_scale)
        env_t = env_f * np.random.uniform(1.2, 2.2)

        for f, t in zip(all_freqs, all_times):
            if 0 <= f < freq_bins and 0 <= t < time_bins:
                _add_gaussian_blob(envelope, f, t, env_f, env_t, 0.01)

        texture = gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=2)
        power_texture = np.clip(0.8 + 0.3 * texture, 0.25, 1.4)
        canvas += envelope * power_texture * (0.55 if mode == "compact" else 0.4 if mode == "spread" else 0.3)

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
def generate_type_4(freq_bins=128, time_bins=256, center=None, freq_sigma=None, time_sigma=None, amplitude=1.9):
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

def generate_type_5(freq_bins=128, time_bins=256, start_freq=None, start_time=None, amplitude=2.0):
    canvas = _blank_canvas(freq_bins, time_bins)

    if start_freq is None:
        start_freq = np.random.uniform(freq_bins * 0.65, freq_bins * 0.9)
    if start_time is None:
        start_time = np.random.uniform(time_bins * 0.12, time_bins * 0.3)

    # Type III precursor
    t0 = start_time
    t1 = min(time_bins - 1, t0 + np.random.uniform(time_bins * 0.05, time_bins * 0.1))
    times3 = np.arange(int(t0), int(t1))

    if len(times3) > 4:
        progress3 = (times3 - t0) / max(t1 - t0, 1)
        drift3 = np.random.uniform(freq_bins * 0.45, freq_bins * 0.75)
        freqs3 = start_freq - drift3 * (progress3 ** np.random.uniform(0.6, 0.9))
        freqs3 += gaussian_filter1d(np.random.normal(0, 1.0, size=len(times3)), sigma=2)

        for f, t in zip(freqs3, times3):
            if 0 <= f < freq_bins and 0 <= t < time_bins:
                _add_gaussian_blob(canvas, f, t, 1.0, 1.6, amplitude * np.random.uniform(0.06, 0.12))

    # Type V: broad frequency band, not tied tightly to Type III endpoint
    type5_start = int(min(time_bins - 1, t0 + np.random.uniform(0, 4)))
    type5_duration = int(np.random.uniform(time_bins * 0.18, time_bins * 0.35))
    type5_end = min(time_bins - 1, type5_start + type5_duration)
    v_times = np.arange(type5_start, type5_end)

    # Pick a broad mid/low frequency band directly
    band_low = np.random.uniform(freq_bins * 0.12, freq_bins * 0.25)
    band_high = np.random.uniform(freq_bins * 0.55, freq_bins * 0.78)

    if band_high <= band_low + 10:
        band_high = band_low + 25

    support = np.zeros((freq_bins, time_bins))

    for t in v_times:
        p = (t - type5_start) / max(type5_end - type5_start, 1)

        # fairly strong at onset, then slowly fades
        env = 0.75 * np.exp(-1.8 * p) + 0.25 * np.exp(-0.3 * p)

        # the band narrows slightly over time
        low_t = band_low + (band_high - band_low) * 0.08 * p
        high_t = band_high - (band_high - band_low) * 0.18 * p

        fmin = max(0, int(low_t))
        fmax = min(freq_bins, int(high_t))
        if fmax <= fmin:
            continue

        freqs = np.arange(fmin, fmax)
        center = 0.5 * (low_t + high_t)
        half_width = max((high_t - low_t) / 2, 1.0)

        # flatter than a Gaussian so it reads as a band, but still has soft edges
        x = np.abs((freqs - center) / half_width)
        profile = np.clip(1.0 - x**4, 0, 1)

        # slightly stronger in the middle of the band
        power = amplitude * 0.13 * env * profile
        power *= np.random.uniform(0.9, 1.1, size=len(freqs))

        canvas[fmin:fmax, t] += power
        support[fmin:fmax, t] += profile

    # Bright onset column across the same band
    for t in range(type5_start, min(type5_start + 4, time_bins)):
        fmin = max(0, int(band_low))
        fmax = min(freq_bins, int(band_high))
        freqs = np.arange(fmin, fmax)
        center = 0.5 * (band_low + band_high)
        half_width = max((band_high - band_low) / 2, 1.0)
        profile = np.clip(1.0 - np.abs((freqs - center) / half_width)**4, 0, 1)
        canvas[fmin:fmax, t] += amplitude * 0.18 * profile
        support[fmin:fmax, t] += profile

    # Internal texture inside the Type V band only
    texture = gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=1.5)
    canvas += support * texture * 0.035

    return canvas