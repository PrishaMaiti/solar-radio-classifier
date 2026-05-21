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


def generate_type_6(freq_bins=128, time_bins=256, n_bursts=None):
    """Type 6: sustained bunch of Type 3 bursts."""
    canvas = _blank_canvas(freq_bins, time_bins)

    if n_bursts is None:
        n_bursts = np.random.randint(3, 7)

    # spread several Type 3-style subbursts across the time axis
    times = np.linspace(0, time_bins, n_bursts + 2)[1:-1]
    for t_anchor in times:
        t_offset = int(t_anchor + np.random.uniform(-0.05, 0.05) * time_bins)
        # create a compact Type 3 and add it with reduced amplitude
        sub = generate_type_3(freq_bins=freq_bins, time_bins=time_bins, start_time=max(0, t_offset))
        canvas += gaussian_filter(sub, sigma=1) * np.random.uniform(0.7, 1.05)

    # mild smoothing and texture
    canvas = gaussian_filter(canvas, sigma=1.2)
    return canvas


def generate_type_7(freq_bins=128, time_bins=256):
    """Type 7: bunch of Type 3's plus at least one Type 5 overlay."""
    canvas = _blank_canvas(freq_bins, time_bins)

    # base: sustained Type 3 activity
    base = generate_type_6(freq_bins=freq_bins, time_bins=time_bins)
    canvas += base

    # add one or two Type 5-like broad bands
    n_type5 = np.random.randint(1, 3)
    for _ in range(n_type5):
        start_freq = np.random.uniform(freq_bins * 0.55, freq_bins * 0.85)
        start_time = np.random.uniform(time_bins * 0.05, time_bins * 0.35)
        type5 = generate_type_5(freq_bins=freq_bins, time_bins=time_bins, start_freq=start_freq, start_time=start_time)
        canvas += type5 * np.random.uniform(0.7, 1.1)

    canvas = gaussian_filter(canvas, sigma=0.9)
    return canvas


def generate_type_8(freq_bins=128, time_bins=256):
    """Type 8: previously Type 1 — broadband clustered spikes / storms."""
    # reuse generate_type_1 behavior (renamed class)
    return generate_type_1(freq_bins=freq_bins, time_bins=time_bins)


def generate_rfi(freq_bins=128, time_bins=256, n_lines=None, n_spikes=None):
    """Generate radio-frequency interference (RFI) examples: narrowband lines and impulsive spikes.

    Intended to represent non-astrophysical interference for the 'RFI' class.
    """
    canvas = _blank_canvas(freq_bins, time_bins, background_std=0.02)

    if n_lines is None:
        n_lines = np.random.randint(1, 5)
    if n_spikes is None:
        n_spikes = np.random.randint(4, 12)

    # horizontal narrowband continuous carriers
    for _ in range(n_lines):
        row = int(np.random.uniform(0, freq_bins))
        amp = np.random.uniform(0.12, 0.4)
        width = np.random.uniform(0.6, 2.6)
        for t in range(time_bins):
            _add_gaussian_blob(canvas, row + np.random.normal(0, 0.3), t, width, 0.6, amp * np.random.uniform(0.85, 1.15))

    # impulsive narrow spikes (vertical-ish)
    for _ in range(n_spikes):
        col = int(np.random.uniform(0, time_bins))
        freq_center = np.random.uniform(0, freq_bins)
        freq_span = np.random.uniform(1.0, 6.0)
        amp = np.random.uniform(0.15, 0.5)
        _add_gaussian_blob(canvas, freq_center, col, freq_span, 0.8, amp)

    # occasional wideband sweeps
    if np.random.random() < 0.4:
        start = np.random.uniform(freq_bins * 0.1, freq_bins * 0.9)
        drift = np.random.uniform(-freq_bins * 0.35, freq_bins * 0.35)
        dur = int(np.random.uniform(time_bins * 0.06, time_bins * 0.22))
        t0 = int(np.random.uniform(0, time_bins - dur))
        for i in range(dur):
            f = start + drift * (i / max(dur - 1, 1))
            _add_gaussian_blob(canvas, f + np.random.normal(0, 1.8), t0 + i, np.random.uniform(1.2, 3.5), 0.8, np.random.uniform(0.05, 0.18))

    # stronger artifact texture
    canvas += gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=1.0) * 0.08

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

def generate_type_3(freq_bins=128, time_bins=256, start_freq=None, start_time=None, thickness=1.2, amplitude=1.9, duration_min=30.0, cap_freq_frac=0.83, vertical_minutes=(5.0, 10.0)):
    canvas = _blank_canvas(freq_bins, time_bins)

    # Draw Type 3 as two near-vertical, jagged stems with pointed triangular tops.
    cap_bin = max(4, int(cap_freq_frac * freq_bins))

    for minute in vertical_minutes:
        t_col = int(round((minute / float(duration_min)) * (time_bins - 1)))
        t_col = max(1, min(time_bins - 2, t_col))

        # Stem: frequency changes rapidly while time stays nearly fixed.
        for f in range(cap_bin):
            if np.random.random() < 0.07:
                continue

            # Keep time jitter very small so line appears vertical.
            t_center = t_col + np.random.uniform(-0.2, 0.2)
            f_center = f + np.random.uniform(-1.4, 1.4)
            p = f / max(1, cap_bin - 1)

            stem_amp = amplitude * np.random.uniform(0.02, 0.075) * (0.55 + 0.9 * p)
            _add_gaussian_blob(canvas, f_center, t_center, 0.7, 0.34, stem_amp)

            # Jagged edge texture on both sides of the stem.
            if np.random.random() < 0.28:
                _add_gaussian_blob(
                    canvas,
                    f_center + np.random.uniform(-1.2, 1.2),
                    t_center + np.random.choice([-1.0, 1.0]) * np.random.uniform(0.2, 0.6),
                    0.75,
                    0.28,
                    stem_amp * np.random.uniform(0.35, 0.7),
                )

        # Triangular tip at top of each stem.
        tip_height = 8
        for layer in range(tip_height):
            f_tip = cap_bin + layer
            if f_tip >= freq_bins:
                break

            # Width shrinks with height to form triangle.
            half_width_t = max(0.08, 1.1 - 0.15 * layer)
            tip_amp = amplitude * np.exp(-0.45 * layer) * np.random.uniform(0.11, 0.2)

            for dt in (-half_width_t, 0.0, half_width_t):
                _add_gaussian_blob(
                    canvas,
                    f_tip + np.random.uniform(-0.7, 0.7),
                    t_col + dt + np.random.uniform(-0.08, 0.08),
                    0.75,
                    0.3,
                    tip_amp,
                )

    # Blend while preserving verticalness.
    canvas = gaussian_filter(canvas, sigma=0.45)
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

    # --- Type III precursor ---
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

    # --- Type V main band ---
    type5_start = int(min(time_bins - 1, t0 + np.random.uniform(0, 4)))
    type5_duration = int(np.random.uniform(time_bins * 0.16, time_bins * 0.30))
    type5_end = min(time_bins - 1, type5_start + type5_duration)
    v_times = np.arange(type5_start, type5_end)

    if len(v_times) == 0:
        return canvas

    band_low = np.random.uniform(freq_bins * 0.02, freq_bins * 0.10)
    band_high = np.random.uniform(freq_bins * 0.68, freq_bins * 0.92)
    if band_high <= band_low + 10:
        band_high = band_low + 25

    support = np.zeros((freq_bins, time_bins))

    # time-varying overall brightness / width
    tail_var = gaussian_filter1d(np.random.uniform(0.7, 1.35, size=len(v_times)), sigma=2)
    tail_var = np.clip(tail_var, 0.55, 1.55)

    width_var = gaussian_filter1d(np.random.uniform(0.85, 1.25, size=len(v_times)), sigma=2)
    width_var = np.clip(width_var, 0.75, 1.35)

    # top / bottom boundary jitter
    low_jitter = gaussian_filter1d(np.random.normal(0, 2.5, size=len(v_times)), sigma=1)
    high_jitter = gaussian_filter1d(np.random.normal(0, 3.5, size=len(v_times)), sigma=1)

    # persistent internal patchiness over time/frequency
    patch_field = gaussian_filter(np.random.uniform(0.6, 1.5, size=(freq_bins, len(v_times))), sigma=(2, 1))
    patch_field = np.clip(patch_field, 0.4, 1.7)

    # --- Key fix: persistent row-specific right-edge endpoints ---
    row_end = np.full(freq_bins, type5_end, dtype=float)
    row_fade = np.full(freq_bins, 2.5, dtype=float)
    row_gain = np.ones(freq_bins)

    f0 = max(0, int(band_low))
    f1 = min(freq_bins, int(band_high))

    if f1 > f0:
        band_rows = np.arange(f0, f1)
        norm_band = (band_rows - f0) / max(f1 - f0 - 1, 1)

        # base shape: rows near the middle often persist longer, but with lots of irregularity
        end_frac = 0.62 + 0.22 * np.sin(norm_band * np.pi)
        end_frac += gaussian_filter1d(np.random.normal(0, 0.22, size=len(band_rows)), sigma=1)
        end_frac += np.random.normal(0, 0.08, size=len(band_rows))
        end_frac = np.clip(end_frac, 0.32, 1.05)

        # convert to actual end columns in time
        row_end_vals = type5_start + end_frac * type5_duration

        # extra jaggedness so the right edge is not smooth
        row_end_vals += gaussian_filter1d(np.random.normal(0, 4.0, size=len(band_rows)), sigma=0.8)
        row_end_vals = np.clip(row_end_vals, type5_start + 4, type5_end + 6)

        row_end[f0:f1] = row_end_vals

        # row-specific fade widths so cutoff softness varies by row
        fade_vals = 1.8 + 2.8 * np.random.uniform(0.6, 1.3, size=len(band_rows))
        fade_vals = gaussian_filter1d(fade_vals, sigma=1)
        row_fade[f0:f1] = np.clip(fade_vals, 1.2, 4.8)

        # row-specific brightness
        gain_vals = 0.85 + 0.45 * np.sin(norm_band * np.pi)
        gain_vals += gaussian_filter1d(np.random.normal(0, 0.18, size=len(band_rows)), sigma=1)
        row_gain[f0:f1] = np.clip(gain_vals, 0.55, 1.5)

    for t in v_times:
        i = t - type5_start
        p = (t - type5_start) / max(type5_duration, 1)

        env = (0.45 + 0.55 * np.exp(-0.85 * p)) * tail_var[i]

        low_t = band_low + (band_high - band_low) * 0.02 * p + low_jitter[i]
        high_t = band_high - (band_high - band_low) * 0.12 * p + high_jitter[i]

        fmin = max(0, int(low_t))
        fmax = min(freq_bins, int(high_t))
        if fmax <= fmin:
            continue

        freqs = np.arange(fmin, fmax)
        center = 0.5 * (low_t + high_t)
        half_width = max((high_t - low_t) / 2, 1.0) * width_var[i]

        x = np.abs((freqs - center) / half_width)
        profile = np.clip(1.0 - x**4, 0, 1)

        # row-specific ragged right edge in absolute time
        edge_mask = 1.0 / (1.0 + np.exp((t - row_end[freqs]) / row_fade[freqs]))

        # internal patchiness
        patch = patch_field[freqs, i]

        col_scale = np.random.uniform(0.92, 1.18)
        power = amplitude * 0.24 * env * profile * edge_mask * patch * row_gain[freqs] * col_scale
        power *= np.random.uniform(0.92, 1.10, size=len(freqs))

        # occasional weak spots / holes
        hole_mask = np.random.uniform(0, 1, size=len(freqs))
        hole_factor = np.ones(len(freqs))
        weak_idx = hole_mask < 0.08
        hole_factor[weak_idx] = np.random.uniform(0.15, 0.55, size=np.sum(weak_idx))
        power *= hole_factor

        canvas[fmin:fmax, t] += power
        support[fmin:fmax, t] += profile * edge_mask

    # onset column, but not overwhelmingly strong
    for t in range(type5_start, min(type5_start + 3, time_bins)):
        fmin = max(0, int(band_low))
        fmax = min(freq_bins, int(band_high))
        freqs = np.arange(fmin, fmax)
        center = 0.5 * (band_low + band_high)
        half_width = max((band_high - band_low) / 2, 1.0)
        profile = np.clip(1.0 - np.abs((freqs - center) / half_width)**4, 0, 1)
        onset_patch = gaussian_filter1d(np.random.uniform(0.8, 1.25, size=len(freqs)), sigma=1)
        canvas[fmin:fmax, t] += amplitude * 0.18 * profile * onset_patch
        support[fmin:fmax, t] += profile

    texture = gaussian_filter(np.random.randn(freq_bins, time_bins), sigma=1.3)
    canvas += support * texture * 0.045

    return canvas