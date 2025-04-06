"""
Adaptive HRTF Compression Module

This module implements an adaptive compression scheme for Head-Related Transfer Functions (HRTFs).
It processes impulse responses from SOFA files by computing frequency responses, applying adaptive
Wiener filtering, and selecting a set of control points that allow reconstructing the frequency
response within an acceptable error threshold. The module uses various signal processing, 
optimization, and interpolation methods (including cubic and PCHIP splines), and leverages parallel
computing via Numba for efficient error metric computation.

Main components:
    - Frequency response computation and filtering.
    - Reconstruction using control points with error metrics.
    - ERB (Equivalent Rectangular Bandwidth) based segmentation and error analysis.
    - Control point detection, refinement, and merging strategies.
    - Adaptive compression applied to HRTFs loaded from SOFA files.
    - CSV file output for control points and compression rates.
    - Plotting and interactive visualization using Matplotlib.

Usage:
    Run the module as a script to open a SOFA file, compress the HRTFs, and display interactive plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.signal import wiener
from scipy.interpolate import CubicSpline, PchipInterpolator, InterpolatedUnivariateSpline
from scipy.optimize import dual_annealing, minimize_scalar
import sofa
import csv
from numba import njit, prange
from matplotlib import rc_context


# -----------------------------
# Frequency Response & Filtering
# -----------------------------
def compute_frequency_response(ir, fs, fft_len=None):
    """
    Compute the frequency response (magnitude in dB) from an impulse response (IR).

    Parameters:
        ir (np.ndarray): Impulse response signal.
        fs (float): Sampling rate in Hz.
        fft_len (int, optional): Length for FFT computation. If None, uses the length of IR.

    Returns:
        freqs (np.ndarray): Frequencies (in Hz) corresponding to the FFT bins up to 20 kHz.
        mag_db (np.ndarray): Magnitude spectrum in decibels.
    
    Process:
        - Pads the IR if necessary.
        - Computes the FFT using a real FFT (rfft).
        - Converts magnitude to dB (with a small constant added to avoid log(0)).
        - Masks out frequencies higher than 20 kHz.
    """
    fft_len = fft_len or len(ir)
    # Compute frequency bins for the FFT result
    freqs = np.fft.rfftfreq(fft_len, d=1/fs)
    padded = np.zeros(fft_len)
    n = min(len(ir), fft_len)
    padded[:n] = ir[:n]
    fft_res = np.fft.rfft(padded)
    # Compute magnitude in dB with a small constant to prevent log(0)
    mag_db = 20 * np.log10(np.abs(fft_res) + 1e-10)
    # Limit frequencies to 20 kHz
    mask = freqs <= 20000
    return freqs[mask], mag_db[mask]


def adaptive_wiener(mag, base_win=11, err_thresh=1.0, alt_win=7, noise=1e-1):
    """
    Apply an adaptive Wiener filter to a magnitude spectrum.

    Parameters:
        mag (np.ndarray): Input magnitude spectrum (in dB).
        base_win (int): Window size for the initial Wiener filtering.
        err_thresh (float): Error threshold to decide if a local re-filtering is needed.
        alt_win (int): Alternative window size for local re-filtering when error is high.
        noise (float): Noise power estimate for Wiener filtering.

    Returns:
        filtered (np.ndarray): The filtered (smoothed) magnitude spectrum.

    How it works:
        - First, a base Wiener filter is applied to the entire signal.
        - Then, for each frequency bin, if the absolute difference (error) between the original
          and the filtered value exceeds err_thresh, a local segment is re-filtered with a smaller window.
    """
    # Apply base Wiener filtering
    base_filtered = wiener(mag, mysize=base_win, noise=noise)
    error = np.abs(mag - base_filtered)
    filtered = base_filtered.copy()
    # Check each point to see if local re-filtering is required
    for i in range(len(mag)):
        if error[i] > err_thresh:
            half = alt_win // 2
            start = max(i - half, 0)
            end = min(i + half + 1, len(mag))
            local_seg = mag[start:end]
            local_filtered = wiener(local_seg, mysize=alt_win, noise=noise)
            filtered[i] = local_filtered[i - start]
    return filtered


def apply_filter(mag, win_size=11, err_thresh=1.0, alt_win=7, noise=1e-1):
    """
    Wrapper for applying the adaptive Wiener filter.

    Parameters:
        mag (np.ndarray): Input magnitude spectrum (in dB).
        win_size (int): Base window size for Wiener filtering.
        err_thresh (float): Threshold for local error.
        alt_win (int): Alternative window size for re-filtering.
        noise (float): Noise power estimate.

    Returns:
        np.ndarray: The filtered (smoothed) magnitude spectrum.
    
    """
    return adaptive_wiener(mag, base_win=win_size, err_thresh=err_thresh, alt_win=alt_win, noise=noise)


# -----------------------------
# Reconstruction & Error Metrics
# -----------------------------
def reconstruct_hrtf(control_pts, method='pchip', grid=None):
    """
    Reconstruct the HRTF magnitude response using interpolation on control points.

    Parameters:
        control_pts (np.ndarray): Array of control points, each a [frequency, magnitude] pair.
        method (str): Interpolation method to use ('cubic' for CubicSpline, otherwise PchipInterpolator).
        grid (np.ndarray, optional): Frequency grid for interpolation. If None, a default grid of 513 points is used.

    Returns:
        grid (np.ndarray): Frequency grid used for reconstruction.
        recon (np.ndarray): Reconstructed magnitude response on the grid.

    How it works:
        - Sorts control points by frequency.
        - Chooses an interpolation function (cubic spline or PCHIP) based on the method.
        - Interpolates the magnitude over the given or default grid.
    """
    # Ensure control points are sorted by frequency
    cp_sorted = control_pts[np.argsort(control_pts[:, 0])]
    cp_freqs = cp_sorted[:, 0]
    cp_mags = cp_sorted[:, 1]
    # Select interpolation method
    if method == 'cubic':
        spline = CubicSpline(cp_freqs, cp_mags, extrapolate=True)
    else:
        spline = PchipInterpolator(cp_freqs, cp_mags, extrapolate=True)
    # Use default grid if none provided
    if grid is None:
        grid = np.linspace(cp_freqs[0], cp_freqs[-1], 513)
    return grid, spline(grid)


@njit(parallel=True, cache=True)
def parallel_spectral_distortion(orig, recon):
    """
    Compute the spectral distortion between the original and reconstructed signals.

    Parameters:
        orig (np.ndarray): Original (smoothed) magnitude spectrum.
        recon (np.ndarray): Reconstructed magnitude spectrum.

    Returns:
        float: The root mean square error between the original and reconstructed signals.

    Numba is used here for more speed.
    """
    n = orig.shape[0]
    total = 0.0
    for i in prange(n):
        diff = orig[i] - recon[i]
        total += diff * diff
    return np.sqrt(total / n)


@njit(parallel=True, cache=True)
def parallel_calc_segment_error(freqs, orig, recon, low, high):
    """
    Calculate the maximum absolute error in a segment of the frequency response.

    Parameters:
        freqs (np.ndarray): Frequency grid.
        orig (np.ndarray): Original (smoothed) magnitude spectrum.
        recon (np.ndarray): Reconstructed magnitude spectrum.
        low (float): Lower bound of the frequency segment.
        high (float): Upper bound of the frequency segment.

    Returns:
        float: Maximum absolute error within the specified segment.

    Numba is used here for more speed.
    """
    n = freqs.shape[0]
    error_val = 0.0
    for i in prange(n):
        if freqs[i] >= low and freqs[i] <= high:
            diff = abs(orig[i] - recon[i])
            if diff > error_val:
                error_val = diff
    return error_val


# Aliases to use our parallel implementations
spectral_distortion = parallel_spectral_distortion
calc_segment_error = parallel_calc_segment_error


def enforce_endpoints(cp, freqs, smooth):
    """
    Enforce that the first and last control points match the endpoints of the smoothed spectrum.

    Parameters:
        cp (np.ndarray): Array of control points (each a [frequency, magnitude] pair).
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.

    Returns:
        np.ndarray: Control points with the endpoints adjusted.
    """
    cp = cp[np.argsort(cp[:, 0])]
    cp[0, 1] = smooth[0]
    cp[-1, 1] = smooth[-1]
    return cp


# -----------------------------
# ERB and Band Spectral Distortion Functions
# -----------------------------
def frequency_to_erb(f):
    """
    Convert a frequency (Hz) to its equivalent ERB (Equivalent Rectangular Bandwidth) scale.

    Parameters:
        f (float): Frequency in Hz.

    Returns:
        float: ERB value corresponding to the frequency.
    """
    return 24.7 * (4.37 * f / 1000 + 1)


def erb_to_frequency(erb):
    """
    Convert an ERB value back to frequency (Hz).

    Parameters:
        erb (float): ERB value.

    Returns:
        float: Frequency in Hz corresponding to the given ERB.
    """
    return (erb / 24.7 - 1) * 1000 / 4.37


def get_erb_bands(freq_min=20, freq_max=20000, n_bands=24):
    """
    Divide the frequency range into ERB-based bands.

    Parameters:
        freq_min (float): Minimum frequency in Hz.
        freq_max (float): Maximum frequency in Hz.
        n_bands (int): Number of bands to create.

    Returns:
        erb_bounds (np.ndarray): ERB bounds for the bands.
        centers (np.ndarray): Center frequencies (geometric mean of bounds) for each band.
    """
    erb_min = frequency_to_erb(freq_min)
    erb_max = frequency_to_erb(freq_max)
    # Create evenly spaced ERB boundaries
    erb_bounds = np.linspace(erb_min, erb_max, n_bands + 1)
    # Convert ERB bounds back to Hz
    bounds_hz = np.array([erb_to_frequency(b) for b in erb_bounds])
    # Compute geometric mean for each band as its center
    centers = np.array([np.sqrt(bounds_hz[i] * bounds_hz[i+1]) for i in range(len(bounds_hz)-1)])
    return erb_bounds, centers


def calculate_band_sd(freqs, orig, recon, erb_bounds):
    """
    Calculate the spectral distortion (SPD) for each ERB band.

    Parameters:
        freqs (np.ndarray): Frequency grid.
        orig (np.ndarray): Original (smoothed) magnitude spectrum.
        recon (np.ndarray): Reconstructed magnitude spectrum.
        erb_bounds (np.ndarray): ERB boundaries in Hz.

    Returns:
        list: RMS error (spectral distortion) for each band. If no frequencies are in a band, np.nan is returned.
    """
    sd_bands = []
    # Loop over each ERB band defined by consecutive bounds
    for i in range(len(erb_bounds)-1):
        low, high = erb_bounds[i], erb_bounds[i+1]
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            rms = np.sqrt(np.mean((orig[mask] - recon[mask]) ** 2))
            sd_bands.append(rms)
        else:
            sd_bands.append(np.nan)
    return sd_bands


# -----------------------------
# Optimized Control Point Processing
# -----------------------------
def find_control_points_via_second_derivative(freqs, mag, threshold=0.1):
    """
    Identify control points based on zeros in the first and second derivative.

    Parameters:
        freqs (np.ndarray): Frequency grid.
        mag (np.ndarray): Smoothed magnitude spectrum.
        threshold (float): Minimum absolute value of the second derivative to consider as a significant inflection.

    Returns:
        np.ndarray: Array of indices in 'freqs' that are selected as control points.

    How it works:
        - Attempts to fit a spline to the data.
        - Finds zeros of the first derivative (potential extrema) and significant inflections from the second derivative.
        - If spline fitting fails, it falls back to using gradient estimation.
        - Finally, ensures that the start and end of the frequency range are included.
    """
    try:
        spline = InterpolatedUnivariateSpline(freqs, mag, k=4)
        first_deriv_zeros = spline.derivative(1).roots()
        second_deriv_zeros = spline.derivative(2).roots()
        second_deriv_vals = np.abs(spline.derivative(2)(second_deriv_zeros))
        significant_inflections = second_deriv_zeros[second_deriv_vals > threshold]
    except:
        # Fallback using numerical gradients if spline fails
        first_deriv = np.gradient(mag, freqs)
        first_deriv_zeros = freqs[np.where(np.diff(np.sign(first_deriv)) != 0)[0]]
        second_deriv = np.gradient(first_deriv, freqs)
        second_sign_change = np.where(np.diff(np.sign(second_deriv)) != 0)[0]
        second_deriv_zeros = freqs[second_sign_change]
        second_deriv_vals = np.abs([second_deriv[i] for i in second_sign_change if i < len(second_deriv)])
        if len(second_deriv_vals) > 0:
            mask = second_deriv_vals > threshold * np.max(np.abs(second_deriv))
            significant_inflections = second_deriv_zeros[mask] if len(mask) == len(second_deriv_zeros) else second_deriv_zeros
        else:
            significant_inflections = second_deriv_zeros
    # Combine critical points from both the first derivative zeros and significant second derivative inflections
    all_critical_pts = np.concatenate([first_deriv_zeros, significant_inflections])
    # Map the critical frequencies to indices in the original frequency grid
    indices = np.unique([np.argmin(np.abs(freqs - r)) for r in all_critical_pts if freqs[0] <= r <= freqs[-1]])
    # Ensure the endpoints are included
    if 0 not in indices:
        indices = np.insert(indices, 0, 0)
    if (len(freqs) - 1) not in indices:
        indices = np.append(indices, len(freqs) - 1)
    return indices


def refine_cp_by_segment(cp, freqs, smooth, method='pchip', seg_thresh=0.9, min_distance=5):
    """
    Refine control points by checking segments for excessive error and inserting new control points.

    Parameters:
        cp (np.ndarray): Array of current control points [frequency, magnitude].
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        method (str): Interpolation method for reconstruction ('pchip' or 'cubic').
        seg_thresh (float): Threshold relative error to trigger insertion of a new control point.
        min_distance (float): Minimum frequency separation required between control points.

    Returns:
        np.ndarray: Refined control points with additional points inserted where necessary.

    How it works:
        - Iteratively reconstructs the magnitude response between control points.
        - For any segment where the maximum error exceeds the threshold,
          a new control point is inserted at the frequency with maximum error.
        - Endpoints are enforced after each insertion.
    """
    cp = cp.copy()
    cp[0, 1] = smooth[0]
    cp[-1, 1] = smooth[-1]
    improved = True
    while improved:
        improved = False
        cp = cp[np.argsort(cp[:, 0])]
        _, recon = reconstruct_hrtf(cp, method, freqs)
        # Loop over each adjacent pair of control points
        for i in range(len(cp) - 1):
            low_f, high_f = cp[i, 0], cp[i+1, 0]
            seg_err = calc_segment_error(freqs, smooth, recon, low_f, high_f)
            # If segment error exceeds threshold, insert a new control point in that segment
            if seg_err > seg_thresh * 1.2:
                mask = (freqs >= low_f) & (freqs <= high_f)
                if not np.any(mask):
                    continue
                seg_freqs = freqs[mask]
                seg_errs = np.abs(smooth[mask] - recon[mask])
                idx = np.argmax(seg_errs)
                new_f = seg_freqs[idx]
                new_m = smooth[mask][idx]
                # Only add if the new frequency is sufficiently separated from existing control points
                if all(np.abs(new_f - existing_cp[0]) > min_distance for existing_cp in cp):
                    cp = np.vstack([cp, [new_f, new_m]])
                    improved = True
                    break
        if improved:
            cp = enforce_endpoints(cp, freqs, smooth)
    return cp


def enhanced_prune_control_points(cp, freqs, smooth, method='pchip', threshold=1.0, dense_factor=12):
    """
    Prune redundant control points while keeping the reconstruction error within a threshold.

    Parameters:
        cp (np.ndarray): Current control points array.
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        method (str): Interpolation method ('pchip' or 'cubic').
        threshold (float): Maximum allowed error threshold.
        dense_factor (int): Factor to create a denser grid for more robust error evaluation.

    Returns:
        np.ndarray: Pruned control points with redundant points removed.
    
    How it works:
        - A dense frequency grid is created.
        - Iteratively, the function attempts to remove each interior control point and evaluates the
          increase in spectral distortion.
        - If removal keeps the error below the threshold and minimizes error increase, the point is removed.
    """
    cp = cp.copy()
    # Create a dense frequency grid for accurate error evaluation
    dense_grid = np.linspace(freqs[0], freqs[-1], len(freqs) * dense_factor)
    baseline = np.interp(dense_grid, freqs, smooth)
    while True:
        cp = cp[np.argsort(cp[:, 0])]
        if len(cp) <= 2:
            break
        best_idx = None
        min_error_increase = float('inf')
        _, baseline_recon = reconstruct_hrtf(cp, method, dense_grid)
        baseline_error = spectral_distortion(baseline, baseline_recon)
        # Try removing each interior control point
        for i in range(1, len(cp)-1):
            candidate = np.delete(cp, i, axis=0)
            _, recon = reconstruct_hrtf(candidate, method, dense_grid)
            new_error = spectral_distortion(baseline, recon)
            error_increase = new_error - baseline_error
            if new_error < threshold and error_increase < min_error_increase:
                min_error_increase = error_increase
                best_idx = i
        if best_idx is None:
            break
        cp = np.delete(cp, best_idx, axis=0)
    return cp


def erb_aware_pruning(cp, freqs, smooth, method='pchip', global_thresh=1.0):
    """
    Prune control points based on ERB band-specific error thresholds.

    Parameters:
        cp (np.ndarray): Current control points array.
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        method (str): Interpolation method ('pchip' or 'cubic').
        global_thresh (float): Base threshold for error that can be adjusted based on the ERB band.

    Returns:
        np.ndarray: Control points after ERB-aware pruning.

    How it works:
        - Determines ERB bands and sets thresholds that vary depending on the center frequency.
        - Iteratively removes a control point if the reconstruction error in all ERB bands stays below the threshold.
    """
    erb_bounds, _ = get_erb_bands(20, 20000, 24)
    erb_thresholds = []
    # Set different thresholds based on frequency range
    for i in range(len(erb_bounds)-1):
        center = np.sqrt(erb_bounds[i] * erb_bounds[i+1])
        if center < 200 or center > 10000:
            erb_thresholds.append(global_thresh * 1.2)
        elif 500 <= center <= 4000:
            erb_thresholds.append(global_thresh * 0.9)
        else:
            erb_thresholds.append(global_thresh)
    cp = cp.copy()
    removal_possible = True
    while removal_possible and len(cp) > 2:
        removal_possible = False
        cp = cp[np.argsort(cp[:, 0])]
        # Try removing each interior control point
        for i in range(1, len(cp)-1):
            candidate = np.delete(cp, i, axis=0)
            _, recon = reconstruct_hrtf(candidate, method, freqs)
            band_errors = calculate_band_sd(freqs, smooth, recon, erb_bounds)
            all_bands_ok = True
            for j, error in enumerate(band_errors):
                if not np.isnan(error) and error > erb_thresholds[j]:
                    all_bands_ok = False
                    break
            if all_bands_ok:
                cp = candidate
                removal_possible = True
                break
    return cp


def post_process_control_points(cp, freqs, smooth, method='pchip', error_tol=1.0):
    """
    Post-process control points by removing or merging points while maintaining the error tolerance.

    Parameters:
        cp (np.ndarray): Array of control points.
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        method (str): Interpolation method.
        error_tol (float): Maximum allowed reconstruction error after processing.

    Returns:
        np.ndarray: Control points after post processing.

    How it works:
        - First attempts to remove a control point if doing so keeps the max error within tolerance.
        - If no removal is possible, attempts to merge adjacent control points.
        - Iterates until no further changes are possible.
    """
    changed = True
    while changed:
        changed = False
        cp = cp[np.argsort(cp[:, 0])]
        # Try removal of one point at a time
        for i in range(1, len(cp) - 1):
            candidate = np.delete(cp, i, axis=0)
            _, recon = reconstruct_hrtf(candidate, method, freqs)
            max_err = np.max(np.abs(smooth - recon))
            if max_err <= error_tol:
                cp = candidate
                changed = True
                break
        # If no removal was successful, try merging adjacent points
        if not changed:
            for i in range(1, len(cp) - 1):
                merged_freq = np.sqrt(cp[i, 0] * cp[i+1, 0])
                merged_mag = (cp[i, 1] + cp[i+1, 1]) / 2
                candidate = np.delete(cp, i+1, axis=0)
                candidate[i, :] = [merged_freq, merged_mag]
                candidate = candidate[np.argsort(candidate[:, 0])]
                _, recon = reconstruct_hrtf(candidate, method, freqs)
                max_err = np.max(np.abs(smooth - recon))
                if max_err <= error_tol:
                    cp = candidate
                    changed = True
                    break
    return cp


def optimized_merge_control_points_grid(cp, freqs, smooth, method='pchip', error_tol=1.0, grid_size=50):
    """
    Merge control points by optimizing the position of new control points over a grid.

    Parameters:
        cp (np.ndarray): Current control points.
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        method (str): Interpolation method.
        error_tol (float): Maximum allowed error tolerance.
        grid_size (int): Number of candidate frequencies between each pair of control points.

    Returns:
        np.ndarray: Control points after merging based on grid optimization.

    How it works:
        - For each adjacent control point pair, candidate new points are evaluated on a dense grid.
        - The candidate with the minimum error (if within tolerance) is inserted between the points.
        - Otherwise, the next point is simply added.
    """
    cp = cp[np.argsort(cp[:, 0])]
    new_cp = [cp[0]]
    i = 0
    while i < len(cp) - 1:
        p1 = cp[i]
        p2 = cp[i+1]
        # Generate candidate frequencies between the current two points
        candidates = np.linspace(p1[0], p2[0], grid_size)
        candidate_errors = []
        for cf in candidates:
            # Interpolate the magnitude at the candidate frequency
            cand_mag = np.interp(cf, freqs, smooth)
            candidate_set = np.vstack((np.array(new_cp, dtype=float),
                                        np.array([[cf, cand_mag]], dtype=float),
                                        cp[i+2:]))
            candidate_set = candidate_set[np.argsort(candidate_set[:, 0])]
            # Ensure the candidate set is valid (monotonic frequencies)
            if np.any(np.diff(candidate_set[:, 0]) <= 0):
                err_val = np.inf
            else:
                candidate_set = np.unique(candidate_set, axis=0)
                if candidate_set.shape[0] < 2:
                    err_val = np.inf
                else:
                    _, recon_candidate = reconstruct_hrtf(candidate_set, method, freqs)
                    err_val = np.max(np.abs(smooth - recon_candidate))
            candidate_errors.append(err_val)
        candidate_errors = np.array(candidate_errors)
        min_err = candidate_errors.min()
        best_cf = candidates[candidate_errors.argmin()]
        if min_err <= error_tol:
            new_cp.append([best_cf, float(np.interp(best_cf, freqs, smooth))])
            i += 2
        else:
            new_cp.append(p2)
            i += 1
    return np.array(new_cp)


def dp_merge_control_points(freqs, smooth, error_tol=1.0):
    """
    Merge control points using a dynamic programming approach to minimize the number of points.

    Parameters:
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        error_tol (float): Maximum allowed error for each segment.

    Returns:
        np.ndarray: Merged control points as a two-column array [frequency, magnitude].

    How it works:
        - Uses dynamic programming (DP) to decide which points to keep such that the error remains within tolerance.
        - The DP array stores the minimum number of segments needed from each index.
    """
    N = len(freqs)
    dp = [np.inf] * N
    parent = [-1] * N
    dp[N-1] = 0
    # Backward dynamic programming to decide segmentation
    for i in range(N-2, -1, -1):
        for j in range(i+1, N):
            seg_freqs = freqs[i:j+1]
            interp = PchipInterpolator([freqs[i], freqs[j]], [smooth[i], smooth[j]])
            recon_seg = interp(seg_freqs)
            err = np.max(np.abs(smooth[i:j+1] - recon_seg))
            if err <= error_tol:
                if 1 + dp[j] < dp[i]:
                    dp[i] = 1 + dp[j]
                    parent[i] = j
    indices = []
    i = 0
    while i != -1 and i < N:
        indices.append(i)
        i = parent[i]
        if i == -1:
            break
    return np.column_stack((freqs[indices], smooth[indices]))


def combined_merge_control_points(cp, freqs, smooth, spline_method='pchip', error_tol=1.0):
    """
    Combine two merging strategies (grid-based and DP-based) and select the one with fewer points.

    Parameters:
        cp (np.ndarray): Initial control points.
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        spline_method (str): Interpolation method ('pchip' or 'cubic').
        error_tol (float): Error tolerance for merging.

    Returns:
        np.ndarray: Final merged control points.

    How it works:
        - Computes control points using both optimized grid merging and dynamic programming merging.
        - Returns the set with fewer control points.
    """
    cp_grid = optimized_merge_control_points_grid(cp, freqs, smooth, method=spline_method, error_tol=error_tol, grid_size=50)
    cp_dp = dp_merge_control_points(freqs, smooth, error_tol=error_tol)
    if len(cp_dp) < len(cp_grid):
        return cp_dp
    else:
        return cp_grid


def post_process_control_points(cp, freqs, smooth, method='pchip', error_tol=1.0):
    """
    Post-process control points by removing or merging points while maintaining the error tolerance.

    Parameters:
        cp (np.ndarray): Array of control points.
        freqs (np.ndarray): Frequency grid.
        smooth (np.ndarray): Smoothed magnitude spectrum.
        method (str): Interpolation method.
        error_tol (float): Maximum allowed reconstruction error after processing.

    Returns:
        np.ndarray: Control points after post processing.
    """
    changed = True
    while changed:
        changed = False
        cp = cp[np.argsort(cp[:, 0])]
        for i in range(1, len(cp) - 1):
            candidate = np.delete(cp, i, axis=0)
            _, recon = reconstruct_hrtf(candidate, method, freqs)
            max_err = np.max(np.abs(smooth - recon))
            if max_err <= error_tol:
                cp = candidate
                changed = True
                break
        if not changed:
            for i in range(1, len(cp) - 1):
                merged_freq = np.sqrt(cp[i, 0] * cp[i+1, 0])
                merged_mag = (cp[i, 1] + cp[i+1, 1]) / 2
                candidate = np.delete(cp, i+1, axis=0)
                candidate[i, :] = [merged_freq, merged_mag]
                candidate = candidate[np.argsort(candidate[:, 0])]
                _, recon = reconstruct_hrtf(candidate, method, freqs)
                max_err = np.max(np.abs(smooth - recon))
                if max_err <= error_tol:
                    cp = candidate
                    changed = True
                    break
    return cp


# -----------------------------
# Adaptive HRTF Compression// Need Fixes, OLD Parameters still in here even tho not used anymore
# -----------------------------
def adaptive_compress_hrtf(ir, fs, fft_len=1024, init_pts=10, global_thresh=1.0,
                           local_thresh=1.0, max_pts=100, win_size=11, err_thresh=1.0,
                           alt_win=7, noise=1e-1, spline_method='pchip',
                           n_bands=24, seg_thresh=0.9, err_bound=0.99):
    """
    Compress a single HRTF measurement using an adaptive control point selection scheme.

    Parameters:
        ir (np.ndarray): Impulse response for one measurement.
        fs (float): Sampling rate.
        fft_len (int): FFT length to use.
        init_pts (int): Minimum number of initial control points.
        global_thresh (float): Global error threshold for pruning.
        local_thresh (float): Local error threshold (not directly used here).
        max_pts (int): Maximum allowed control points.
        win_size (int): Window size for Wiener filtering.
        err_thresh (float): Error threshold for adaptive filtering.
        alt_win (int): Alternative window size for adaptive filtering.
        noise (float): Noise level estimate.
        spline_method (str): Interpolation method ('pchip' or 'cubic').
        n_bands (int): Number of ERB bands for spectral analysis.
        seg_thresh (float): Segment error threshold for refinement.
        err_bound (float): Error bound for iterative control point addition.

    Returns:
        dict: Dictionary containing:
            - frequencies: Frequency grid used.
            - raw_mag: Raw magnitude spectrum (in dB).
            - smoothed_mag: Smoothed magnitude after filtering.
            - control_points: Final control points [frequency, magnitude].
            - reconstructed: Reconstructed magnitude spectrum.
            - spectral_distortion: Overall spectral distortion (RMSE).
            - num_control_points: Final number of control points.
            - points_added: Number of points added compared to initial selection.
            - points_removed: Number of points removed compared to initial selection.

    How it works:
        1. Computes the frequency response from the impulse response.
        2. Applies an adaptive Wiener filter to smooth the magnitude response.
        3. Identifies initial control points via second derivative analysis.
        4. If not enough initial points are found, extra points are added uniformly.
        5. Endpoints are enforced and then iterative refinement, pruning, and merging is performed.
        6. Finally, the reconstruction error is checked and additional control points are added if necessary.
    """
    # Compute the frequency response and convert to dB scale
    freqs, mag = compute_frequency_response(ir, fs, fft_len)
    # Apply adaptive Wiener filtering to smooth the magnitude response
    smooth = apply_filter(mag, win_size, err_thresh, alt_win, noise)
    raw = mag.copy()
    # Identify initial control point indices using second derivative analysis
    cp_idx = find_control_points_via_second_derivative(freqs, smooth, threshold=0.1)
    cp_freqs = freqs[cp_idx]
    cp_mags = smooth[cp_idx]
    # If not enough control points, add extra uniformly spaced points
    if len(cp_freqs) < init_pts:
        extra_idx = np.linspace(0, len(freqs)-1, init_pts, dtype=int)
        cp_freqs = np.unique(np.concatenate((cp_freqs, freqs[extra_idx])))
        cp_freqs = np.sort(cp_freqs)
        cp_mags = np.interp(cp_freqs, freqs, smooth)
    # Ensure endpoints match exactly
    cp_freqs[0], cp_mags[0] = freqs[0], smooth[0]
    cp_freqs[-1], cp_mags[-1] = freqs[-1], smooth[-1]
    cp = np.column_stack((cp_freqs, cp_mags))
    
    # Record the initial number of control points
    initial_cp_count = len(cp)
    
    # Iteratively refine and prune control points using several methods
    cp = refine_cp_by_segment(cp, freqs, smooth, spline_method, seg_thresh, min_distance=5)
    cp = enforce_endpoints(cp, freqs, smooth)
    cp = enhanced_prune_control_points(cp, freqs, smooth, method=spline_method, threshold=global_thresh, dense_factor=12)
    cp = enforce_endpoints(cp, freqs, smooth)
    cp = erb_aware_pruning(cp, freqs, smooth, method=spline_method, global_thresh=global_thresh)
    cp = enforce_endpoints(cp, freqs, smooth)
    cp = enhanced_prune_control_points(cp, freqs, smooth, method=spline_method, threshold=global_thresh*0.95, dense_factor=12)
    cp = enforce_endpoints(cp, freqs, smooth)
    
    # Reconstruct the magnitude response from the refined control points
    _, recon = reconstruct_hrtf(cp, spline_method, freqs)
    max_abs_error = np.max(np.abs(smooth - recon))
    # If error is too high, add additional control points iteratively
    while max_abs_error > 1.0:
        error = smooth - recon
        idx = np.argmax(np.abs(error))
        if np.abs(error[idx]) > 1.0:
            new_freq = freqs[idx]
            new_mag = smooth[idx]
            if not np.any(np.isclose(cp[:, 0], new_freq, atol=1e-5)):
                cp = np.vstack([cp, [new_freq, new_mag]])
                cp = cp[np.argsort(cp[:, 0])]
                _, recon = reconstruct_hrtf(cp, spline_method, freqs)
                max_abs_error = np.max(np.abs(smooth - recon))
            else:
                break
        else:
            break

    cp = post_process_control_points(cp, freqs, smooth, method=spline_method, error_tol=1.0)
    cp = combined_merge_control_points(cp, freqs, smooth, spline_method, error_tol=1.0)
    
    final_cp_count = len(cp)
    points_added = max(final_cp_count - initial_cp_count, 0)
    points_removed = max(initial_cp_count - final_cp_count, 0)
    
    sd = spectral_distortion(smooth, recon)
    return {
        'frequencies': freqs,
        'raw_mag': raw,
        'smoothed_mag': smooth,
        'control_points': cp,
        'reconstructed': recon,
        'spectral_distortion': sd,
        'num_control_points': final_cp_count,
        'points_added': points_added,
        'points_removed': points_removed
    }


# -----------------------------
# SOFA File Handling
# -----------------------------
def compress_all_hrtfs(sofa_path, adaptive=True, init_pts=10, global_thresh=1.0,
                       max_pts=100, win_size=11, err_thresh=1.0, alt_win=7, noise=1e-1,
                       local_thresh=1.0, spline_method='pchip', fft_len=1024,
                       n_bands=24, max_meas=5, seg_thresh=0.9, err_bound=0.99):
    """
    Process all HRTF measurements from a SOFA file and compress them using adaptive compression.

    Parameters:
        sofa_path (str): Path to the SOFA file.
        adaptive (bool): Whether to use adaptive compression.
        init_pts (int): Initial control point count.
        global_thresh (float): Global error threshold.
        max_pts (int): Maximum allowed control points.
        win_size (int): Window size for Wiener filtering.
        err_thresh (float): Error threshold for filtering.
        alt_win (int): Alternative window size for filtering.
        noise (float): Noise level estimate.
        local_thresh (float): Local error threshold (unused directly here).
        spline_method (str): Interpolation method.
        fft_len (int): FFT length.
        n_bands (int): Number of ERB bands.
        max_meas (int): Maximum number of measurements to process.
        seg_thresh (float): Segment threshold for control point refinement.
        err_bound (float): Error bound for control point addition.

    Returns:
        dict: Contains frequencies used, left and right compression results, and source positions.
    
    How it works:
        - Opens the SOFA file and reads HRTF measurements.
        - For each measurement (up to max_meas), compresses the left and right ear impulse responses.
        - Periodically prints progress.
    """
    try:
        hrtf_db = sofa.Database.open(sofa_path)
    except Exception as e:
        print(f"Error opening SOFA file: {e}")
        sys.exit(1)
    fs = hrtf_db.Data.SamplingRate.get_values()
    total = hrtf_db.Source.Position.get_values().shape[0]
    if max_meas is not None and max_meas < total:
        total = max_meas
    # Use one channel to get the frequency grid for plotting
    left_sample = hrtf_db.Data.IR.get_values(indices={"M": 0, "R": 0, "E": 0})
    freqs, _ = compute_frequency_response(left_sample, fs, fft_len)
    left_results, right_results = [], []
    for m in range(total):
        # Retrieve left and right impulse responses for measurement m
        left_ir = hrtf_db.Data.IR.get_values(indices={"M": m, "R": 0, "E": 0})
        right_ir = hrtf_db.Data.IR.get_values(indices={"M": m, "R": 1, "E": 0})
        if adaptive:
            left_res = adaptive_compress_hrtf(left_ir, fs, fft_len, init_pts, global_thresh,
                                              local_thresh, max_pts, win_size, err_thresh, alt_win, noise,
                                              spline_method, n_bands, seg_thresh, err_bound)
            right_res = adaptive_compress_hrtf(right_ir, fs, fft_len, init_pts, global_thresh,
                                               local_thresh, max_pts, win_size, err_thresh, alt_win, noise,
                                               spline_method, n_bands, seg_thresh, err_bound)
        else:
            left_res = None
            right_res = None
        left_results.append(left_res)
        right_results.append(right_res)
        if (m+1) % 50 == 0 or m == total - 1:
            print(f"Processed {m+1}/{total} measurements.")
    positions = hrtf_db.Source.Position.get_values()
    hrtf_db.close()
    return {
        'frequencies': freqs,
        'left_results': left_results,
        'right_results': right_results,
        'positions': positions
    }


def save_all_control_points(results, filename):
    """
    Save all left ear control points from the measurements to a CSV file.

    Each row contains:
       Measurement, Azimuth, Elevation, Frequency, Magnitude

    Parameters:
        results (dict): Dictionary with the compression results (must include 'left_results' and optionally 'positions').
        filename (str): Name of the CSV file to write.

    How it works:
        - Iterates over left ear results.
        - If position information is available, includes azimuth and elevation.
        - Writes each control point as a row in the CSV.
    """
    left_results = results.get('left_results', [])
    positions = results.get('positions', None)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Measurement", "Azimuth", "Elevation", "Frequency", "Magnitude"])
        for i, res in enumerate(left_results):
            if positions is not None and positions.shape[0] > i:
                az = positions[i, 0]
                el = positions[i, 1]
            else:
                az = ""
                el = ""
            cp = res.get('control_points')
            if cp is not None:
                for row in cp:
                    writer.writerow([i, az, el, row[0], row[1]])
    print(f"All control points saved to {filename}")


def save_compression_rates(results, filename):
    """
    Save the compression rates (in the byte domain) to a CSV file.

    The original data size is computed from the frequency response used:
        used_bins x 2 channels x 4 bytes per bin.

    Each row contains:
        Measurement, Left CP, Right CP, Combined CR, Left CR, Right CR

    Parameters:
        results (dict): Dictionary with compression results (must include 'frequencies', 'left_results', 'right_results').
        filename (str): CSV filename to save the compression rates.
    """
    used_bins = len(results['frequencies'])
    original_bytes = used_bins * 2 * 4  # 2 channels, 4 bytes each
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Measurement", "Left CP", "Right CP", "Combined CR", "Left CR", "Right CR"])
        for i, (left_res, right_res) in enumerate(zip(results['left_results'], results['right_results'])):
            left_cp = left_res.get('num_control_points', None)
            right_cp = right_res.get('num_control_points', None)
            if left_cp is None or right_cp is None or left_cp == 0 or right_cp == 0:
                combined_cr = None
                left_cr = None
                right_cr = None
            else:
                left_bytes = left_cp * 2 * 4  # each control point uses 8 bytes (2 channels x 4 bytes)
                right_bytes = right_cp * 2 * 4
                total_compressed = left_bytes + right_bytes
                combined_cr = original_bytes / total_compressed
                left_cr = original_bytes / left_bytes
                right_cr = original_bytes / right_bytes
            writer.writerow([i, left_cp, right_cp, combined_cr, left_cr, right_cr])
    print(f"Compression rates saved to {filename}")


# -----------------------------
# Plotting & Visualization Functions
# -----------------------------
def plot_reconstruction(result, ax, title_suffix="", az=None, el=None):
    """
    Plot the original and reconstructed magnitude responses along with control points.

    Parameters:
        result (dict): Dictionary with keys 'frequencies', 'raw_mag', 'reconstructed', 'control_points', and optionally 'spectral_distortion' and 'num_control_points'.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        title_suffix (str): Additional string to add to the title.
        az (float, optional): Azimuth value (for display in title).
        el (float, optional): Elevation value (for display in title).
    """
    ax.clear()
    freqs = result['frequencies']
    raw = result['raw_mag']
    recon = result['reconstructed']
    cp = result.get('control_points')
    sd = result.get('spectral_distortion')
    num_cp = result.get('num_control_points', 'N/A')
    ax.semilogx(freqs, raw, 'b-', alpha=0.7, label="Original Data")
    label = f"Reconstructed (SD: {sd:.2f} dB, CP: {num_cp})" if sd is not None else "Reconstructed"
    ax.semilogx(freqs, recon, 'k--', label=label)
    if cp is not None:
        ax.plot(cp[:, 0], cp[:, 1], 'ro', markersize=4, label="Control Points")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    title = "Original vs. Reconstructed " + title_suffix
    if az is not None and el is not None:
        title += f" (Az: {az:.1f}°, Elev: {el:.1f}°)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.set_xlim(20, 22000)
    ax.set_ylim(-45, 10)


def plot_sd_boxplot(measurements, n_bands=24, az=None, el=None):
    """
    Plot a boxplot of spectral distortion per ERB band for a set of measurements.

    Parameters:
        measurements (list): List of measurement dictionaries.
        n_bands (int): Number of ERB bands to compute.
        az (float, optional): Azimuth for plot title.
        el (float, optional): Elevation for plot title.
    """
    erb_bounds, _ = get_erb_bands(20, 20000, n_bands)
    band_errs = [[] for _ in range(len(erb_bounds)-1)]
    for res in measurements:
        freqs = res['frequencies']
        smooth = res['smoothed_mag']
        recon = res['reconstructed']
        sd_bands = calculate_band_sd(freqs, smooth, recon, erb_bounds)
        for i, val in enumerate(sd_bands):
            band_errs[i].append(val)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(band_errs, showfliers=True)
    ax.set_xlabel("ERB Band", fontsize=30)
    ax.set_ylabel("Spectral Distortion (dB)", fontsize=30)
    title = "Spectral Distortion per ERB Band"
    if az is not None and el is not None:
        title += f" (Az: {az:.1f}°, Elev: {el:.1f}°)"
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', which='major', labelsize=25)
    n = len(erb_bounds) - 1
    tick_positions = list(range(1, n+1))
    tick_labels = [str(i) for i in range(1, n+1)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_error_details(result, title_suffix="", az=None, el=None):
    """
    Plot the original and reconstructed responses along with the error curve.

    Parameters:
        result (dict): Dictionary containing frequency response, original data, and reconstruction.
        title_suffix (str): Suffix for the plot title.
        az (float, optional): Azimuth value.
        el (float, optional): Elevation value.
    """
    freqs = result['frequencies']
    raw = result['raw_mag']
    recon = result['reconstructed']
    err = recon - raw
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    title1 = title_suffix
    title2 = "Error"
    if az is not None and el is not None:
        extra = f" (Az: {az:.1f}°, Elev: {el:.1f}°)"
        title1 += extra
        title2 += extra
    ax1.semilogx(freqs, raw, 'b-', alpha=0.7, label="Original Data")
    ax1.semilogx(freqs, recon, 'r--', alpha=0.7, label="Reconstructed")
    ax1.set_ylabel("Magnitude (dB)", fontsize=35)
    ax1.legend()
    ax1.tick_params(axis='both', which='major', labelsize=35)
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.set_ylim(-45, 10)
    ax2.semilogx(freqs, err, 'k-', label="Error (Reconstructed - Original)")
    ax2.tick_params(axis='both', which='major', labelsize=35)
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Frequency (Hz)", fontsize=35)
    ax2.set_ylabel("Error (dB)", fontsize=35)
    ax2.set_ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.show()


def plot_filtering_error(result, title_suffix="", az=None, el=None):
    """
    Plot the filtering error (difference between smoothed and raw magnitudes) and display RMS error.

    Parameters:
        result (dict): Dictionary containing frequency response and filtered data.
        title_suffix (str): Suffix for the title.
        az (float, optional): Azimuth for title.
        el (float, optional): Elevation for title.
    """
    freqs = result['frequencies']
    raw = result['raw_mag']
    smooth = result['smoothed_mag']
    err = smooth - raw
    rms = np.sqrt(np.mean(err**2))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(freqs, err, 'm-', label=f"Error (RMS: {rms:.2f} dB)")
    title = title_suffix
    if az is not None and el is not None:
        title += f" (Az: {az:.1f}°, Elev: {el:.1f}°)"
    ax.set_xlabel("Frequency (Hz)", fontsize=35)
    ax.set_ylabel("Error (dB)", fontsize=35)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.legend()
    ax.set_ylim(-0.75, 0.75)
    plt.show()


def plot_filtering_stats(result, title_suffix="", az=None, el=None):
    """
    Plot a bar chart showing maximum absolute error and standard deviation of filtering error.

    Parameters:
        result (dict): Dictionary containing raw and smoothed magnitudes.
        title_suffix (str): Suffix for the title.
        az (float, optional): Azimuth.
        el (float, optional): Elevation.
    """
    raw = result['raw_mag']
    smooth = result['smoothed_mag']
    err = smooth - raw
    abs_err = np.max(np.abs(err))  # Maximum absolute error
    std_err = np.std(err)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['Absolute Error', 'Std Dev'], [abs_err, std_err])
    title = title_suffix
    if az is not None and el is not None:
        title += f" (Az: {az:.1f}°, Elev: {el:.1f}°)"
    ax.set_ylabel("Error (dB)", fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=45)
    plt.ylim(0, 0.5)
    print(f"ERROR AND DEVIA: {abs_err:.2f}, {std_err:.2f}")
    plt.show()


def plot_average_filtering_error(measurements):
    """
    Plot the average filtering error across all measurements with a 95% confidence interval.
    
    Parameters:
        measurements (list): List of measurement dictionaries.
    
    How it works:
        - Computes the error (smoothed minus raw) for each measurement.
        - Computes the mean error and the standard error across measurements at each frequency.
        - Calculates the 95% confidence interval (using a multiplier of 1.96).
        - Plots the average error and shades the area between the lower and upper CI bounds.
    """
    if len(measurements) == 0:
        print("No measurements provided for average filtering error plot.")
        return

    # Assume that all measurements share the same frequency grid
    freq = measurements[0]['frequencies']
    
    # Stack the error arrays from each measurement
    errors = np.array([res['smoothed_mag'] - res['raw_mag'] for res in measurements])
    
    # Calculate the mean error and standard error at each frequency
    avg_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    n = len(measurements)
    se = std_error / np.sqrt(n)
    
    # Define the multiplier for a 95% confidence interval (approximately 1.96 for large n)
    ci_multiplier = 1.96
    lower_bound = avg_error - ci_multiplier * se
    upper_bound = avg_error + ci_multiplier * se

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(freq, avg_error, 'm-', label="Average Filtering Error")
    ax.fill_between(freq, lower_bound, upper_bound, color='m', alpha=0.3, label="95% CI")
    ax.set_xlabel("Frequency (Hz)", fontsize=30)
    ax.set_ylabel("Error (dB)", fontsize=30)
    ax.set_title("Average Filtering Error Across All Measurements", fontsize=35)    
    ax.set_ylim(-0.10, 0.05)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.show()



# -----------------------------
# Main Interactive Interface
# -----------------------------
def main():
    """
    Main entry point for interactive HRTF compression and visualization.

    Process:
        - Checks for the existence of a SOFA file.
        - Sets parameters for compression and filtering.
        - Calls compress_all_hrtfs to process the SOFA file.
        - Computes average compression ratio and filtering error statistics.
        - Saves control points and compression rates to CSV files.
        - Sets up an interactive Matplotlib figure with sliders and text boxes for exploring the results.
    """
    sofa_path = "src/sofa/minPhase_NoITD.sofa"
    if not os.path.exists(sofa_path):
        print(f"ERROR: SOFA file not found at '{sofa_path}'")
        sys.exit(1)
    
    # Parameters for compression and filtering
    FFT_LEN = 1024
    SG_WINDOW = 3
    ERROR_THRESHOLD = 0.95
    ALT_WINDOW = 3
    NOISE_EST = 0.1
    NUM_BANDS = 24
    INIT_POINTS = 10
    THRESHOLD_SD = 0.95
    LOCAL_THRESHOLD = 0.95
    MAX_POINTS = 100
    SPLINE_KIND = 'pchip'
    SEGMENT_THRESHOLD = 0.9
    ERROR_BOUND = 0.95
    adaptive = True
    
    # Compress all HRTF measurements from the SOFA file
    compressed_db = compress_all_hrtfs(
        sofa_path,
        adaptive=adaptive,
        init_pts=INIT_POINTS,
        global_thresh=THRESHOLD_SD,
        max_pts=MAX_POINTS,
        win_size=SG_WINDOW,
        err_thresh=ERROR_THRESHOLD,
        alt_win=ALT_WINDOW,
        noise=NOISE_EST,
        local_thresh=LOCAL_THRESHOLD,
        spline_method=SPLINE_KIND,
        fft_len=FFT_LEN,
        n_bands=NUM_BANDS,
        max_meas=None,
        seg_thresh=SEGMENT_THRESHOLD,
        err_bound=ERROR_BOUND
    )
    print("Compression complete.")
    
    left_results = compressed_db['left_results']
    right_results = compressed_db['right_results']
    positions = compressed_db.get('positions', None)  # Retrieve measurement positions from the SOFA file
    num_meas = len(left_results)
    
    total_cp = 0
    count = 0
    for res in left_results + right_results:
        if res is not None and 'num_control_points' in res:
            total_cp += res['num_control_points']
            count += 1
    if count > 0:
        avg_cp = total_cp / count
        print(f"Average number of control points (both ears): {avg_cp:.2f}")
    else:
        print("No valid measurements found.")

    
    # Compute combined compression ratio for each measurement
    freq_length = len(left_results[0]['frequencies'])
    original_bytes = freq_length * 2 * 4  # 2 channels, 4 bytes each
    combined_cr_list = []
    for left_res, right_res in zip(left_results, right_results):
        left_cp = left_res.get('num_control_points', 0)
        right_cp = right_res.get('num_control_points', 0)
        if left_cp > 0 and right_cp > 0:
            total_bytes = (left_cp + right_cp) * 8  # each control point uses 8 bytes
            combined_cr = original_bytes / total_bytes
            combined_cr_list.append(combined_cr)
    if combined_cr_list:
        avg_combined_cr = np.mean(combined_cr_list)
        print(f"Average Combined Compression Ratio: {avg_combined_cr:.2f}")
    else:
        print("No valid measurements for combined CR.")
    
    # Compute and print average filtering error statistics across all measurements
    total_abs_error = 0.0
    total_std_error = 0.0
    for res in left_results:
        err = res['smoothed_mag'] - res['raw_mag']
        total_abs_error += np.max(np.abs(err))
        total_std_error += np.std(err)
    avg_abs_error = total_abs_error / num_meas
    avg_std_error = total_std_error / num_meas
    print(f"In average, the measurements have {avg_abs_error:.2f} dB absolute error and {avg_std_error:.2f} dB standard deviation after filtering.")
    
    # Save compression rates and control points to CSV files
    save_compression_rates(compressed_db, "compression_rates.csv")
    save_all_control_points(compressed_db, "all_control_points.csv")
    
    # Set up interactive plotting interface with Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.1, bottom=0.42)
    
    def update_slider(val):
        """
        Update the reconstruction plot when the measurement slider is moved.
        """
        idx = int(meas_slider.val)
        if positions is not None and positions.shape[0] > idx:
            az = positions[idx, 0]
            el = positions[idx, 1]
        else:
            az = el = None
        plot_reconstruction(left_results[idx], ax, title_suffix=f"(Left Ear, Measurement {idx})", az=az, el=el)
        fig.canvas.draw_idle()
    
    # Create a slider for selecting measurement index
    slider_ax = plt.axes([0.15, 0.36, 0.7, 0.03])
    meas_slider = Slider(slider_ax, 'Measurement', 0, num_meas - 1, valinit=0, valstep=1)
    meas_slider.on_changed(update_slider)
    
    # Create a text box for triggering a spectral distortion boxplot
    textbox_sd_ax = plt.axes([0.15, 0.02, 0.7, 0.05])
    sd_box = TextBox(textbox_sd_ax, 'SD Boxplot', initial="all")
    
    def submit_boxplot(text):
        """
        Submit callback for the SD Boxplot text box.
        If 'all' is entered, plot for all measurements, otherwise for specified indices.
        """
        text = text.strip().lower()
        if text == "all":
            sel = left_results
            plot_sd_boxplot(sel, n_bands=NUM_BANDS)
        else:
            try:
                indices = [int(x.strip()) for x in text.split(",") if x.strip()]
                sel = [left_results[i] for i in indices if 0 <= i < num_meas]
                if len(sel) == 1 and positions is not None and positions.shape[0] > indices[0]:
                    az = positions[indices[0], 0]
                    el = positions[indices[0], 1]
                    plot_sd_boxplot(sel, n_bands=NUM_BANDS, az=az, el=el)
                else:
                    plot_sd_boxplot(sel, n_bands=NUM_BANDS)
            except Exception:
                print("Invalid input; using all measurements.")
                plot_sd_boxplot(left_results, n_bands=NUM_BANDS)
    
    sd_box.on_submit(submit_boxplot)
    
    # Create a text box for error details plotting
    textbox_err_ax = plt.axes([0.15, 0.08, 0.7, 0.05])
    err_box = TextBox(textbox_err_ax, 'Error Details', initial="0")
    
    def submit_error(text):
        """
        Submit callback for the error details text box.
        Plots the error details for a selected measurement index.
        """
        try:
            idx = int(text.strip())
            if 0 <= idx < num_meas:
                if positions is not None and positions.shape[0] > idx:
                    az = positions[idx, 0]
                    el = positions[idx, 1]
                else:
                    az = el = None
                plot_error_details(left_results[idx], title_suffix=f"(Left Ear, Measurement {idx})", az=az, el=el)
            else:
                print("Index out of range.")
        except Exception as e:
            print("Invalid input for error details:", e)
    
    err_box.on_submit(submit_error)
    
    # Create a text box for filtering error plot
    textbox_filt_ax = plt.axes([0.15, 0.15, 0.7, 0.05])
    filt_box = TextBox(textbox_filt_ax, 'Filtering Error', initial="0")
    
    def submit_filter(text):
        """
        Submit callback for filtering error text box.
        Plots filtering error for a selected measurement.
        """
        try:
            idx = int(text.strip())
            if 0 <= idx < num_meas:
                if positions is not None and positions.shape[0] > idx:
                    az = positions[idx, 0]
                    el = positions[idx, 1]
                else:
                    az = el = None
                plot_filtering_error(left_results[idx], title_suffix=f"(Left Ear, Measurement {idx})", az=az, el=el)
            else:
                print("Index out of range for filtering error.")
        except Exception as e:
            print("Invalid input for filtering error:", e)
    
    filt_box.on_submit(submit_filter)
    
    # Create a text box for filtering stats plot
    textbox_filt_stats_ax = plt.axes([0.15, 0.22, 0.7, 0.05])
    filt_stats_box = TextBox(textbox_filt_stats_ax, 'Filtering Stats', initial="0")
    
    def submit_filter_stats(text):
        """
        Submit callback for filtering stats text box.
        Plots filtering statistics for a selected measurement.
        """
        try:
            idx = int(text.strip())
            if 0 <= idx < num_meas:
                if positions is not None and positions.shape[0] > idx:
                    az = positions[idx, 0]
                    el = positions[idx, 1]
                else:
                    az = el = None
                plot_filtering_stats(left_results[idx], title_suffix=f"(Left Ear, Measurement {idx})", az=az, el=el)
            else:
                print("Index out of range for filtering stats.")
        except Exception as e:
            print("Invalid input for filtering stats:", e)
    
    filt_stats_box.on_submit(submit_filter_stats)
    
    # Create a text box for saving the current plot
    save_ax = plt.axes([0.15, 0.28, 0.7, 0.05])
    save_box = TextBox(save_ax, 'Save Plot', initial="measurement_plot.png")
    
    def submit_save(text):
        """
        Submit callback for the save plot text box.
        Saves the current figure to the specified filename.
        """
        filename = text.strip()
        if filename:
            try:
                fig.savefig(filename, dpi=300)
                print(f"Plot saved as {filename}")
            except Exception as e:
                print("Error saving plot:", e)
    
    save_box.on_submit(submit_save)
    
    # Plot the average filtering error across all measurements
    plot_average_filtering_error(left_results)
    
    print(sofa.__version__)
    
    plt.show()


if __name__ == "__main__":
    main()
