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
# from numba import njit, prange
from matplotlib import rc_context
import pickle
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

# ——————————————————————————————————————————————————————
# New: Auditory-smoothing implementations
# ——————————————————————————————————————————————————————

def erb_smooth(mag, freqs, n_bands=42):
    """
    ERB-band smoothing via Gammatone-like filterbank.
    """
    # compute ERB edges
    erb_min = 24.7 * (4.37 * freqs[0] / 1000 + 1)
    erb_max = 24.7 * (4.37 * freqs[-1] / 1000 + 1)
    erb_edges = np.linspace(erb_min, erb_max, n_bands+1)
    # back to Hz
    hz_edges = ( (erb_edges/24.7 - 1) * 1000 / 4.37 )
    smooth = np.zeros_like(mag)
    for i in range(n_bands):
        lo, hi = hz_edges[i], hz_edges[i+1]
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            # energy sum then back to dB
            e = np.mean(10**(mag[mask]/10))
            smooth[mask] = 10*np.log10(e + 1e-12)
    return smooth

def octave_smooth(mag, freqs, frac_oct=3):
    """
    Fractional-octave smoothing: window spans [f/2^(1/frac_oct), f*2^(1/frac_oct)].
    """
    log_mag = mag.copy()
    smooth = np.zeros_like(log_mag)
    for i,f in enumerate(freqs):
        k = 2**(1/frac_oct)
        lo, hi = f/k, f*k
        mask = (freqs >= lo) & (freqs <= hi)
        smooth[i] = np.mean(log_mag[mask]) if np.any(mask) else log_mag[i]
    return smooth

def cepstral_smooth(mag, lifter_len=20):
    """
    Cepstral liftering: keep lowest lifter_len quefrency bins.
    """
    # work in log-mag
    cep = np.fft.ifft(mag).real
    liftered = np.concatenate([cep[:lifter_len], np.zeros_like(cep[lifter_len:])])
    smooth = np.real(np.fft.fft(liftered))
    return smooth

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




# -----------------------------
# Reconstruction & Error Metrics
# -----------------------------
def apply_filter(mag,
                 freqs=None,
                 method='wiener',
                 win_size=11,
                 err_thresh=1.0,
                 alt_win=7,
                 noise=1e-1,
                 n_bands=42,
                 frac_oct=3,
                 lifter_len=20):
    """
    Dispatch to one of four smoothing methods:
      - 'wiener'   : adaptive Wiener filtering
      - 'erb'      : ERB-band (gammatone-like) smoothing
      - 'octave'   : fractional-octave (1/frac_oct) smoothing
      - 'cepstral' : cepstral liftering
    """
    if method == 'wiener':
        return adaptive_wiener(
            mag,
            base_win=win_size,
            err_thresh=err_thresh,
            alt_win=alt_win,
            noise=noise
        )

    elif method == 'erb':
        if freqs is None:
            raise ValueError("ERB smoothing requires `freqs` array")
        return erb_smooth(
            mag,
            freqs,
            n_bands=n_bands
        )

    elif method == 'octave':
        if freqs is None:
            raise ValueError("Octave smoothing requires `freqs` array")
        return octave_smooth(
            mag,
            freqs,
            frac_oct=frac_oct
        )

    elif method == 'cepstral':
        return cepstral_smooth(
            mag,
            lifter_len=lifter_len
        )

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

# -----------------------------
# Reconstruction & Error Metrics
# -----------------------------
def reconstruct_hrtf(control_pts, method='pchip', grid=None):
    """
    Reconstruct the HRTF magnitude response using control points.

    Parameters:
        control_pts (np.ndarray): Nx2 array [frequency, magnitude].
        method       (str)       : 'cubic' for CubicSpline, else PCHIP.
        grid         (np.ndarray): Frequencies to interpolate on (default 513 samples).

    Returns:
        grid (np.ndarray), recon (np.ndarray)
    """
    # sort control points by frequency
    cp = control_pts[np.argsort(control_pts[:,0])]
    freqs_cp = cp[:,0]
    mags_cp  = cp[:,1]
    # print(f"Reconstructing HRTF with {len(cp)} control points")
    # choose interpolator
    if method == 'cubic':
        interp_fn = CubicSpline(freqs_cp, mags_cp, extrapolate=True)
    else:
        interp_fn = PchipInterpolator(freqs_cp, mags_cp, extrapolate=True)

    # build output grid
    if grid is None:
        grid = np.linspace(freqs_cp[0], freqs_cp[-1], 513)

    # interpolate
    recon = interp_fn(grid)
    return grid, recon


# @njit(parallel=True, cache=True)
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
    for i in range(n):
        diff = orig[i] - recon[i]
        total += diff * diff
    return np.sqrt(total / n)


# @njit(parallel=True, cache=True)
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
    for i in range(n):
        if freqs[i] >= low and freqs[i] <= high:
            diff = abs(orig[i] - recon[i])
            if diff > error_val:
                error_val = diff
    return error_val


# Aliases to use our parallel implementations
spectral_distortion = parallel_spectral_distortion
calc_segment_error = parallel_calc_segment_error


def enforce_endpoints(cp, freqs, raw):
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
    cp[0, 1] = raw[0]
    cp[-1, 1] = raw[-1]
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


def get_erb_bands(freq_min=20, freq_max=20000, n_bands=42):
    erb_min = frequency_to_erb(freq_min)
    erb_max = frequency_to_erb(freq_max)
    erb_bounds = np.linspace(erb_min, erb_max, n_bands + 1)
    # convert back to Hz so that masking in calculate_band_sd uses the same units as freqs[]
    bounds_hz = np.array([erb_to_frequency(e) for e in erb_bounds])
    centers = np.sqrt(bounds_hz[:-1] * bounds_hz[1:])
    return bounds_hz, centers



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


def refine_cp_by_segment(cp, freqs, raw, method='pchip', seg_thresh=0.9, min_distance=5):
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
    cp[0, 1] = raw[0]
    cp[-1, 1] = raw[-1]
    improved = True
    while improved:
        improved = False
        cp = cp[np.argsort(cp[:, 0])]
        _, recon = reconstruct_hrtf(cp, method, freqs)
        # Loop over each adjacent pair of control points
        for i in range(len(cp) - 1):
            low_f, high_f = cp[i, 0], cp[i+1, 0]
            seg_err = calc_segment_error(freqs, raw, recon, low_f, high_f)
            # If segment error exceeds threshold, insert a new control point in that segment
            if seg_err > seg_thresh * 1.2:
                mask = (freqs >= low_f) & (freqs <= high_f)
                if not np.any(mask):
                    continue
                seg_freqs = freqs[mask]
                seg_errs = np.abs(raw[mask] - recon[mask])
                idx = np.argmax(seg_errs)
                new_f = seg_freqs[idx]
                new_m = raw[mask][idx]
                # Only add if the new frequency is sufficiently separated from existing control points
                if all(np.abs(new_f - existing_cp[0]) > min_distance for existing_cp in cp):
                    cp = np.vstack([cp, [new_f, new_m]])
                    improved = True
                    break
        if improved:
            cp = enforce_endpoints(cp, freqs, raw)
    return cp


def enhanced_prune_control_points(cp, freqs, raw, method='pchip', threshold=1.0, dense_factor=12):
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
    baseline = np.interp(dense_grid, freqs, raw)
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


def erb_aware_pruning(cp, freqs, raw, method='pchip', global_thresh=1.0):
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
    erb_bounds, _ = get_erb_bands(20, 20000, 42)
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
            band_errors = calculate_band_sd(freqs, raw, recon, erb_bounds)
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


# def post_process_control_points(cp, freqs, raw, method='pchip', error_tol=1.0):
#     """
#     Post-process control points by removing or merging points while maintaining the error tolerance.

#     Parameters:
#         cp (np.ndarray): Array of control points.
#         freqs (np.ndarray): Frequency grid.
#         smooth (np.ndarray): Smoothed magnitude spectrum.
#         method (str): Interpolation method.
#         error_tol (float): Maximum allowed reconstruction error after processing.

#     Returns:
#         np.ndarray: Control points after post processing.

#     How it works:
#         - First attempts to remove a control point if doing so keeps the max error within tolerance.
#         - If no removal is possible, attempts to merge adjacent control points.
#         - Iterates until no further changes are possible.
#     """
#     changed = True
#     while changed:
#         changed = False
#         cp = cp[np.argsort(cp[:, 0])]
#         for i in range(1, len(cp) - 1):
#             candidate = np.delete(cp, i, axis=0)
#             _, recon = reconstruct_hrtf(candidate, method, freqs)
#             max_err = np.max(np.abs(raw - recon))
#             if max_err <= error_tol:
#                 cp = candidate
#                 changed = True
#                 break
#         if not changed:
#             for i in range(1, len(cp) - 1):
#                 merged_freq = np.sqrt(cp[i, 0] * cp[i+1, 0])
#                 merged_mag = (cp[i, 1] + cp[i+1, 1]) / 2
#                 candidate = np.delete(cp, i+1, axis=0)
#                 candidate[i, :] = [merged_freq, merged_mag]
#                 candidate = candidate[np.argsort(candidate[:, 0])]
#                 _, recon = reconstruct_hrtf(candidate, method, freqs)
#                 max_err = np.max(np.abs(raw - recon))
#                 if max_err <= error_tol:
#                     cp = candidate
#                     changed = True
#                     break
#     return cp


def optimized_prune_control_points_grid(cp, freqs, raw,
                                        method='pchip',
                                        error_tol=1.0,
                                        grid_size=50,
                                        max_iters=5000):
    """
    Grid-based pruning of control points: tries to remove points while keeping
    max reconstruction error ≤ error_tol. Does NOT re-add points if error exceeds tol.
    
    Parameters:
        cp         (ndarray): initial control points, shape (M,2)
        freqs      (ndarray): full frequency grid
        raw        (ndarray): original magnitude spectrum
        method     (str)    : interpolation method, 'pchip' or 'cubic'
        error_tol  (float)  : max allowed abs error
        grid_size  (int)    : number of candidates between each pair
        max_iters  (int)    : safeguard on total iterations
    
    Returns:
        merged_cp  (ndarray): pruned control points
    """
    # 1) sort and initialize
    cp = cp[np.argsort(cp[:,0])]
    new_cp = [cp[0].tolist()]
    i = 0
    iters = 0

    # 2) try removing points via grid-based search
    while i < len(cp)-1 and iters < max_iters:
        iters += 1
        p1, p2 = cp[i], cp[i+1]
        candidates = np.linspace(p1[0], p2[0], grid_size)
        candidate_errors = []

        for cf in candidates:
            mag_cf = float(np.interp(cf, freqs, raw))
            # build candidate set: new_cp + this cf + remaining originals
            cand_set = np.vstack([ new_cp,
                                   [cf, mag_cf],
                                   cp[i+2:] ])
            cand_set = cand_set[np.argsort(cand_set[:,0])]
            # compute error
            _, recon = reconstruct_hrtf(cand_set, method, freqs)
            e = np.max(np.abs(raw - recon))
            candidate_errors.append(e)

        candidate_errors = np.array(candidate_errors)
        min_err = candidate_errors.min()
        best_cf = candidates[candidate_errors.argmin()]
        best_mag = float(np.interp(best_cf, freqs, raw))

        if min_err <= error_tol:
            # accept the best candidate instead of cp[i+1]
            new_cp.append([best_cf, best_mag])
            i += 2
        else:
            # keep the original cp[i+1]
            new_cp.append(cp[i+1].tolist())
            i += 1

    # 3) finalize: ensure last endpoint
    merged_cp = np.array(new_cp)
    if merged_cp[-1,0] != cp[-1,0]:
        merged_cp = np.vstack([merged_cp, cp[-1]])

    return merged_cp




def optimized_merge_control_points_grid(cp, freqs, raw,
                                        method='pchip',
                                        error_tol=1.0,
                                        grid_size=50):
    """
    Merge control points by optimizing the position of new control points over a grid,
    then enforce a maximum absolute error tolerance by inserting worst-offending bins.
    Includes safeguards against infinite loops and ensures at least two control points.
    """
    # 1) sort and initialize
    cp = cp[np.argsort(cp[:, 0])]
    new_cp = [cp[0]]
    i = 0
    err_val = float('inf')
    # max_iters = len(cp) * grid_size * 2  # safeguard against infinite loops
    max_iters = min(len(cp) * grid_size, 5000)

    iters = 0

    # 2) grid-based merge loop: step through adjacent pairs while error > tol
    while i < len(cp) - 1 and err_val > error_tol and iters < max_iters:
        iters += 1
        p1, p2 = cp[i], cp[i+1]
        candidates = np.linspace(p1[0], p2[0], grid_size)
        candidate_errors = []

        for cf in candidates:
            cand_mag = np.interp(cf, freqs, raw)
            candidate_set = np.vstack([new_cp,
                                        [cf, cand_mag],
                                        cp[i+2:]])
            candidate_set = candidate_set[np.argsort(candidate_set[:, 0])]

            # invalid candidate sets get infinite error
            if np.any(np.diff(candidate_set[:, 0]) <= 0) or candidate_set.shape[0] < 2:
                e = float('inf')
            else:
                _, recon = reconstruct_hrtf(candidate_set, method, freqs)
                e = np.max(np.abs(raw - recon))
            candidate_errors.append(e)

        candidate_errors = np.array(candidate_errors)
        min_err = candidate_errors.min()
        best_cf = candidates[candidate_errors.argmin()]

        if min_err > error_tol:
            # back-track but don't go below zero
            i = max(0, i - 1)
            new_cp.pop()
            new_cp.append(cp[i])
            err_val = min_err
            continue

        # insert the best new point or the next original point
        if min_err <= error_tol:
            new_cp.append([best_cf, float(np.interp(best_cf, freqs, raw))])
            i += 2
        else:
            new_cp.append(cp[i+1])
            i += 1

        err_val = min_err

    # 3) finalize merged control points and ensure endpoints
    merged_cp = np.array(new_cp)
    # ensure at least two control points
    if merged_cp.shape[0] < 2:
        merged_cp = cp.copy()
    else:
        # enforce original endpoints
        if merged_cp[0, 0] != cp[0, 0]:
            merged_cp = np.vstack([cp[0], merged_cp])
        if merged_cp[-1, 0] != cp[-1, 0]:
            merged_cp = np.vstack([merged_cp, cp[-1]])

    # 4) ensure reconstruction error ≤ tolerance by inserting worst bins
    _, recon = reconstruct_hrtf(merged_cp, method, freqs)
    max_err = np.max(np.abs(raw - recon))

    while max_err > error_tol:
        idx = np.argmax(np.abs(raw - recon))
        new_pt = [freqs[idx], raw[idx]]
        merged_cp = np.vstack([merged_cp, new_pt])
        merged_cp = merged_cp[np.argsort(merged_cp[:, 0])]
        _, recon = reconstruct_hrtf(merged_cp, method, freqs)
        max_err = np.max(np.abs(raw - recon))

    return merged_cp





def combined_merge_control_points(cp, freqs, raw, spline_method='pchip', error_tol=1.0):
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
    cp_grid = optimized_merge_control_points_grid(cp, freqs, raw, method=spline_method, error_tol=error_tol, grid_size=50)
    # cp_dp = dp_merge_control_points(freqs, raw, error_tol=error_tol)
    # print(f"Grid-based control points: {len(cp_grid)}, DP-based control points: {len(cp_dp)}")
    # if len(cp_dp) < len(cp_grid):
    #     return cp_dp
    # else:
    return cp_grid


def post_process_control_points(cp, freqs, raw, method='pchip', error_tol=1.0):
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
            max_err = np.max(np.abs(raw - recon))
            if max_err <= error_tol:
                cp = candidate
                # print(f" max_err <= error_tol Removed control point at index {i}, new count: {len(cp)}")
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
                max_err = np.max(np.abs(raw - recon))
                if max_err <= error_tol:
                    cp = candidate
                    changed = True
                    break
    # print(f"FINAL: Post-processing control points, current count: {len(cp)}")
    return cp


# -----------------------------
# Adaptive HRTF Compression// Need Fixes, OLD Parameters still in here even tho not used anymore
# -----------------------------
# Add these new functions to your Code 1 imports and function definitions section
# (Right after the existing control point processing functions)

def dp_optimize_control_points_fixed(freqs, raw, error_tol=1.0, method='pchip'):
    """
    Use dynamic programming to find minimum control points with correct interpolation
    """
    N = len(freqs)
    dp = [float('inf')] * N
    parent = [-1] * N
    dp[N-1] = 0
    
    # Pre-compute valid segments to avoid redundant calculations
    valid_segments = {}
    
    for i in range(N-2, -1, -1):
        for j in range(i+1, N):
            # Create segment from i to j
            if j - i == 1:
                # Adjacent points - always acceptable with 0 error
                segment_error = 0.0
            else:
                # Check if we already computed this segment
                seg_key = (i, j)
                if seg_key not in valid_segments:
                    # Build control points for this segment
                    segment_cp = np.column_stack((freqs[i:j+1:max(1, (j-i)//10)], 
                                                raw[i:j+1:max(1, (j-i)//10)]))
                    # Ensure endpoints are included
                    if segment_cp[0, 0] != freqs[i]:
                        segment_cp = np.vstack([[freqs[i], raw[i]], segment_cp])
                    if segment_cp[-1, 0] != freqs[j]:
                        segment_cp = np.vstack([segment_cp, [freqs[j], raw[j]]])
                    
                    # Remove duplicates and sort
                    segment_cp = np.unique(segment_cp, axis=0)
                    segment_cp = segment_cp[np.argsort(segment_cp[:, 0])]
                    
                    if len(segment_cp) < 2:
                        segment_error = float('inf')
                    else:
                        try:
                            # Use the same interpolation method as final reconstruction
                            if method == 'cubic':
                                interp = CubicSpline(segment_cp[:, 0], segment_cp[:, 1], extrapolate=True)
                            else:
                                interp = PchipInterpolator(segment_cp[:, 0], segment_cp[:, 1], extrapolate=True)
                            
                            seg_recon = interp(freqs[i:j+1])
                            segment_error = np.max(np.abs(raw[i:j+1] - seg_recon))
                        except:
                            segment_error = float('inf')
                    
                    valid_segments[seg_key] = segment_error
                else:
                    segment_error = valid_segments[seg_key]
            
            # Update DP if this segment is valid and improves the solution
            if segment_error <= error_tol and 1 + dp[j] < dp[i]:
                dp[i] = 1 + dp[j]
                parent[i] = j
    
    # Reconstruct optimal path
    indices = []
    i = 0
    while i != -1 and i < N:
        indices.append(i)
        i = parent[i]
        if i == -1:
            break
    
    if len(indices) < 2:
        # Fallback: return endpoints
        return np.column_stack(([freqs[0], freqs[-1]], [raw[0], raw[-1]]))
    
    return np.column_stack((freqs[indices], raw[indices]))



def aggressive_prune_control_points_enhanced(cp, freqs, raw, method='pchip', error_tol=1.0, max_iterations=20):
    """
    Enhanced aggressive pruning with better merging strategies
    """
    cp = cp[np.argsort(cp[:, 0])]
    
    # Remove duplicates
    cp = np.unique(cp, axis=0)
    
    if len(cp) < 2:
        return np.column_stack(([freqs[0], freqs[-1]], [raw[0], raw[-1]]))
    
    changed = True
    iterations = 0
    
    while changed and len(cp) > 2 and iterations < max_iterations:
        changed = False
        iterations += 1
        best_removal = None
        best_error_increase = float('inf')
        
        # Get current reconstruction error
        _, baseline_recon = reconstruct_hrtf(cp, method, freqs)
        baseline_error = np.max(np.abs(raw - baseline_recon))
        
        # Try removing each interior point
        for i in range(1, len(cp) - 1):
            candidate = np.delete(cp, i, axis=0)
            _, recon = reconstruct_hrtf(candidate, method, freqs)
            max_err = np.max(np.abs(raw - recon))
            error_increase = max_err - baseline_error
            
            if max_err <= error_tol and error_increase < best_error_increase:
                best_error_increase = error_increase
                best_removal = i
        
        if best_removal is not None:
            cp = np.delete(cp, best_removal, axis=0)
            changed = True
            continue
        
        # If no single removal works, try merging adjacent points
        best_merge = None
        best_error_increase = float('inf')
        
        for i in range(len(cp) - 1):
            # Try different merging strategies
            merge_strategies = [
                # Geometric mean freq, arithmetic mean mag
                (np.sqrt(cp[i, 0] * cp[i+1, 0]), (cp[i, 1] + cp[i+1, 1]) / 2),
                # Arithmetic mean for both
                ((cp[i, 0] + cp[i+1, 0]) / 2, (cp[i, 1] + cp[i+1, 1]) / 2),
            ]
            
            for merged_freq, merged_mag in merge_strategies:
                candidate = cp.copy()
                candidate[i] = [merged_freq, merged_mag]
                candidate = np.delete(candidate, i+1, axis=0)
                
                try:
                    _, recon = reconstruct_hrtf(candidate, method, freqs)
                    max_err = np.max(np.abs(raw - recon))
                    error_increase = max_err - baseline_error
                    
                    if max_err <= error_tol and error_increase < best_error_increase:
                        best_error_increase = error_increase
                        best_merge = (i, merged_freq, merged_mag)
                        break
                except:
                    continue
        
        if best_merge is not None:
            i, merged_freq, merged_mag = best_merge
            cp[i] = [merged_freq, merged_mag]
            cp = np.delete(cp, i+1, axis=0)
            changed = True
    
    return cp




def dp_optimize_with_validation(freqs, raw, error_tol=1.0, method='pchip', max_attempts=5):
    """
    DP optimization with iterative refinement if error exceeds tolerance
    """
    current_tol = error_tol * 0.8  # Start with stricter tolerance
    
    for attempt in range(max_attempts):
        cp = dp_optimize_control_points_fixed(freqs, raw, current_tol, method)
        
        # Validate the result
        _, recon = reconstruct_hrtf(cp, method, freqs)
        actual_error = np.max(np.abs(raw - recon))
        
        # print(f"Attempt {attempt + 1}: {len(cp)} control points, error = {actual_error:.3f}")
        
        if actual_error <= error_tol:
            return cp
        
        # If error too high, be more conservative
        current_tol *= 0.7
        
        # Also try adding intermediate points where error is highest
        if attempt < max_attempts - 1:
            error_diff = np.abs(raw - recon)
            worst_indices = np.argsort(error_diff)[-5:]  # Top 5 worst errors
            
            # Add some intermediate points
            additional_points = []
            for idx in worst_indices:
                if not np.any(np.isclose(cp[:, 0], freqs[idx], atol=1e-5)):
                    additional_points.append([freqs[idx], raw[idx]])
            
            if additional_points:
                cp = np.vstack([cp, additional_points])
                cp = cp[np.argsort(cp[:, 0])]
    
    # Final fallback: use aggressive pruning instead
    # print("DP failed, falling back to aggressive pruning")
    return aggressive_prune_control_points_enhanced(cp, freqs, raw, method, error_tol)





def multi_strategy_optimize(freqs, raw, error_tol=1.0, method='pchip'):
    """
    Try multiple optimization strategies and return the best one
    """
    # print(f"Starting multi-strategy optimization with {len(freqs)} frequency points")
    
    strategies = []
    
    # Strategy 1: Enhanced aggressive pruning on current control points
    def strategy_aggressive(cp_input):
        return aggressive_prune_control_points_enhanced(cp_input, freqs, raw, method, error_tol)
    
    # Strategy 2: DP with validation
    def strategy_dp():
        return dp_optimize_with_validation(freqs, raw, error_tol, method)
    
    # Strategy 3: Hybrid coarse-to-fine
    def strategy_hybrid():
        # Start with coarse sampling
        N = len(freqs)
        initial_step = max(N // 15, 1)
        indices = list(range(0, N, initial_step))
        if indices[-1] != N-1:
            indices.append(N-1)
        
        cp = np.column_stack((freqs[indices], raw[indices]))
        
        # Iteratively add worst-error points
        max_iterations = 30
        for iteration in range(max_iterations):
            _, recon = reconstruct_hrtf(cp, method, freqs)
            error = np.abs(raw - recon)
            max_error = np.max(error)
            
            if max_error <= error_tol:
                break
            
            # Add worst error points
            cp_freq_set = set(cp[:, 0])
            candidates = []
            
            for i in np.argsort(error)[::-1]:
                if freqs[i] not in cp_freq_set and error[i] > error_tol * 0.3:
                    candidates.append((freqs[i], raw[i]))
                    if len(candidates) >= 2:
                        break
            
            if not candidates:
                break
            
            new_points = np.array(candidates)
            cp = np.vstack([cp, new_points])
            cp = cp[np.argsort(cp[:, 0])]
        
        # Final pruning
        return aggressive_prune_control_points_enhanced(cp, freqs, raw, method, error_tol)
    
    # Test strategies
    strategies_to_try = [
        # ("DP with Validation", strategy_dp),
        # ("Hybrid Coarse-to-Fine", strategy_hybrid),
    ]
    
    best_cp = None
    best_count = float('inf')
    best_strategy = None
    
    for strategy_name, strategy_func in strategies_to_try:
        try:
            # print(f"Trying {strategy_name}...")
            cp = strategy_func()
            
            # Validate result
            _, recon = reconstruct_hrtf(cp, method, freqs)
            actual_error = np.max(np.abs(raw - recon))
            
            # print(f"{strategy_name}: {len(cp)} points, error = {actual_error:.3f}")
            
            if actual_error <= error_tol and len(cp) < best_count:
                best_cp = cp
                best_count = len(cp)
                best_strategy = strategy_name
                
        except Exception as e:
            print(f"{strategy_name} failed: {e}")
    
    # print(f"Best strategy: {best_strategy} with {best_count} control points")
    return best_cp if best_cp is not None else np.column_stack(([freqs[0], freqs[-1]], [raw[0], raw[-1]]))


# MODIFIED adaptive_compress_hrtf function - REPLACE the existing one in Code 1
def adaptive_compress_hrtf(ir, fs, fft_len=1024,
                           init_pts=10, global_thresh=1.0, local_thresh=1.0, max_pts=100,
                           win_size=11, err_thresh=1.0, alt_win=7, noise=1e-1,
                           spline_method='pchip',
                           smooth_method='wiener',
                           n_bands=42, frac_oct=3, lifter_len=20,
                           seg_thresh=0.9, err_bound=0.99):
    """
    Compress a single HRTF measurement using an adaptive control point selection scheme.
    """
    # Compute the frequency response and convert to dB scale
    freqs, mag = compute_frequency_response(ir, fs, fft_len)

    # Apply adaptive filtering to smooth the magnitude response
    smooth = apply_filter(mag, freqs=freqs, method=smooth_method,
                          win_size=win_size, err_thresh=err_thresh, alt_win=alt_win, noise=noise,
                          n_bands=n_bands, frac_oct=frac_oct, lifter_len=lifter_len)

    raw = mag.copy()

    # Find control points using the smoothed magnitude spectrum
    cp_idx = find_control_points_via_second_derivative(freqs, smooth, threshold=0.1)
    cp_freqs = freqs[cp_idx]
    cp_mags = mag[cp_idx]

    # Combine frequency and magnitude into control points
    cp = np.column_stack((cp_freqs, cp_mags))

    # Ensure endpoints match exactly
    cp_freqs[0], cp_mags[0] = freqs[0], raw[0]
    cp_freqs[-1], cp_mags[-1] = freqs[-1], raw[-1]
    cp = np.column_stack((cp_freqs, cp_mags))

    # Record the initial number of control points
    initial_cp_count = len(cp)

    # Initial refinement and pruning
    cp = refine_cp_by_segment(cp, freqs, raw, spline_method, seg_thresh, min_distance=5)
    cp = enforce_endpoints(cp, freqs, raw)
    cp = enhanced_prune_control_points(cp, freqs, raw, method=spline_method, threshold=global_thresh, dense_factor=12)
    cp = enforce_endpoints(cp, freqs, raw)
    cp = erb_aware_pruning(cp, freqs, raw, method=spline_method, global_thresh=global_thresh)
    cp = enforce_endpoints(cp, freqs, raw)

    # Apply the new multi-strategy optimization instead of the old merging
    # print(f"Before multi-strategy optimization: {len(cp)} control points")
    cp = multi_strategy_optimize(freqs, raw, error_tol=1.0, method=spline_method)
    # print(f"After multi-strategy optimization: {len(cp)} control points")

    # Final reconstruction and validation
    _, recon = reconstruct_hrtf(cp, spline_method, freqs)
    max_abs_error = np.max(np.abs(raw - recon))

    # If error is still too high, add points iteratively (safety net)
    safety_iterations = 0
    while max_abs_error > 1.0 and safety_iterations < 10:
        safety_iterations += 1
        error = raw - recon
        idx = np.argmax(np.abs(error))
        if np.abs(error[idx]) > 1.0:
            new_freq = freqs[idx]
            new_mag = raw[idx]
            if not np.any(np.isclose(cp[:, 0], new_freq, atol=1e-5)):
                cp = np.vstack([cp, [new_freq, new_mag]])
                cp = cp[np.argsort(cp[:, 0])]
                _, recon = reconstruct_hrtf(cp, spline_method, freqs)
                max_abs_error = np.max(np.abs(raw - recon))
                print(f"Safety net: Added point, now {len(cp)} control points, error = {max_abs_error:.3f}")
            else:
                break
        else:
            break

    # Final post-processing cleanup
    cp = post_process_control_points(cp, freqs, raw, method=spline_method, error_tol=1.0)
    
    # Final reconstruction for output
    _, recon = reconstruct_hrtf(cp, spline_method, freqs)
    final_cp_count = len(cp)
    points_added = max(final_cp_count - initial_cp_count, 0)
    points_removed = max(initial_cp_count - final_cp_count, 0)
    sd = spectral_distortion(raw, recon)

    print(f"Final result: {final_cp_count} control points, error = {np.max(np.abs(raw - recon)):.3f}, SD = {sd:.3f}")

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
def compress_all_hrtfs(sofa_path, adaptive=True,
                       init_pts=10, global_thresh=1.0,
                       max_pts=100, win_size=11, err_thresh=1.0, alt_win=7, noise=1e-1,
                       local_thresh=1.0, spline_method='pchip', fft_len=1024,
                       n_bands=42, frac_oct=3, lifter_len=20,
                       smooth_method='wiener',
                       max_meas=5, seg_thresh=0.9, err_bound=0.99):
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
    # print(f"Loaded {os.path.basename(sofa_path)}, sampling rate = {fs} Hz")
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
            left_res = adaptive_compress_hrtf(
                left_ir, fs, fft_len,
                init_pts, global_thresh, local_thresh, max_pts,
                win_size, err_thresh, alt_win, noise,
                spline_method,
                smooth_method,  # ← forward
                n_bands, frac_oct, lifter_len,
                seg_thresh, err_bound)
            right_res = adaptive_compress_hrtf(
                right_ir, fs, fft_len,
                init_pts, global_thresh, local_thresh, max_pts,
                win_size, err_thresh, alt_win, noise,
                spline_method,
                smooth_method,  # ← forward
                n_bands, frac_oct, lifter_len,
                seg_thresh, err_bound)
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
    ax.set_ylim(0, 1.1)
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
    ax1.set_title(title1, fontsize=35)
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
    ax2.set_ylim(-1.2, 1.2)
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
    ax.set_ylim(-0.5, 0.5)
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
    plt.ylim(0, 1)
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
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.show()


# -----------------------------
# Main Interactive Interface
# -----------------------------
def main():
    compressed_db_path = "src/sofa/testing"
    os.makedirs(compressed_db_path, exist_ok=True)
    sofa_dir = "src/sofa/"
    sofa_files = glob.glob(os.path.join(sofa_dir, "*.sofa"))[:1]
    if not sofa_files:
        print(f"No SOFA files found in '{sofa_dir}'")
        sys.exit(1)

    # Parameters for compression and filtering
    FFT_LEN = 1024
    SG_WINDOW = 3
    ERROR_THRESHOLD = 0.95
    ALT_WINDOW = 3
    NOISE_EST = 0.1
    NUM_BANDS = 42
    INIT_POINTS = 10
    THRESHOLD_SD = 0.95
    LOCAL_THRESHOLD = 0.95
    MAX_POINTS = 100
    SPLINE_KIND = 'pchip'
    SEGMENT_THRESHOLD = 0.9
    ERROR_BOUND = 0.95
    SMOOTH_METHOD = 'wiener'
    
    all_left_results, all_right_results, all_positions, all_file_indices = [], [], [], []
    def _run_file(sofa_path, file_idx):
        print(f"Processing file {file_idx}: {os.path.basename(sofa_path)}")
        compressed_db = compress_all_hrtfs(
            sofa_path,
            adaptive=True,
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
            frac_oct=6,
            lifter_len=80,
            smooth_method=SMOOTH_METHOD,
            max_meas=1,
            seg_thresh=SEGMENT_THRESHOLD,
            err_bound=ERROR_BOUND
        )
        file_name = os.path.basename(sofa_path)
        with open(os.path.join(compressed_db_path, file_name.replace('sofa', 'pkl')), 'wb') as f:
            pickle.dump(compressed_db, f, protocol=pickle.HIGHEST_PROTOCOL)
        return compressed_db
    print('entring loop')
    compressed_dbs = Parallel(n_jobs=50)(
        delayed(_run_file)(sofa_path, file_idx)
        for file_idx, sofa_path in tqdm(enumerate(sofa_files, start=1), total=len(sofa_files), desc="Processing SOFA files")
    )
    
if __name__ == "__main__":
    main()
