import os
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox


# ——————————————————————————————————————————————————————
# Constants
# ——————————————————————————————————————————————————————
FREQ_MIN = 20
FREQ_MAX = 20000
DEFAULT_ERB_BANDS = 42
PKL_DIR = "src/sofa/FinalResultsPKL"


# ——————————————————————————————————————————————————————
# ERB-Scale Utilities
# ——————————————————————————————————————————————————————
def frequency_to_erb(frequency_hz: float) -> float:
    """
    Convert frequency in Hz to the ERB-rate scale.
    """
    return 24.7 * (4.37 * frequency_hz / 1000 + 1)


def erb_to_frequency(erb_value: float) -> float:
    """
    Convert ERB-rate value back to frequency in Hz.
    """
    return (erb_value / 24.7 - 1) * 1000 / 4.37


def get_erb_bands(freq_min: float, freq_max: float, n_bands: int) -> np.ndarray:
    """
    Divide [freq_min, freq_max] into n_bands equally spaced ERB intervals,
    returning the frequency edges for each band.
    """
    erb_min = frequency_to_erb(freq_min)
    erb_max = frequency_to_erb(freq_max)
    erb_edges = np.linspace(erb_min, erb_max, n_bands + 1)
    return erb_to_frequency(erb_edges)


def calculate_band_spectral_distortion(
    freqs: np.ndarray,
    orig_mag: np.ndarray,
    recon_mag: np.ndarray,
    erb_bounds: np.ndarray
) -> list[float]:
    """
    Compute per-ERB-band spectral distortion (RMS error) between orig_mag and recon_mag.
    """
    sd_list = []
    for i in range(len(erb_bounds) - 1):
        band_lo, band_hi = erb_bounds[i], erb_bounds[i + 1]
        mask = (freqs >= band_lo) & (freqs < band_hi)
        if np.any(mask):
            error_rms = np.sqrt(np.mean((orig_mag[mask] - recon_mag[mask]) ** 2))
            sd_list.append(error_rms)
        else:
            sd_list.append(np.nan)
    return sd_list


# ——————————————————————————————————————————————————————
# Plotting Routines
# ——————————————————————————————————————————————————————
def plot_reconstruction(ax, result: dict, title_suffix: str = ""):
    """
    Plot original vs. reconstructed magnitude response with control points.
    """
    ax.clear()
    freqs = result["frequencies"]
    raw_mag = result["raw_mag"]
    recon_mag = result["reconstructed"]
    control_pts = result.get("control_points", None)
    spectral_dist = result.get("spectral_distortion", None)
    num_cp = result.get("num_control_points", "N/A")

    ax.semilogx(freqs, raw_mag, color="blue", alpha=0.7, label="Original")
    label_recon = (
        f"Reconstructed (SD={spectral_dist:.2f} dB, CP={num_cp})"
        if spectral_dist is not None
        else "Reconstructed"
    )
    ax.semilogx(freqs, recon_mag, color="black", linestyle="--", label=label_recon)

    if control_pts is not None:
        ax.scatter(
            control_pts[:, 0],
            control_pts[:, 1],
            color="red",
            s=20,
            marker="o",
            label="Control Points",
        )

    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(-50, 5)
    ax.set_xlabel("Frequency (Hz)", fontsize=18)
    ax.set_ylabel("Magnitude (dB)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    if title_suffix:
        ax.set_title(f"Reconstruction {title_suffix}", fontsize=16)


def plot_sd_boxplot(all_results: list[dict], n_bands: int = DEFAULT_ERB_BANDS):
    """
    Create a boxplot of per-ERB-band spectral distortion across a list of results.
    """
    erb_bounds = get_erb_bands(FREQ_MIN, FREQ_MAX, n_bands)
    band_errors = [[] for _ in range(n_bands)]

    for res in all_results:
        sd_vector = calculate_band_spectral_distortion(
            res["frequencies"],
            res["smoothed_mag"],
            res["reconstructed"],
            erb_bounds,
        )
        for idx, val in enumerate(sd_vector):
            band_errors[idx].append(val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(band_errors, showfliers=True)
    x_ticks = [str(i + 1) for i in range(n_bands)]
    ax.set_xticklabels(x_ticks, rotation=90, fontsize=6)
    ax.set_xlabel("ERB Band", fontsize=14)
    ax.set_ylabel("Spectral Distortion (dB)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.show()


def plot_average_filtering_error(all_results: list[dict]):
    """
    Plot the average absolute filtering error (|smoothed_mag - raw_mag|) with 95% CI.
    """
    freqs = all_results[0]["frequencies"]
    errors = np.array(
        [np.abs(r["smoothed_mag"] - r["raw_mag"]) for r in all_results]
    )  # shape: (N, len(freqs))

    avg_error = np.mean(errors, axis=0)
    sem = np.std(errors, axis=0) / np.sqrt(len(all_results))
    ci_half_width = 1.96 * sem

    print(f"Max 95% CI half-width = {np.max(ci_half_width):.4f} dB")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        freqs,
        avg_error - ci_half_width,
        avg_error + ci_half_width,
        color="lightcoral",
        alpha=0.5,
        label="95% CI",
        zorder=1,
    )
    ax.semilogx(
        freqs, avg_error, color="red", linewidth=2, label="Avg |Filtering Error|", zorder=2
    )

    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_xlabel("Frequency (Hz)", fontsize=18)
    ax.set_ylabel("Absolute Error (dB)", fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_details(result: dict, save_pdf: bool = True):
    """
    Plot original vs. reconstructed and error curve for a single measurement.
    Optionally save as a PDF.
    """
    freqs = result["frequencies"]
    raw_mag = result["raw_mag"]
    recon_mag = result["reconstructed"]
    control_pts = result.get("control_points", None)
    file_name = result.get("file_name", "Unknown")
    az = result.get("azimuth", np.nan)
    el = result.get("elevation", np.nan)
    error = recon_mag - raw_mag

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top plot: original & reconstructed with control points
    ax_top.semilogx(freqs, raw_mag, color="blue", label="Original")
    ax_top.semilogx(freqs, recon_mag, color="red", linestyle="--", label="Reconstructed")
    if control_pts is not None:
        ax_top.scatter(
            control_pts[:, 0], control_pts[:, 1], color="black", s=25, marker="o", label="Control Points"
        )
    ax_top.set_ylabel("Magnitude (dB)", fontsize=16)
    ax_top.legend(fontsize=12)
    ax_top.grid(True, which="both", linestyle="--", alpha=0.5)
    title = f"{file_name} | Az={az:.1f}°, El={el:.1f}°"
    ax_top.set_title(f"Measurement: {title}", fontsize=14)
    ax_top.tick_params(axis="both", which="major", labelsize=12)

    # Bottom plot: error curve
    ax_bottom.semilogx(freqs, error, color="black")
    ax_bottom.set_xlabel("Frequency (Hz)", fontsize=18)
    ax_bottom.set_ylabel("Error (dB)", fontsize=16)
    ax_bottom.tick_params(axis="both", which="major", labelsize=12)
    ax_bottom.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_pdf:
        plt.savefig("ErrorDetails301Subjects.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()


def plot_filtering_error(result: dict):
    """
    Plot per-frequency filtering error for a single measurement.
    """
    freqs = result["frequencies"]
    error = result["smoothed_mag"] - result["raw_mag"]
    mean_abs_error = np.mean(np.abs(error))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(
        freqs,
        error,
        color="green",
        label=f"Per-bin Error (Mean |error| = {mean_abs_error:.2f} dB)",
    )
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel("Error (dB)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_filtering_stats(result: dict):
    """
    Bar chart of maximum absolute filtering error and standard deviation for a single measurement.
    """
    error = result["smoothed_mag"] - result["raw_mag"]
    abs_err = np.max(np.abs(error))
    std_err = np.std(error)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Max |Error|", "Std Dev"], [abs_err, std_err], color=["orange", "skyblue"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Error (dB)", fontsize=14)
    ax.set_title("Filtering Statistics", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ——————————————————————————————————————————————————————
# Data Loading & Aggregation
# ——————————————————————————————————————————————————————
def load_all_results(pkl_dir: str):
    pkl_paths = glob.glob(os.path.join(pkl_dir, "*.pkl"))[:1]
    all_left = []
    all_right = []
    positions_list = []
    file_indices = []

    for subj_idx, path in enumerate(pkl_paths, start=1):
        with open(path, "rb") as f:
            subj_data = pickle.load(f)

        file_name = os.path.basename(path)
        left_results = subj_data["left_results"]
        right_results = subj_data["right_results"]
        positions = subj_data.get("positions", np.empty((0, 2)))  # could be >2 columns
        n_meas = len(left_results)

        # Pad or truncate to exactly n_meas rows
        if positions.shape[0] < n_meas:
            pad_rows = np.full((n_meas - positions.shape[0], positions.shape[1]), np.nan)
            positions = np.vstack((positions, pad_rows))
        else:
            positions = positions[:n_meas, :]

        # Inject only the first two columns (azimuth, elevation)
        for i in range(n_meas):
            azimuth, elevation = positions[i, :2]
            left_results[i].update(
                {"file_name": file_name, "azimuth": azimuth, "elevation": elevation}
            )
            right_results[i].update(
                {"file_name": file_name, "azimuth": azimuth, "elevation": elevation}
            )

        all_left.extend(left_results)
        all_right.extend(right_results)
        positions_list.append(positions[:, :2])   # store only the first two columns
        file_indices.extend([subj_idx] * n_meas)

    stacked_positions = np.vstack(positions_list)
    return all_left, all_right, stacked_positions, file_indices



# ——————————————————————————————————————————————————————
# Summary Metric Computations
# ——————————————————————————————————————————————————————
def compute_control_point_stats(
    left_results: list[dict], right_results: list[dict], freqs: np.ndarray
) -> None:
    """
    Compute and print statistics about control points and compression ratios.
    """
    N = len(left_results)
    # Total control points per measurement (both ears)
    total_cp_counts = [
        l["num_control_points"] + r["num_control_points"]
        for l, r in zip(left_results, right_results)
    ]
    avg_total_cp = np.mean(total_cp_counts)
    sigma_cp = np.std(total_cp_counts)

    avg_left_cp = np.mean([l["num_control_points"] for l in left_results])
    avg_right_cp = np.mean([r["num_control_points"] for r in right_results])
    avg_per_ear_cp = 0.5 * (avg_left_cp + avg_right_cp)

    print(f"Average #CP (both ears): {avg_total_cp:.2f}")
    print(f"Control-point count σ: {sigma_cp:.2f}")
    print(f"Average #CP (left ear): {avg_left_cp:.2f}")
    print(f"Average #CP (right ear): {avg_right_cp:.2f}")
    print(f"Average #CP per ear: {avg_per_ear_cp:.2f}")

    # Full-band compression ratio (0–20 kHz)
    original_bytes_full = len(freqs) * 2 * 4  # 2 ears, 4 bytes per float
    combined_cr_full = []
    for cp_count in total_cp_counts:
        if cp_count > 0:
            combined_cr_full.append(original_bytes_full / (cp_count * 8))
    if combined_cr_full:
        print(
            f"Avg full-band compression ratio (0–20 kHz): {np.mean(combined_cr_full):.2f}×"
        )
        print(f"STD of full-band CR: {np.std(combined_cr_full, ddof=1):.2f}×")
    else:
        print("No valid measurements for full-band CR.")

    # Band-limited compression ratio (100–16 kHz)
    mask_band = (freqs >= 100) & (freqs <= 16000)
    original_bytes_band = np.sum(mask_band) * 2 * 4
    combined_cr_band = []
    cp_counts_band = []

    for l_res, r_res in zip(left_results, right_results):
        cp_l = l_res["control_points"]
        cp_r = r_res["control_points"]
        cp_l_in = np.sum((cp_l[:, 0] >= 100) & (cp_l[:, 0] <= 16000))
        cp_r_in = np.sum((cp_r[:, 0] >= 100) & (cp_r[:, 0] <= 16000))
        total_cp_in = cp_l_in + cp_r_in
        cp_counts_band.append(total_cp_in)
        if total_cp_in > 0:
            combined_cr_band.append(original_bytes_band / (total_cp_in * 8))

    if combined_cr_band:
        print(f"Avg band-limited CR (100–16 kHz): {np.mean(combined_cr_band):.2f}×")
        print(
            f"Average #CP (100–16 kHz): {np.mean(cp_counts_band):.2f}"
        )
    else:
        print("No control points in 100–16 kHz for any measurement.")


def compute_filtering_error_stats(left_results: list[dict]) -> None:
    """
    Compute and print statistics about filtering error for the left ear across all measurements.
    """
    abs_errs = [np.max(np.abs(res["smoothed_mag"] - res["raw_mag"])) for res in left_results]
    std_errs = [np.std(res["smoothed_mag"] - res["raw_mag"]) for res in left_results]

    print(f"Average absolute filtering error (left ear): {np.mean(abs_errs):.2f} dB")
    print(f"Average filtering-error std dev (left ear): {np.mean(std_errs):.2f} dB")


def compute_spectral_distortion_stats(
    left_results: list[dict], right_results: list[dict]
) -> None:
    """
    Compute and print average and standard deviation of full-band spectral distortion.
    """
    # Left-ear SD
    sds_left = [
        np.sqrt(np.mean((res["raw_mag"] - res["reconstructed"]) ** 2))
        for res in left_results
    ]
    avg_sd_left = np.mean(sds_left)
    print(f"Avg full-band SD (left ear): {avg_sd_left:.3f} dB")

    # Both-ears SD per measurement
    both_sds = []
    for l_res, r_res in zip(left_results, right_results):
        sd_l = np.sqrt(np.mean((l_res["raw_mag"] - l_res["reconstructed"]) ** 2))
        sd_r = np.sqrt(np.mean((r_res["raw_mag"] - r_res["reconstructed"]) ** 2))
        both_sds.append(0.5 * (sd_l + sd_r))
    avg_sd_both = np.mean(both_sds)
    print(f"Avg full-band SD (both ears): {avg_sd_both:.3f} dB")

    # Pooled σ over all SDs (left + right)
    sds_right = [
        np.sqrt(np.mean((res["raw_mag"] - res["reconstructed"]) ** 2))
        for res in right_results
    ]
    all_sds = sds_left + sds_right
    sigma_sd = np.std(all_sds)
    print(f"Spectral-distortion σ (all meas): {sigma_sd:.3f} dB")

    # Overall SD across all measurements (both ears)
    avg_sd_all = np.mean(all_sds)
    sigma_sd_all = np.std(all_sds)
    print(
        f"Avg full-band SD (all meas): {avg_sd_all:.3f} ± {sigma_sd_all:.3f} dB"
    )


def compute_erb_band_sd_overall(
    left_results: list[dict], right_results: list[dict], n_bands: int = DEFAULT_ERB_BANDS
) -> None:
    """
    Compute and print average ERB-band spectral distortion across all measurements.
    """
    erb_bounds = get_erb_bands(FREQ_MIN, FREQ_MAX, n_bands)
    band_errs = [[] for _ in range(n_bands)]

    for res in left_results + right_results:
        sd_vector = calculate_band_spectral_distortion(
            res["frequencies"],
            res["raw_mag"],
            res["reconstructed"],
            erb_bounds,
        )
        for idx, val in enumerate(sd_vector):
            band_errs[idx].append(val)

    avg_per_band = [np.nanmean(band) for band in band_errs]
    avg_band_sd = np.nanmean(avg_per_band)
    print(f"Avg ERB-band SD (all meas): {avg_band_sd:.3f} dB")


# ——————————————————————————————————————————————————————
# Interactive Plot Controls
# ——————————————————————————————————————————————————————
def setup_interactive_reconstruction(
    fig, ax, left_results: list[dict], positions: np.ndarray, file_indices: list[int]
):
    """
    Set up the slider and text boxes to interactively visualize reconstructions and statistics.
    """
    N = len(left_results)
    plt.subplots_adjust(left=0.1, bottom=0.4)

    # Slider for choosing measurement index
    slider_ax = plt.axes([0.15, 0.33, 0.7, 0.03])
    meas_slider = Slider(
        slider_ax, "Measurement", 0, N - 1, valinit=0, valstep=1
    )

    def update_recon(index):
        idx = int(index)
        res = left_results[idx]
        subj_idx = file_indices[idx]
        az, el = positions[idx, :]
        suffix = f"(subj={subj_idx}, idx={idx}, Az={az:.1f}°, El={el:.1f}°)"
        plot_reconstruction(ax, res, suffix)
        fig.canvas.draw_idle()

    update_recon(0)
    meas_slider.on_changed(update_recon)

    # TextBox for SD Boxplot
    sd_ax = plt.axes([0.15, 0.25, 0.7, 0.04])
    sd_box = TextBox(sd_ax, "SD Boxplot", initial="all")

    def on_sd_submit(text):
        text = text.strip().lower()
        if text == "all":
            selection = left_results
        else:
            try:
                indices = [int(x) for x in text.split(",")]
                selection = [left_results[i] for i in indices]
            except (ValueError, IndexError):
                selection = left_results
        plot_sd_boxplot(selection)

    sd_box.on_submit(on_sd_submit)

    # TextBox for Error Details
    err_ax = plt.axes([0.15, 0.18, 0.7, 0.04])
    err_box = TextBox(err_ax, "Error Details", initial="0")

    def on_error_submit(text):
        try:
            idx = int(text.strip())
            plot_error_details(left_results[idx])
        except (ValueError, IndexError):
            pass

    err_box.on_submit(on_error_submit)

    # TextBox for Filtering Error
    filt_ax = plt.axes([0.15, 0.11, 0.7, 0.04])
    filt_box = TextBox(filt_ax, "Filtering Error", initial="0")

    def on_filter_submit(text):
        try:
            idx = int(text.strip())
            plot_filtering_error(left_results[idx])
        except (ValueError, IndexError):
            pass

    filt_box.on_submit(on_filter_submit)

    # TextBox for Filtering Stats
    stats_ax = plt.axes([0.15, 0.04, 0.7, 0.04])
    stats_box = TextBox(stats_ax, "Filtering Stats", initial="0")

    def on_stats_submit(text):
        try:
            idx = int(text.strip())
            plot_filtering_stats(left_results[idx])
        except (ValueError, IndexError):
            pass

    stats_box.on_submit(on_stats_submit)


# ——————————————————————————————————————————————————————
# Main Execution Flow
# ——————————————————————————————————————————————————————
def main():
    # 1) Load and aggregate all measurements
    left_results, right_results, positions, file_indices = load_all_results(PKL_DIR)
    if not left_results:
        print(f"No PKL files found in {PKL_DIR}. Exiting.")
        return

    freqs = left_results[0]["frequencies"]
    total_measurements = len(left_results)
    print(f"Loaded {len(set(file_indices))} subjects → {total_measurements} total measurements.")

    # 2) Compute and display summary statistics
    compute_control_point_stats(left_results, right_results, freqs)
    compute_filtering_error_stats(left_results)
    compute_spectral_distortion_stats(left_results, right_results)
    compute_erb_band_sd_overall(left_results, right_results, n_bands=DEFAULT_ERB_BANDS)

    # 3) Global plots
    plot_sd_boxplot(left_results, n_bands=DEFAULT_ERB_BANDS)
    plot_average_filtering_error(left_results)

    # 4) Interactive reconstruction viewer
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_interactive_reconstruction(fig, ax, left_results, positions, file_indices)
    plt.show()


if __name__ == "__main__":
    main()
