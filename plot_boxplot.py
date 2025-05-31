import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import os
import glob

# ——————————————————————————————————————————————————————
# ERB
# ——————————————————————————————————————————————————————
def frequency_to_erb(f):
    return 24.7 * (4.37 * f / 1000 + 1)

def erb_to_frequency(erb):
    return (erb / 24.7 - 1) * 1000 / 4.37

def get_erb_bands(freq_min, freq_max, n_bands):
    erb_min   = frequency_to_erb(freq_min)
    erb_max   = frequency_to_erb(freq_max)
    erb_edges = np.linspace(erb_min, erb_max, n_bands + 1)
    return erb_to_frequency(erb_edges)

def calculate_band_sd(freqs, orig, recon, erb_bounds):
    sd = []
    for i in range(len(erb_bounds)-1):
        lo, hi = erb_bounds[i], erb_bounds[i+1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            sd.append(np.sqrt(np.mean((orig[mask] - recon[mask])**2)))
        else:
            sd.append(np.nan)
    return sd

# ——————————————————————————————————————————————————————
# Plotting routines
# ——————————————————————————————————————————————————————
def plot_reconstruction(ax, res, suffix=""):
    ax.clear()
    f    = res['frequencies']
    raw  = res['raw_mag']
    rc   = res['reconstructed']
    cp   = res.get('control_points', None)
    sd   = res.get('spectral_distortion', None)
    nc   = res.get('num_control_points', 'N/A')
    ax.semilogx(f, raw, 'b-', alpha=0.7, label="Original")
    lbl = f"Recon (SD={sd:.2f}, CP={nc})" if sd is not None else "Recon"
    ax.semilogx(f, rc, 'k--', label=lbl)
    if cp is not None:
        ax.plot(cp[:,0], cp[:,1], 'ro', markersize=4)
    ax.set_xlim(20, 22000)
    ax.set_ylim(-50, 5)
    # ax.set_title("Reconstruction " + suffix, fontsize=15)
    ax.set_xlabel("Frequency (Hz)", fontsize=22)
    ax.set_ylabel("Magnitude (dB)" , fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.legend(fontsize=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

def plot_sd_boxplot(all_left, n_bands=42):
    erb_bounds = get_erb_bands(20, 20000, n_bands)
    band_errs  = [[] for _ in range(n_bands)]
    for res in all_left:
        sd = calculate_band_sd(res['frequencies'],
                               res['smoothed_mag'],
                               res['reconstructed'],
                               erb_bounds)
        for i, v in enumerate(sd):
            band_errs[i].append(v)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.boxplot(band_errs, showfliers=True)
    ax.set_xticklabels([str(i+1) for i in range(n_bands)],
                       rotation=90, fontsize=6)
    ax.set_xlabel("ERB Band", fontsize=22)
    ax.set_ylabel("Spectral Distortion (dB)", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18) 
    # ax.set_title("Per-band Spectral Distortion Across All Measurements")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.show()

def plot_average_filtering_error(all_left):
    f    = all_left[0]['frequencies']
    errs = np.abs([r['smoothed_mag'] - r['raw_mag'] for r in all_left])
    avg  = np.mean(errs, axis=0)
    se   = np.std(errs, axis=0) / np.sqrt(len(all_left))
    ci   = 1.96 * se

    print(f"Max CI half-width = {np.max(ci):.4f} dB")   # debug

    fig, ax = plt.subplots(figsize=(10,4))

    # draw the fill _first_, with higher alpha and contrasting color
    ax.fill_between(f, avg-ci, avg+ci,
                    facecolor='lightcoral',   # a light red
                    alpha=0.5,                # more opaque
                    label="95 % CI",
                    zorder=1)

    # then the average curve on top
    ax.semilogx(f, avg, 'r-', linewidth=2, label="Avg |filter error|", zorder=2)

    ax.set_xlim(20, 22000)
    ax.set_xlabel("Frequency (Hz)", fontsize=26)
    ax.set_ylabel("Absolute Error (dB)", fontsize=26)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    #  zoom in to that narrow band:
    # ymin, ymax = np.min(avg-ci), np.max(avg+ci)
    # ax.set_ylim(ymin - 0.01, ymax + 0.01)

    plt.tight_layout()
    plt.show()




def plot_error_details(res):
    f    = res['frequencies']
    raw  = res['raw_mag']
    rc   = res['reconstructed']
    fn = res.get('file_name', 'Unknown')
    az  = res.get('azimuth', np.nan)
    el  = res.get('elevation', np.nan)
    err  = rc - raw
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax1.semilogx(f, raw, 'b-', label="Original")
    ax1.semilogx(f, rc, 'r--', label="Reconstructed")
    cp = res.get('control_points', None)
    if cp is not None:
        ax1.plot(cp[:,0], cp[:,1], 'ko', ms=5.5, label="Control Pts")
    ax1.set_ylabel("Mag (dB)")
    ax1.legend(fontsize=17)
    # ax1.set_xlabel("Frequency (Hz)", fontsize=20)  # X-axis label font size
    ax1.set_ylabel("Magnitude(dB)", fontsize=24)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    # ax1.set_title(f"Measurement {fn} (Az={az:.1f}°, El={el:.1f}°)", fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=22)  # Major tick font size
    ax2.semilogx(f, err, 'k-')
    ax2.set_xlabel("Frequency (Hz)", fontsize=28); ax2.set_ylabel("Error (dB)",fontsize=24)
    ax2.set_ylabel("Error (dB)", fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=22)  # Major tick font size
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('ErrorDetails301Subjects.pdf', bbox_inches='tight', pad_inches=0.01)
    
    plt.show()

def plot_filtering_error(res):
    f      = res['frequencies']
    err    = res['smoothed_mag'] - res['raw_mag']
    # ← compute mean absolute error
    mean_abs = np.mean(np.abs(err))

    fig, ax = plt.subplots(figsize=(10,4))
    ax.semilogx(f, err, 'g-',
                 label=f"Per-bin Error (Mean |err|={mean_abs:.2f} dB)")
    ax.set_xlim(20, 22000)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Error (dB)")
    ax.legend(fontsize=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_filtering_stats(res):
    err   = res['smoothed_mag'] - res['raw_mag']
    abs_e = np.max(np.abs(err)); std_e = np.std(err)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(['AbsErr','StdDev'], [abs_e, std_e])
    ax.set_ylim(0,1); ax.set_ylabel("dB")
    ax.set_title("Filtering Stats")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()

# ——————————————————————————————————————————————————————
# Main
# ——————————————————————————————————————————————————————
def main():
    # 1) discover all per-subject PKLs
    # pkl_dir   = "src/sofa/testPKL"
    pkl_dir   = "src/sofa/FinalResultsPKL"
    pkl_paths = glob.glob(os.path.join(pkl_dir, "*.pkl"))

    # 2) flat lists for left/right results and metadata
    all_left_results  = []
    all_right_results = []
    all_positions     = []
    all_file_indices  = []

    # 3) load each subject’s PKL and aggregate
    for subj_idx, path in enumerate(pkl_paths, start=1):
        with open(path, 'rb') as f:
            subj = pickle.load(f)
            # … inside the loop over pkl_paths …
            fn = os.path.basename(path)                  # e.g. "P0137.pkl"
            pos = subj.get('positions', np.empty((0,2))) # shape (n,2) of [az,el]

            # make sure pos has as many rows as there are measurements:
            left, right = subj['left_results'], subj['right_results']
            n = len(left)
            if pos.shape[0] < n:
                pad = np.full((n - pos.shape[0], 2), np.nan)
                pos = np.vstack([pos, pad])
            else:
                pos = pos[:n]

            # now inject metadata into each measurement dict:
            for i in range(n):
                left[i]['file_name']  = fn
                left[i]['azimuth']    = pos[i,0]
                left[i]['elevation']  = pos[i,1]
                right[i]['file_name'] = fn
                right[i]['azimuth']   = pos[i,0]
                right[i]['elevation'] = pos[i,1]
        left, right = subj['left_results'], subj['right_results']
        pos         = subj.get('positions', np.empty((0,2)))
        n           = len(left)

        all_left_results .extend(left)
        all_right_results.extend(right)

        # pad or truncate positions to match measurement count
        if pos.shape[0] < n:
            pad = np.full((n - pos.shape[0], 2), np.nan)
            pos = np.vstack([pos, pad])
        else:
            pos = pos[:n]
        all_positions.append(pos)
        all_file_indices.extend([subj_idx] * n)

    # 4) stack positions, grab frequencies & total measurements
    all_positions = np.vstack(all_positions)
    freqs         = all_left_results[0]['frequencies']
    N             = len(all_left_results)
    print(f"Loaded {len(pkl_paths)} PKLs → {N} total measurements.")

    # --- compute summary metrics ---

    # 1) average number of control points (both ears, full-band)
    total_cp = sum(
        l['num_control_points'] + r['num_control_points']
        for l, r in zip(all_left_results, all_right_results)
    )
    avg_cp = total_cp / N
    print(f"Average number of control points (both ears): {avg_cp:.2f}")
    avg_left_cp  = np.mean([l['num_control_points'] for l in all_left_results])
    avg_right_cp = np.mean([r['num_control_points'] for r in all_right_results])
    avg_per_ear = 0.5*(avg_left_cp + avg_right_cp)

    print(f"Avg left‐ear CP:  {avg_left_cp:.2f}")
    print(f"Avg right‐ear CP: {avg_right_cp:.2f}")
    print(f"Avg total CP per measurement (both ears): {avg_cp:.2f}")
    print(f"Avg CP left ear:  {avg_left_cp:.2f}")
    print(f"Avg CP right ear: {avg_right_cp:.2f}")
    print(f"Avg CP per ear (mean of both): {avg_per_ear:.2f}")

       # → pooled σ over all measurements:
    sigma_cp = np.std([l['num_control_points'] + r['num_control_points']
                       for l,r in zip(all_left_results, all_right_results)])
    print(f"Control-point count σ (all meas): {sigma_cp:.2f}")
    # 2) average combined compression ratio
        # — full-band (0–20 kHz)
    original_bytes = len(freqs) * 2 * 4
    combined_crs = []
    for l, r in zip(all_left_results, all_right_results):
        cp_sum = l['num_control_points'] + r['num_control_points']
        if cp_sum > 0:
            combined_crs.append(original_bytes / (cp_sum * 8))

    if combined_crs:
        avg_cr = np.mean(combined_crs)
        std_cr = np.std(combined_crs, ddof=1)   # sample‐std across HRTFs
        print(f"Average combined compression ratio (0–20 kHz): {avg_cr:.2f}×")
        print(f"STD of combined compression ratio   (0–20 kHz): {std_cr:.2f}×")
    else:
        print("No valid measurements for full-band CR.")


    # — band-limited (100–16 kHz)
    mask = (freqs >= 100) & (freqs <= 16000)
    original_bytes_band = np.sum(mask) * 2 * 4

    combined_crs_band = []
    cp_counts_band   = []
    for l, r in zip(all_left_results, all_right_results):
        cp_l = l['control_points']
        cp_r = r['control_points']

        # count only CPs in [100, 16k] Hz
        cp_l_in = np.sum((cp_l[:,0] >= 100) & (cp_l[:,0] <= 16000))
        cp_r_in = np.sum((cp_r[:,0] >= 100) & (cp_r[:,0] <= 16000))
        total_cp_in = cp_l_in + cp_r_in
        cp_counts_band.append(total_cp_in)

        if total_cp_in > 0:
            combined_crs_band.append(original_bytes_band / (total_cp_in * 8))

    if combined_crs_band:
        print(f"Average band-limited CR (100–16 kHz): {np.mean(combined_crs_band):.2f}")
    else:
        print("No control points found in 100–16 kHz for any measurement.")

    # — average number of CPs in 100–16 kHz
    if cp_counts_band:
        print(f"Average number of control points in 100–16 kHz: {np.mean(cp_counts_band):.2f}")
    else:
        print("No measurements to count CPs in 100–16 kHz.")

    # 3) filtering error (left ear)
    abs_errs = [np.max(np.abs(r['smoothed_mag'] - r['raw_mag'])) for r in all_left_results]
    std_errs = [np.std(r['smoothed_mag'] - r['raw_mag'])       for r in all_left_results]
    print(f"Average absolute filtering error (left ear): {np.mean(abs_errs):.2f} dB")
    print(f"Average filtering-error standard deviation (left ear): {np.mean(std_errs):.2f} dB")

    # 4) full-band SD left
    sds_left    = [np.sqrt(np.mean((r['raw_mag'] - r['reconstructed'])**2))
                   for r in all_left_results]
    avg_sd_left = np.mean(sds_left)
    print(f"Average full-band spectral distortion (left ear): {avg_sd_left:.3f} dB")

    # 5) full-band SD both ears
    both_sds = []
    for l, r in zip(all_left_results, all_right_results):
        sd_l = np.sqrt(np.mean((l['raw_mag'] - l['reconstructed'])**2))
        sd_r = np.sqrt(np.mean((r['raw_mag'] - r['reconstructed'])**2))
        both_sds.append(0.5 * (sd_l + sd_r))
    avg_sd_both = np.mean(both_sds)
    print(f"Average full-band spectral distortion (both ears): {avg_sd_both:.3f} dB")
       # → pooled σ over *all* SDs (left+right)
    sds_right = [np.sqrt(np.mean((r['raw_mag'] - r['reconstructed'])**2))
                 for r in all_right_results]
    all_sds = sds_left + sds_right
    sigma_sd = np.std(all_sds)
    print(f"Spectral-distortion σ (all meas): {sigma_sd:.3f} dB")
    # … follow with ERB-band SD, plots, and interactive slider 


    # — NEW METRIC A: overall SD across all measurements
    all_sds     = sds_left + [
        np.sqrt(np.mean((r['raw_mag']-r['reconstructed'])**2))
        for r in all_right_results
    ]
    avg_sd_all   = np.mean(all_sds)
    sigma_sd_all = np.std(all_sds)
    print(f"Average full-band spectral distortion (all measurements): "
          f"{avg_sd_all:.3f} ± {sigma_sd_all:.3f} dB")

    # — NEW METRIC B: average ERB-band SD across all measurements
    erb_bounds  = get_erb_bands(20, 20000, n_bands=42)
    band_errs   = [[] for _ in range(len(erb_bounds)-1)]
    for res in all_left_results + all_right_results:
        sd_vec = calculate_band_sd(res['frequencies'],
                                   res['raw_mag'],
                                   res['reconstructed'],
                                   erb_bounds)
        for i,v in enumerate(sd_vec):
            band_errs[i].append(v)
    avg_band_sd = np.nanmean([np.nanmean(b) for b in band_errs])
    print(f"Average ERB-band spectral distortion: {avg_band_sd:.3f} dB")

    # global plots
    plot_sd_boxplot(all_left_results, n_bands=42)
    plot_average_filtering_error(all_left_results)

    # interactive reconstruction
    # … after computing all_left_results, all_positions, all_file_indices, N …
    # set up the main figure + slider
    fig, ax = plt.subplots(figsize=(12,6))
    plt.subplots_adjust(left=0.1, bottom=0.4)

    # measurement slider
    slider_ax   = plt.axes([0.15, 0.33, 0.7, 0.03])
    meas_slider = Slider(slider_ax, 'Measurement', 0, N-1,
                         valinit=0, valstep=1)

    def update_plot(idx):
        res  = all_left_results[idx]
        subj = all_file_indices[idx]
        az, el = all_positions[idx, :2]
        suffix = f"(subj={subj}, meas={idx}, Az={az:.1f}°, El={el:.1f}°)"
        plot_reconstruction(ax, res, suffix)
        fig.canvas.draw_idle()

    # initial draw
    update_plot(0)
    meas_slider.on_changed(lambda val: update_plot(int(val)))

    # — TextBox for SD Boxplot —
    sd_ax = plt.axes([0.15, 0.25, 0.7, 0.04])
    sd_box = TextBox(sd_ax, 'SD Boxplot', initial="all")
    def on_sd(text):
        text = text.strip().lower()
        if text == "all":
            sel = all_left_results
        else:
            try:
                idxs = [int(x) for x in text.split(',')]
                sel = [all_left_results[i] for i in idxs]
            except:
                sel = all_left_results
        plot_sd_boxplot(sel)
    sd_box.on_submit(on_sd)

    # — TextBox for Error Details —
    err_ax = plt.axes([0.15, 0.18, 0.7, 0.04])
    err_box = TextBox(err_ax, 'Error Details', initial="0")
    def on_error(text):
        try:
            idx = int(text.strip())
            plot_error_details(all_left_results[idx])
        except:
            pass
    err_box.on_submit(on_error)

    # — TextBox for Filtering Error —
    filt_ax = plt.axes([0.15, 0.11, 0.7, 0.04])
    filt_box = TextBox(filt_ax, 'Filtering Error', initial="0")
    def on_filter(text):
        try:
            idx = int(text.strip())
            plot_filtering_error(all_left_results[idx])
        except:
            pass
    filt_box.on_submit(on_filter)

    # — TextBox for Filtering Stats —
    stats_ax = plt.axes([0.15, 0.04, 0.7, 0.04])
    stats_box = TextBox(stats_ax, 'Filtering Stats', initial="0")
    def on_stats(text):
        try:
            idx = int(text.strip())
            plot_filtering_stats(all_left_results[idx])
        except:
            pass
    stats_box.on_submit(on_stats)

    plt.show()


if __name__ == '__main__':
    main()
