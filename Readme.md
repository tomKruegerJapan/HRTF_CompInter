```markdown
# Adaptive HRTF Compression

This project implements an adaptive compression scheme for Head-Related Transfer Functions (HRTFs) from SOFA files. It processes impulse responses, computes frequency responses, applies adaptive Wiener filtering, and selects control points to reconstruct the magnitude response within an acceptable error range.

## Features

- **Frequency Response & Filtering:** FFT-based magnitude response computation and adaptive Wiener filtering.
- **Control Point Processing:** Detection, refinement, pruning, and merging of control points using spline interpolation and error metrics.
- **ERB-Based Analysis:** Divides the frequency range into ERB bands and evaluates spectral distortion.
- **Adaptive Compression:** Iterative error correction to compress left/right HRTF measurements.
- **Visualization:** Interactive plots and error analysis with Matplotlib.
- **CSV Export:** Saves control points and compression rates for further analysis.

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- Numba
- sofa (https://python-sofa.readthedocs.io/en/latest/)

## Usage

1. **Prepare SOFA File:**  
   Place your SOFA file (e.g., `src/sofa/minPhase_NoITD.sofa`) in the expected directory.
   (Downloaded from : https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/P0001/HRTF/HRTF/48kHz/ )
2. **Run the Script:**  
   ```bash
   python main.py
   ```

3. **Interact:**  
   Use the interactive sliders and text boxes in the Matplotlib interface to explore reconstruction, spectral distortion, and filtering errors.

4. **Output Files:**  
   CSV files with control points and compression rates will be generated.

## Configuration & Customization

- **Processing All Measurements:**  
  To process all measurements from a SOFA file, set the `max_meas` parameter to `None` in the `compress_all_hrtfs` function call within the `main()` function.  
  _Example:_
  ```python
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
      max_meas=None,  # Use all measurements
      seg_thresh=SEGMENT_THRESHOLD,
      err_bound=ERROR_BOUND
  )
  ```

- **Tuning Parameters:**  
  Adjust variables in the `main()` function to control:
  - **FFT_LEN:** FFT length for frequency response.
  - **SG_WINDOW:** Base window size for Wiener filtering.
  - **ERROR_THRESHOLD:** Error threshold for adaptive filtering.
  - **SPLINE_KIND:** Interpolation method (`'pchip'` or `'cubic'`).
  - Other thresholds and window sizes to fine-tune compression quality.
