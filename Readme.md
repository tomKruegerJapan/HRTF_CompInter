# HRTF Compression Toolkit

A Python code for compressing Head-Related Transfer Functions (HRTFs) using adaptive control point extraction, selection, pruning and filtering. 

## What this does

This code compresses HRTF data from SOFA files by finding the most important points (control points). Also plotting and verifying the results.

### Main features:
- **Wiener filtering**: Smooth the original data and extract the first set of control points from it
- **Adaptive compression**: Automatically finds the best points to keep by segmentation and pruning
- **Multiple smoothing methods**: Wiener filtering, octave smoothing, cepstral
- **Lots of analysis**: Error plots, boxplots, statistics, dynamic plot
- **Parallel processing**: Uses parallel processing

## Installation

You need these packages:
```bash
pip install numpy matplotlib scipy sofa joblib tqdm
```

Make sure you have Python 3.11+ (probably works on older versions but haven't tested).

## How to use

### Step 1: Compress your HRTF data
```bash
python main.py
```

Put your SOFA files in `src/sofa/` folder first. The script will:
- Load all the SOFA files it finds
- Compress each HRTF measurement 
- Save results as pickle files (for now) in `src/sofa/FinalResultsPKL/`
- Print some stats along the way

### Step 2: Look at your results
```bash
python plot_boxplot.py
```

This gives you:
- Interactive plots where you can browse through measurements
- Boxplots showing error distributions
- Filtering error analysis
- A lot of other useful plots

## File structure
```
├── main.py              # The main compression code
├── plot_boxplot.py      # All the plotting and analysis stuff
├── src/
│   └── sofa/           # Put your SOFA files here
│       ├── testing/    # PKL files get saved here (main.py output)
│       └── FinalResultsPKL/  # Where plot_boxplot.py looks for PKL files 
```

## Important parameters you might want to change

In `main.py` around line 720:

```python
FFT_LEN = 1024              # FFT size (bigger = more frequency resolution)
THRESHOLD_SD = 0.95         # Error threshold (lower = better quality, more points)
SG_WINDOW = 11              # Smoothing window size
SMOOTH_METHOD = 'wiener'    # Smoothing method ('wiener', 'erb', 'octave', 'cepstral')
NUM_BANDS = 42             # Number of ERB bands for analysis
INIT_POINTS = 10           # Initial control point count
MAX_POINTS = 100           # Maximum allowed control points
SPLINE_KIND = 'pchip'      # Interpolation method ('pchip' or 'cubic')
ERROR_BOUND = 0.95         # Error bound for control point addition
n_jobs = 50                # Number of parallel workers (line ~780)
max_meas = 1               # Max measurements per file (None = all)
```

### Where outputs are stored:
```python
compressed_db_path = "src/sofa/testing"     # PKL files saved here
```

In `plot_boxplot.py`:
```python
PKL_DIR = "src/sofa/FinalResultsPKL"        # Where it looks for PKL files
```

## What the algorithms do

### Basic pipeline:
1. **Load HRTF data** from SOFA files
2. **Convert to frequency domain** using FFT
3. **Smooth the data** to remove noise/artifacts  
4. **Find control points** using derivative analysis (finds peaks, valleys, inflection points)
5. **Segementation and pruning** Removes as many Points as possible by staying under the threshold
6. **Optimize control points** using dynamic programming and grid search (need fix right now)
7. **Validate quality** - make sure error is below threshold
8. **Save results** as compressed representation

### Control point selection:
The code finds important frequencies by looking at:
- Zeros of first derivative (peaks and valleys)
- Inflection points from second derivative  
- Points where reconstruction error is high
- Perceptually important regions (using ERB scale)

### Quality control:
- Keeps spectral distortion below configurable threshold
- Uses ERB-bands (matches human hearing)
- Iteratively adds points where error is too high
- Multiple optimization strategies to minimize control points (remove)

## Interactive features

`plot_boxplot.py` has interactive plot:

- **Slider**: Browse through all measurements 
- **Text boxes**: Type measurement numbers to see specific plots
- **Boxplots**: See error distribution across ERB frequency bands
- **Error analysis**: Detailed reconstruction error plots

Just run it and play around with the controls

## Configuration 

### Input data requirements:
- Standard SOFA format files (No ITD, Min_Phase)
- Left and right ear data
- Any sampling rate (code handles resampling)
- Any number of measurement positions

### Output format:
Results saved as pickle files with this structure:
```python
{
    'frequencies': np.array,        # Frequency grid (Hz)
    'left_results': list,          # Left ear compression results  
    'right_results': list,         # Right ear compression results
    'positions': np.array          # [azimuth, elevation] in degrees
}
```

# Each result contains:
- Original and smoothed magnitude spectra
- Control points [frequency, magnitude] 
- Reconstructed spectrum
- Error statistics
- Compression metadata

## If something breaks

Common issues:
- **"No SOFA files found"**: Put .sofa files in `src/sofa/` directory
- **Memory errors**: Reduce `max_meas` to reduce time
- **Weird plots**: Make sure you have matplotlib correctly installed
- **Slow processing**: Reduce number of files or measurements

## Future improvements (TODO)

- [ ] Refactoring and cleaning the code more
- [ ] More parallel functionality for the functions
- [ ] Optimization of functions
- [ ] Merging function fix
- [ ] Getting rid of redundancies

## Credits

Code written for research project on HRTF compression. Uses Python libraries and includes knowledge from various papers and people who helped me.

**Note**: This is research code so it might have bugs or weird behaviors. Test it on your own data before using for anything important!