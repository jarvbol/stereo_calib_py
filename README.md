# Stereo Camera Calibration

Two-stage calibration pipeline for stereo camera systems: intrinsic calibration (per-camera parameters) followed by extrinsic calibration (relative pose between cameras).

## Intrinsic Camera Calibration Module

File: calibration/calib_intrinsics.py
Purpose: High-precision intrinsic parameter estimation for monocular cameras using chessboard patterns.

This module performs intrinsic camera calibration:
- Camera matrix (K)    
- Distortion coeffs (D)
- Reprojection errors

---

**Overview**
This module performs intrinsic camera calibration using OpenCV's cv.calibrateCamera().
It estimates:
| Parameter | Description | Units |
| :--- | :--- | :--- |
| **Camera Matrix (K)** | Focal lengths (fx, fy) and optical center (cx, cy) | pixels |
| **Distortion Coefficients** | Radial (k₁, k₂, k₃) and tangential (p₁, p₂) distortion | dimensionless |
| **Reprojection Error** | Mean deviation between detected and projected corners | pixels |

---

**Key Features**

* Configurable via YAML file
* Adaptive outlier detection
* Optional corner refinement via linear regression
* Detailed logging and visualization
* Quality validation with warnings

---

### Preparation of the calibration dataset
- Use a printed checkerboard with known square size
- Capture 30-50 images from different angles/distances
- Ensure good lighting and minimal motion blur
- Save as PNG/JPG in a single folder

---

### Configuration File Structure

The calibration parameters are stored in a YAML file for flexible configuration without modification of the code. Complete information on all available parameters, their types, default values, and impact on the process is provided below.

#### calibration.checkerboard
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `rows` | int | Inner corners along vertical axis |
| `columns` | int | Inner corners along horizontal axis |
| `square_size` | float | Distance between adjacent corners in **`meters`** |
> Critical: rows and columns refer to inner corners, not squares. A 7×10 square board has 6×9 inner corners.

---

#### calibration.image_processing
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `resize_factor` | float | Scale factor for images (1.0 = original size) |
| `invert` | bool | Invert image colors (useful for IR/white-on-black) |
| `contrast_alpha` | float | Contrast multiplier: new = alpha * old + beta |
| `contrast_beta` | int | Brightness offset |

---

#### calibration.corner_detection

##### `criteria` sub-section:
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `term_criteria` | list[str] | OpenCV termination criteria for `cornerSubPix` |
| `max_iterations` | int | Max iterations for subpixel refinement |
| `epsilon` | float | Desired accuracy for refinement (in pixels) |

##### Top-level parameters:
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `subpix_window_size` | int | Half-size of search window for `cornerSubPix` (e.g., 3 = 7×7 window) |
| `flags` | list[str] | Primary flags for `findChessboardCorners` |
| `alt_flags` | list[str] | Fallback flags for `findChessboardCornersSB` if primary fails |
> Supported flags (prefix with cv.):
> * CALIB_CB_ADAPTIVE_THRESH — Adaptive thresholding
> * CALIB_CB_NORMALIZE_IMAGE — Histogram equalization
> * CALIB_CB_FILTER_QUADS — Remove quadrilateral distortions
> * CALIB_CB_ACCURACY — Higher accuracy mode (slower)
> * CALIB_CB_EXHAUSTIVE — Exhaustive search (slowest, most robust)

---

#### calibration.algorithm
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `calibration_flags` | list[str] | Flags responsible for the calibration algorithm |

---

#### calibration.outlier_detection
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Enable outlier filtering |
| `reprojection_error_threshold` | float | -1 = adaptive, > 0 = absolute threshold in pixels |

---

#### calibration.corner_refinement
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Apply custom linear regression refinement |
| `ransac_iterations` | int | RANSAC iterations for robust fitting |
| `ransac_threshold` | float | Inlier threshold for RANSAC (in pixels) |
> Warning: The custom refine_corners_with_linear_regression may distort corner ordering. 
> Keep enabled: false unless you've validated it for your setup.
---

#### calibration.output
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `original_corners_dir` | str | Subdirectory for images with detected corners |
| `refined_corners_dir` | str | Subdirectory for images with refined corners |
| `intrinsics_filename` | str | Filename for NumPy archive with calibration results |
| `report_filename` | str | Filename for human-readable text report |

---

#### visualization
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Show OpenCV windows during calibration |
| `delay` | int | Delay between frames in ms (for manual inspection) |
| `save_frames` | bool | Save visualization frames to disk (requires `debug.enabled`) |

---

#### debug
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Enable debug image saving |
| `path` | str | Directory for debug visualizations |

---

#### logging
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | str | Path to log file |
| `max_bytes` | int | Max log file size before rotation |
| `backup_count` | int | Number of rotated log files to keep |
| `level` | str | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

---

### Run calibration

**Settings for running the camera calibration script:**

```python
main(imgs = 'calibration_data/*',                                          #  path to the image directory
         config_path = 'config/calibration_config_intrinsics_banner.yaml', # path to the configuration file
         save_path = 'calibration_results/left/intrinsics',                # path to the directory to save the results
         visualize = False,                                                # redefining the visualization settings
         save_corner_images=True,                                          # redefining save corner setting
         invert = True                                                     # redefining the image inversion setting
    )
```
> With `--visualize=True`, OpenCV windows with detected angles are displayed during processing.

---

Stages of execution:
1. **Configuration Loading**: Parameters are read from the YAML file. CLI arguments override config values.
2. **Image Preprocessing**: Files matching the `imgs` glob pattern are loaded, sorted, and optionally resized, inverted, or contrast-adjusted.
3. **Corner Detection**: 
   - `cv.findChessboardCorners()` is called with primary flags.
   - Fallback to `cv.findChessboardCornersSB()` if detection fails.
   - Subpixel refinement via `cv.cornerSubPix()`.
   - Optional custom linear regression refinement (`corner_refinement.enabled`).
4. **Calibration**: `cv.calibrateCamera()` estimates intrinsic parameters using all successfully detected frames.
5. **Quality Assessment**: Reprojection errors are computed per frame. Outliers are flagged using adaptive (`μ + 2σ`) or absolute thresholds.
6. **Output Generation**: Camera matrix, distortion coefficients, error statistics, and quality assessment are saved to `.npz` and `.txt` reports. Visualizations are exported if requested.

---

### Output Files

**`intrinsics.npz`**

```python
import numpy as np
data = np.load('intrinsics.npz')
mtx = data['mtx']   # Camera matrix (3×3)
dist = data['dist'] # Distortion coefficients (5×1)
```

Camera matrix format:

```bash
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```
> - fx, fy: Focal lengths in pixels
> - cx, cy: Principal point (optical center) in pixels

Distortion coefficients (5-parameter model):

```bash
[k1, k2, p1, p2, k3]
```
> - k1,k2,k3: Radial distortion
> - p1,p2: Tangential distortion

**`calibration_report.txt`**

```text
Calibration Results for calibration_data/cam0/*
==================================================

Camera Matrix:
[[600.02809434   0.         305.92865377]
 [  0.         601.80288785 267.26831753]
 [  0.           0.           1.        ]]

Distortion Coefficients:
[[-0.12764954  0.81300767  0.00426253 -0.00777627 -1.96463107]]

Reprojection Error Statistics:
  RMS:  1.094793 pixels
  Mean: 0.135718 pixels
  Std:  0.061453 pixels
  Min:  0.069993 pixels
  Max:  0.309758 pixels
  Outlier threshold: 0.259 pixels [adaptive (mean + 2*std)]
  Outliers: 1/20 images
  Outlier image indices: [12]

Calibration Quality Assessment:
  ✓ Good calibration quality
```

> **Optional: Corner Visualization Images**
> If `save_corner_images=True`, two subdirectories are created inside `--save_path`:
> * `original_corners/` — Images with detected checkerboard corners overlaid
> * `refined_corners/` — Images with refined corner positions (if `corner_refinement.enabled: true`)

---

## Extrinsic Camera Calibration Module

File: calibration/calib_stereo.py
Purpose: Computation of extrinsic parameters (relative pose) for a stereo camera pair using synchronized chessboard images.

This module performs stereo (extrinsic) calibration — estimation of the relative position and orientation between two cameras:

> Input:
> * Intrinsic params (cam1, cam2)
> * Synchronized image pairs
> * YAML configuration

> Output:
> * Rotation matrix R (3×3)
> * Translation vector T (3×1)
> * Essential matrix E
> * Fundamental matrix F
> * Reprojection error metrics

---

**Key Features**
* Configurable via YAML file
* Fixed intrinsics mode (`CALIB_FIX_INTRINSIC`) for stable extrinsic estimation
* Synchronized pair processing with automatic orientation validation
* Optional corner refinement via linear regression
* Multi-method corner detection with sub-pixel accuracy
* Detailed logging, visualization, and debug export
* Quality assessment with RMS reprojection error reporting
> Prerequisite: Stereo calibration requires pre-computed intrinsic parameters for both cameras. Run calib_intrinsics.py first and ensure:
> 1. Same square_size value in both configurations
> 2. Same image preprocessing parameters
> 3. High-quality intrinsics (mean reprojection error < 0.3 px)

---

### Prerequisites

* Completed intrinsic calibration for both cameras (intrinsics.npz files)
* 20-40 synchronized image pairs with checkerboard visible in both views
* Images captured from varied poses (different distances, angles, orientations)
* Consistent preprocessing (same resolution, inversion, contrast settings)

---

### Configuration File Structure

The calibration parameters are stored in a YAML file for flexible configuration without modification of the code. Complete information on all available parameters, their types, default values, and impact on the process is provided below.

#### calibration.checkerboard
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `rows` | int | Inner corners along vertical axis |
| `columns` | int | Inner corners along horizontal axis |
| `square_size` | float | Distance between adjacent corners in **`meters`** |
> Critical: rows and columns refer to inner corners, not squares. A 7×10 square board has 6×9 inner corners.

---

#### calibration.image_processing
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `resize_factor` | float | Scale factor for images (1.0 = original size) |
| `invert` | bool | Invert image colors (useful for IR/white-on-black) |
| `contrast_alpha` | float | Contrast multiplier: new = alpha * old + beta |
| `contrast_beta` | int | Brightness offset |

---

#### calibration.corner_detection

##### `criteria` sub-section:
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `term_criteria` | list[str] | OpenCV termination criteria for `cornerSubPix` |
| `max_iterations` | int | Max iterations for subpixel refinement |
| `epsilon` | float | Desired accuracy for refinement (in pixels) |

##### Top-level parameters:
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `subpix_window_size` | int | Half-size of search window for `cornerSubPix` (e.g., 3 = 7×7 window) |
| `flags` | list[str] | Primary flags for `findChessboardCorners` |
| `alt_flags` | list[str] | Fallback flags for `findChessboardCornersSB` if primary fails |
> Supported flags (prefix with cv.):
> * CALIB_CB_ADAPTIVE_THRESH — Adaptive thresholding
> * CALIB_CB_NORMALIZE_IMAGE — Histogram equalization
> * CALIB_CB_FILTER_QUADS — Remove quadrilateral distortions
> * CALIB_CB_ACCURACY — Higher accuracy mode (slower)
> * CALIB_CB_EXHAUSTIVE — Exhaustive search (slowest, most robust)

---

#### calibration.algorithm
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `calibration_flags` | list[str] | Flags responsible for the calibration algorithm |

---

#### calibration.outlier_detection
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Enable outlier filtering |
| `reprojection_error_threshold` | float | -1 = adaptive, > 0 = absolute threshold in pixels |

---

#### calibration.corner_refinement
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Apply custom linear regression refinement to detected corners |
| `ransac_iterations` | int | RANSAC iterations for robust line fitting (if used in refinement) |
| `ransac_threshold` | float | Inlier threshold for RANSAC in pixels |
> Warning: Custom refinement may alter corner ordering between cameras.
> Keep `enabled: false` unless you've validated orientation consistency for your stereo setup.

---

#### calibration.save_corners
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `save_corners` | bool | If `true`, saves detected and refined corner coordinates for both cameras to `corners_dir` |
> Saved files contain: `cam1_original_corners`, `cam2_original_corners`,
> `cam1_refined_corners`, `cam2_refined_corners`, `object_points`

---

#### calibration.output
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `stereo_filename` | str | Filename for NumPy archive with stereo calibration results (`R`, `T`, `E`, `F`) |
| `report_filename` | str | Filename for human-readable text report |
| `corners_dir` | str | Subdirectory for saving detected corner coordinates (`.npz` per pair) |
| `debug_cam1_dir` | str | Subdirectory for Camera 1 corner visualization images |
| `debug_cam2_dir` | str | Subdirectory for Camera 2 corner visualization images |
| `debug_cam1_refined_dir` | str | Subdirectory for Camera 1 refined corner visualizations |
| `debug_cam2_refined_dir` | str | Subdirectory for Camera 2 refined corner visualizations |

---

#### visualization
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Show OpenCV windows during calibration |
| `delay` | int | Delay between frames in ms (for manual inspection) |
| `save_frames` | bool | Save visualization frames to `debug_*_dir` subdirectories |
> Stereo-specific: Four windows are shown when enabled: 
> Camera 1/2 original corners + Camera 1/2 refined corners. Press ESC to stop.

---

#### debug
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | bool | Enable saving of debug visualizations and corner coordinate exports |
| `path` | str | Base directory for debug output (visualizations + text diagnostics) |
> When enabled, creates:
> * `debug_cam1_dir/`, `debug_cam2_dir/` — original corner overlays
> * `debug_cam1_refined_dir/`, `debug_cam2_refined_dir/` — refined corner overlays
> * `debug_stereo/pair_XXXX.txt` — detailed corner coordinate dumps with orientation checks

---

#### logging
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | str | Path to log file |
| `max_bytes` | int | Max log file size before rotation |
| `backup_count` | int | Number of rotated log files to keep |
| `level` | str | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
> Additional log: Corner orientation diagnostics are written to `logs/corner_orientation_check.log` for pair-by-pair validation.

---

### Run calibration

**Settings for running the camera calibration script:**

```python
main(
        cam1_intrinsics_path = 'cam1/intrinsics/intrinsics.npz',  # path to the camera's intrinsics 1
        cam2_intrinsics_path = 'cam2/intrinsics/intrinsics.npz',  # path to the camera's intrinsics 2
        cam1_imgs = 'calibration_data/cam1/*.png',                # glob-image pattern of camera 1
        cam2_imgs = 'calibration_data/cam2/*.png',                # glob-image pattern of camera 2
        config_path = 'config/calibration_config_stereo.yaml',    # path to the configuration file
        save_path = 'calibration_results/stereo',                 # path to the directory to save the results
        visualize = False,                                        # redefining the visualization setting
        invert = True                                             # redefining the image inversion setting
    )
```
> Important: The images in `cam1_imgs` and `cam2_imgs` must be synchronized — the files are sorted by name,
> and a pair with the same index is considered to have been taken at the same time.

> With `--visualize=True`, OpenCV windows with detected angles for both cameras are displayed during processing.
> To continue processing, you need to press the keys, and to exit, press ESC.

---

**Stages of Execution**

1. **Configuration Loading**: Parameters are read from the YAML file. CLI arguments (`--visualize`, `--invert`) override config values.
2. **Intrinsics Loading**: `mtx` and `dist` for both cameras are loaded from `.npz` files. Consistency with stereo config is logged.
3. **Image Loading & Preprocessing**: 
   * Files matching `cam1_imgs` and `cam2_imgs` glob patterns are loaded and sorted by filename.
   * Each image is preprocessed: resized, inverted, contrast-adjusted per config.
   * Pair count mismatch triggers a warning.
4. **Paired Corner Detection & Orientation Validation**:
   * `cv.findChessboardCorners()` with primary flags for each camera.
   * Fallback to `cv.findChessboardCornersSB()` if primary detection fails.
   * Sub-pixel refinement via `cv.cornerSubPix()`.
   * **Orientation check**: `corner_orientation()` computes cosine similarity of edge vectors. Pairs with `cos_mean ≤ -0.9` (flipped geometry) are automatically skipped.
   * Debug coordinates exported to `debug_stereo/pair_XXXX.txt` if `debug.enabled: true`.
   * Pair is accepted only if detection succeeds in **both** cameras.
5. **Stereo Calibration (OpenCV)**: 
   * `cv.stereoCalibrate()` is called with `flags=cv.CALIB_FIX_INTRINSIC`.
   * Intrinsics (`mtx`, `dist`) are fixed; only `R` (rotation) and `T` (translation) are optimized.
   * Essential (`E`) and Fundamental (`F`) matrices are computed internally.
6. **Post-Processing & Logging**:
   * Rotation matrix `R` is converted to Euler angles (XYZ convention) via `scipy.spatial.transform.Rotation`.
   * RMS reprojection error, camera matrices, distortion coefficients, `R`, `T`, and Euler angles are logged.
7. **Results Saving** (if `save_path` specified):
   * `stereo.npz`: `R`, `T`, `euler_angles_rad`, `euler_angles_deg`.
   * `stereo_calibration_report.txt`: Human-readable summary with RMS, rotation, translation.
   * Corner coordinates (`.npz` per pair) to `corner_coordinates/` if `save_corners: true`.
   * Visualization images to `debug_*_dir/` subdirectories if `debug.enabled: true`.

---

### Output Files

**`stereo.npz`**

```python
import numpy as np
data = np.load('stereo.npz')

R = data['R']                          # Rotation matrix (3×3)
T = data['T']                          # Translation vector (3×1)
euler_rad = data['euler_angles_rad']   # Euler angles in radians (3,)
euler_deg = data['euler_angles_deg']   # Euler angles in degrees (3,)
```

Rotation matrix format (camera 2 → camera 1):

```bash
[[ r11, r12, r13 ],
 [ r21, r22, r23 ],
 [ r31, r32, r33 ]]
```
> * Describes rotation of camera 2 coordinate system relative to camera 1
> * Should be orthogonal: R @ R.T ≈ I, det(R) ≈ +1


Translation vector format:

```bash
[[ Tx ],
 [ Ty ],
 [ Tz ]]
```
> * Units: Same as square_size (typically meters)
> * Interpretation: Center of camera 2 is located at position T in camera 1's coordinate system
> * Baseline distance: np.linalg.norm(T)

Euler angles (XYZ convention): 

```python
[euler_x, euler_y, euler_z]  # scipy: Rotation.as_euler('xyz', extrinsic=True)
```

**`calibration_report.txt`**

```text
Stereo Calibration Results
==================================================

Camera 1: cam1/intrinsics/intrinsics.npz
Camera 2: cam2/intrinsics/intrinsics.npz

Calibration Quality:
  RMS Reprojection Error: 0.553559 pixels

Rotation Matrix:
[[ 9.99984389e-01 -1.73328923e-05 -5.58765929e-03]
 [ 4.67861529e-06  9.99997436e-01 -2.26468811e-03]
 [ 5.58768422e-03  2.26462661e-03  9.99981824e-01]]

Euler Angles (XYZ convention):
  Radians: [0.002265, -0.005588, 0.000005]
  Degrees: [0.129756, -0.320152, 0.000268]

Translation Vector:
[[-0.05266758]
 [ 0.00171367]
 [-0.02287942]]
```

**`corner_coordinates/` (if `save_corners=true`)**

* NumPy archives with detected and refined corner positions
* Filename pattern: corners_XXXX.npz where XXXX is pair index
* Contains: cam1_original_corners, cam2_original_corners, cam1_refined_corners, cam2_refined_corners, object_points

**`debug_visualizations/` (if `debug.enabled=true`)**

* PNG images showing detected corners overlaid on original frames
* Separate subdirectories for each camera and refinement stage
