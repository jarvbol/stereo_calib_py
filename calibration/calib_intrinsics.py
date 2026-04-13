import os
import sys
import cv2 as cv
import glob
import logging
import numpy as np

from fire import Fire
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from calibration.utils.config_utils import load_config, setup_logging, get_calibration_flags
from calibration.utils.corner_refinement import refine_corners_with_linear_regression
from calibration.utils.corner_detection import (
    detect_checkerboard_corners,
    create_visualization_frame,
    create_refined_corners_visualization,
    preprocess_calibration_image,
)

from logging.handlers import RotatingFileHandler

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
        handlers=[RotatingFileHandler('logs/log_intrinsics_camera_1.log', maxBytes=100000, backupCount=10),
                  logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')

logger = logging.getLogger(__name__)

def calibrate_camera_intrinsics(
    images_folder: str, 
    config_path: str = "config/calibration_config.yaml", 
    visualize: bool = None,
    invert: bool = None,
    save_path: str = None,
    save_corner_images: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the intrinsic calibration routine for a single camera
    and returns the results.

    Parameters
    ----------
    images_folder : str
        Path to the folder containing the checkerboard images.
        The path should be given as a glob pattern.
    config_path : str, optional
        Path to the configuration file.
    visualize : bool, optional
        If True, the images with the detected checkerboard corners
        will be shown during the calibration process. Overrides config if provided.
    invert : bool, optional
        If True, inverts the images before processing. Overrides config if provided.
    save_corner_images : bool, optional
        If True, saves all images with detected corners to the save_path directory.
        Requires save_path to be provided.

    Returns
    -------
    mtx : np.ndarray
        Camera matrix.
    dist : np.ndarray
        Distortion coefficients.
    errors : np.ndarray
        Reprojection errors for each calibration image.
    outlier_indices : np.ndarray
        Indices of outlier images with high reprojection errors.
    outlier_threshold : float
        Applied outlier detection threshold in pixels. 
        - If adaptive mode: threshold = mean_error + 2×std_error
        - If absolute mode: threshold = configured value (e.g., 0.5 px)
    threshold_method : str
        Detection method.
        - "adaptive (mean + 2*std)" — dynamic threshold based on error distribution
        - "absolute (X.XXX px)" — fixed threshold from configuration
        - "disabled (invalid config value)" — outlier detection disabled

    """
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    # Override config with command-line parameters if provided
    if visualize is not None:
        config['visualization']['enabled'] = visualize
    if invert is not None:
        config['calibration']['image_processing']['invert'] = invert
    
    # Extract parameters from config
    calib_config = config['calibration']
    rows = calib_config['checkerboard']['rows']
    columns = calib_config['checkerboard']['columns']
    world_scaling = calib_config['checkerboard']['square_size']
    
    img_proc_config = calib_config['image_processing']
    resize_factor = img_proc_config['resize_factor']
    invert_images = img_proc_config['invert']
    contrast_alpha = img_proc_config['contrast_alpha']
    contrast_beta = img_proc_config['contrast_beta']
    
    corner_config = calib_config['corner_detection']
    criteria_max_iter = corner_config['criteria']['max_iterations']
    criteria_epsilon = corner_config['criteria']['epsilon']
    subpix_window_size = corner_config['subpix_window_size']
    flags = get_calibration_flags(corner_config['flags'])
    alt_flags = get_calibration_flags(corner_config['alt_flags'])
    
    
    output_config = calib_config.get('output', {})
    orig_dir = output_config.get('original_corners_dir', 'original_corners')
    refined_dir = output_config.get('refined_corners_dir', 'refined_corners')
    
    
    # Reading the emission threshold from the config
    outlier_config = calib_config.get('outlier_detection', {})
    outlier_threshold_config = outlier_config.get('reprojection_error_threshold', -1)

    
    algo_config = calib_config['algorithm']
    calib_flags = get_calibration_flags(algo_config.get('calibration_flags', []))
    
    vis_config = config['visualization']
    visualize_enabled = vis_config['enabled']
    visualize_delay = vis_config['delay']

    # Create directories for saving corner images if requested
    if save_corner_images and save_path is not None:
        vis_dir = Path(save_path, orig_dir)
        refined_vis_dir = Path(save_path, refined_dir)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(refined_vis_dir, exist_ok=True)
    else:
        vis_dir = None
        refined_vis_dir = None
    
    # Load and preprocess images
    images_names = sorted(glob.glob(images_folder))
    if not images_names:
        logger.error(f"No images found at {images_folder}")
        raise FileNotFoundError(f"No images found at {images_folder}")
    
    logger.info(f"Found {len(images_names)} images for calibration")
    
    images = []
    for imname in images_names:
        im = cv.imread(imname)
        if im is None:
            logger.warning(f"Failed to read image: {imname}")
            continue
            
        # Preprocess the image
        im = preprocess_calibration_image(
            im, resize_factor, invert_images, contrast_alpha, contrast_beta
        )
        
        images.append(im)
    
    if not images:
        logger.error("No valid images found for calibration")
        raise ValueError("No valid images found for calibration")
    

    
    # Set up calibration criteria
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 
        criteria_max_iter, 
        criteria_epsilon
    )

    # Prepare object points
    # objp = np.zeros((rows * columns, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    # objp = world_scaling * objp
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)  # ← Поменяли местами!
    objp = world_scaling * objp

    # Get frame dimensions
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Initialize arrays for detected points
    imgpoints = []  # 2D points in image plane
    objpoints = []  # 3D points in world space
    refined_imgpoints = [] # 2D points in image plane
    # Detect chessboard corners
    total_frames = len(images)
    

    successful_detections = 0
    failed_detections = 0
    corner_counts = []

    
    for i, (frame, image_path) in enumerate(zip(images, images_names)):
        # Detect corners
        ret, corners = detect_checkerboard_corners(
            frame, rows, columns, subpix_window_size, criteria, flags, alt_flags
        )
        
        if ret:
            successful_detections += 1
            num_corners = len(corners)
            corner_counts.append(num_corners)
            

            logger.info(f"Image {i}: {num_corners} corners detected")
            if num_corners != rows * columns:
                logger.warning(f"  ⚠ Expected {rows*columns} corners, got {num_corners}")
            if i == 0:  
                logger.info(f"  First corner: {corners[0, 0]}")
                logger.info(f"  Last corner: {corners[-1, 0]}")
                logger.info(f"  Corner at [0, {columns-1}]: {corners[columns-1, 0]}")
                logger.info(f"  Corner at [{rows-1}, 0]: {corners[(rows-1)*columns, 0]}")

            
            # Create visualization with original corners
            vis_frame = create_visualization_frame(frame, corners, rows, columns, ret)
            
            # Add points to calibration lists
            objpoints.append(objp)
            imgpoints.append(corners)
            

            corner_refinement_config = calib_config.get('corner_refinement', {})
            refinement_enabled = corner_refinement_config.get('enabled', True)
            
            if refinement_enabled:
                logger.info(f"Refining corners for image {i}")
                refined_corners = refine_corners_with_linear_regression(corners, rows, columns)
                
                if i == 0:
                    diff = np.abs(refined_corners - corners)
                    logger.info(f"  Original first: {corners[0, 0]}")
                    logger.info(f"  Refined first: {refined_corners[0, 0]}")
                    logger.info(f"  Max diff: {np.max(diff):.2f} px")
                    if np.max(diff) > 50:
                        logger.error("  Refinement changed corners too much!")
                
                refined_imgpoints.append(refined_corners)
            else:
                logger.info(f"Corner refinement disabled - using original corners for image {i}")
                refined_imgpoints.append(corners)  


            refined_vis_frame = create_refined_corners_visualization(
                    frame, corners, refined_corners if refinement_enabled else corners
                )
            
            # Save corner images if requested
            if save_corner_images and vis_dir is not None and refined_vis_dir is not None:
                # Extract original filename without extension
                original_filename = Path(image_path).stem
                original_ext = Path(image_path).suffix
                
                # Save with original filename
                cv.imwrite(str(Path(vis_dir, f'{original_filename}_original_corners{original_ext}')), vis_frame)
                cv.imwrite(str(Path(refined_vis_dir, f'{original_filename}_refined_corners{original_ext}')), refined_vis_frame)

            # Display if visualization is enabled
            if visualize_enabled:
                cv.imshow('Original Corners - Press ESC to quit', vis_frame)
                key = cv.waitKey(visualize_delay)
                if key == 27:  # ESC key
                    logger.info("Visualization interrupted by user")
                    cv.destroyAllWindows()
                    visualize_enabled = False
            
            logger.info(f"Successfully detected corners in image {i}")
        else:
            failed_detections += 1
            logger.warning(f"Failed to detect corners in image {i}")
            if visualize_enabled:
                cv.imshow('Failed detection - Press ESC to quit', frame)
                key = cv.waitKey(visualize_delay)
                if key == 27:  # ESC key
                    logger.info("Visualization interrupted by user")
                    cv.destroyAllWindows()
                    visualize_enabled = False


    logger.info("=" * 60)
    logger.info("DETECTION STATISTICS:")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Successful detections: {successful_detections}")
    logger.info(f"Failed detections: {failed_detections}")
    logger.info(f"Frames with detected corners: {len(imgpoints)}")
    if corner_counts:
        logger.info(f"Corner counts - Min: {min(corner_counts)}, Max: {max(corner_counts)}, Avg: {np.mean(corner_counts):.1f}")
    logger.info("=" * 60)


    if not imgpoints:
        logger.error("No chessboard corners detected in any images")
        raise ValueError("No chessboard corners detected in any images")
   
    # Run calibration with refined points
    logger.info(f"Running calibration with {len(refined_imgpoints)} images")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, refined_imgpoints, (width, height), None, None, flags=calib_flags
    )
    

    logger.info("=" * 60)
    logger.info("CAMERA MATRIX VALIDATION:")
    fx, fy = mtx[0, 0], mtx[1, 1]
    expected_fx_min = width * 0.5  
    expected_fy_min = height * 0.5 
    
    logger.info(f"fx = {fx:.2f}, fy = {fy:.2f}")
    logger.info(f"Expected fx range: {expected_fx_min:.0f} - {width*1.5:.0f}")
    logger.info(f"Expected fy range: {expected_fy_min:.0f} - {height*1.5:.0f}")
    
    if fx < expected_fx_min or fy < expected_fy_min:
        logger.error("CRITICAL: Focal lengths are too small!")
        logger.error("Possible causes:")
        logger.error("  1. Wrong square_size in config")
        logger.error("  2. Wrong checkerboard dimensions (rows/columns)")
        logger.error("  3. Incorrect corner detection")
        logger.error("  4. Images are being resized incorrectly")
    else:
        logger.info("Focal lengths are in expected range")
    logger.info("=" * 60)

    
    # Calculate and report reprojection error
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(refined_imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        errors.append(error)
    
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    
    # Check for outliers
    # outlier_threshold = mean_error + 2 * std_error
    
    if outlier_threshold_config == -1:
        outlier_threshold = mean_error + 2 * std_error
        threshold_method = "adaptive (mean + 2*std)"
        logger.info(f"Outlier detection: using {threshold_method}")
    elif outlier_threshold_config > 0:
        outlier_threshold = float(outlier_threshold_config)
        threshold_method = f"absolute ({outlier_threshold:.3f} px)"
        logger.info(f"Outlier detection: using {threshold_method}")
    else:
        outlier_threshold = float('inf')
        threshold_method = "disabled (invalid config value)"
        logger.warning(f"Outlier detection: {threshold_method}")
    
    outliers = np.sum(errors > outlier_threshold)
    outlier_indices = np.where(errors > outlier_threshold)[0]
    
    logger.info(f"Reprojection error statistics:")
    logger.info(f"  Mean: {mean_error:.3f} pixels")
    logger.info(f"  Std:  {std_error:.3f} pixels")
    logger.info(f"  Min:  {min_error:.3f} pixels")
    logger.info(f"  Max:  {max_error:.3f} pixels")
    logger.info(f"  Outliers: {outliers}/{len(errors)} images (threshold: {outlier_threshold:.3f} pixels)")
    
    if outliers > 0:
        logger.warning(f"Outlier images detected: {outlier_indices.tolist()}")
        logger.warning("Consider removing these images and re-calibrating for better results")
    
    if mean_error > 1.0:
        logger.warning("⚠ High mean reprojection error - calibration quality may be poor")
    elif outliers > len(errors) * 0.1:
        logger.warning("⚠ Many outliers detected - inconsistent calibration data")
    else:
        logger.info("✓ Good calibration quality")
    
    logger.info(f"INTRINSIC CALIBRATION RESULTS FOR {images_folder}:")
    logger.info(f'RMS reprojection error: {ret}')
    logger.info(f'Mean reprojection error: {mean_error}')
    logger.info(f'Camera matrix:\n{mtx}')
    logger.info(f'Distortion coefficients:\n{dist}')
    
    if visualize_enabled:
        cv.destroyAllWindows()
    
    return mtx, dist, errors, outlier_indices, outlier_threshold, threshold_method, ret

def main(
    imgs: str,
    config_path: str = "config/calibration_config.yaml",
    save_path: str = None,
    visualize: bool = None,
    invert: bool = None,
    save_corner_images: bool = False,
) -> None:
    """
    Runs the intrinsic calibration routine for a single camera
    and saves the results to a file.

    Parameters
    ----------
    imgs : str
        Path to the folder containing the checkerboard images.
        The path should be given as a glob pattern.
    config_path : str, optional
        Path to the configuration file.
    save_path : str, optional
        Path to save the intrinsic calibration results.
        If not given, the results will not be saved.
    visualize : bool, optional
        If True, the images with the detected checkerboard corners
        will be shown during the calibration process. Overrides config if provided.
    invert : bool, optional
        If True, inverts the images before processing. Overrides config if provided.
    save_corner_images : bool, optional
        If True, saves all images with detected corners to the save_path directory.
        Requires save_path to be provided.
    """
    # Load configuration for logging and visualization parameters
    config = load_config(config_path)
    logger = setup_logging(config)
    
    ###
    calib_config = config.get('calibration', {})
    output_config = calib_config.get('output', {})
    
    intrinsics_name = output_config.get('intrinsics_filename', 'intrinsics.npz')
    report_name = output_config.get('report_filename', 'calibration_report.txt')
    ###
    
    
    # Override config with command-line parameters if provided
    if visualize is not None:
        config['visualization']['enabled'] = visualize
    if invert is not None:
        config['calibration']['image_processing']['invert'] = invert
    
    # Save results if a save path is provided
    if save_path is not None:
        # Create the directory if it doesn't exist
        os.makedirs(Path(save_path), exist_ok=True)
    # Run calibration
    try:
        # added outlier_threshold, threshold_method
        mtx, dist, errors, outlier_indices, outlier_threshold, threshold_method, rms_error = calibrate_camera_intrinsics(
            imgs, config_path, visualize, invert, save_path, save_corner_images
        )
        
        # Calculate error statistics for report
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        outliers = len(outlier_indices)
        # outlier_threshold = mean_error + 2 * std_error
        
        # Save results if a save path is provided
        if save_path is not None:
            
        
            # Save calibration results
            calib_file = Path(save_path, intrinsics_name)
            # calib_file = Path(save_path, 'intrinsics.npz')
            np.savez(calib_file, mtx=mtx, dist=dist)
            logger.info(f"Saved calibration results to {calib_file}")
            
            # Save readable text report
            report_file = Path(save_path, report_name)
            # report_file = Path(save_path, 'calibration_report.txt')
            with open(report_file, 'w') as f:
                f.write(f"Calibration Results for {imgs}\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Camera Matrix:\n{mtx}\n\n")
                f.write(f"Distortion Coefficients:\n{dist}\n\n")
                f.write(f"Reprojection Error Statistics:\n")
                f.write(f"  RMS:  {rms_error:.6f} pixels\n")
                f.write(f"  Mean: {mean_error:.6f} pixels\n")
                f.write(f"  Std:  {std_error:.6f} pixels\n")
                f.write(f"  Min:  {min_error:.6f} pixels\n")
                f.write(f"  Max:  {max_error:.6f} pixels\n")
                
                f.write(f"  Outlier threshold: {outlier_threshold:.3f} pixels [{threshold_method}]\n")
                f.write(f"  Outliers: {outliers}/{len(errors)} images\n")
                if outliers > 0:
                    f.write(f"  Outlier image indices: {outlier_indices.tolist()}\n")
                
                    
                    
                f.write(f"\nCalibration Quality Assessment:\n")
                if mean_error > 1.0:
                    f.write(f"  High mean reprojection error - calibration quality may be poor\n")
                elif outliers > len(errors) * 0.1:
                    f.write(f" Many outliers detected - inconsistent calibration data\n")
                else:
                    f.write(f" Good calibration quality\n")
            logger.info(f"Saved calibration report to {report_file}")
            
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise

if __name__ == "__main__":
    # Fire(main)
    main(imgs = '/Users/cruelangel/Desktop/JARV/Ichthyander/calibration_data0/ir_left/*',
         config_path = 'config/calibration_config_intrinsics_banner.yaml',
         save_path = 'calibration_results/left/intrinsics',
         visualize = False,
         save_corner_images=True,
         invert = True)