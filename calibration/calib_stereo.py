import os
import sys
import cv2 as cv
import glob
import logging
import numpy as np

from fire import Fire
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from scipy.spatial.transform import Rotation

from calibration.utils.config_utils import load_config, setup_logging, get_calibration_flags
from calibration.utils.corner_refinement import refine_corners_with_linear_regression
from calibration.utils.corner_detection import (
    detect_checkerboard_corners,
    create_visualization_frame,
    create_refined_corners_visualization,
    preprocess_calibration_image,
)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
        handlers=[RotatingFileHandler('logs/log_stereo.log', maxBytes=100000, backupCount=10),
                  logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')

logger = logging.getLogger(__name__)

orientation_logger = logging.getLogger('corner_orientation')
orientation_handler = logging.FileHandler('logs/corner_orientation_check.log', mode='w')
orientation_handler.setLevel(logging.DEBUG)
orientation_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S'
))
orientation_logger.addHandler(orientation_handler)
orientation_logger.propagate = False 

def write_corner_debug_txt(
    filepath: Path,
    pair_idx: int,
    path1: str,
    path2: str,
    corners1: np.ndarray,
    corners2: np.ndarray,
    rows: int,
    cols: int
) -> None:

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Stereo Calibration Debug — Pair {pair_idx:04d}\n")
        f.write(f"camera1_image: {Path(path1).name}\n")
        f.write(f"camera2_image: {Path(path2).name}\n")
        f.write(f"board_size: {rows}x{cols} corners\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"## CAMERA 1 CORNERS (shape: {corners1.shape})\n")
        f.write(f"{'idx':>4} | {'x':>10} | {'y':>10} | {'row':>3} | {'col':>3}\n")
        f.write(f"{'-'*40}\n")
        for idx, (x, y) in enumerate(corners1.reshape(-1, 2)):
            row, col = divmod(idx, cols)
            f.write(f"{idx:4d} | {x:10.2f} | {y:10.2f} | {row:3d} | {col:3d}\n")
        f.write("\n")
        
        f.write(f"## CAMERA 2 CORNERS (shape: {corners2.shape})\n")
        f.write(f"{'idx':>4} | {'x':>10} | {'y':>10} | {'row':>3} | {'col':>3}\n")
        f.write(f"{'-'*40}\n")
        for idx, (x, y) in enumerate(corners2.reshape(-1, 2)):
            row, col = divmod(idx, cols)
            f.write(f"{idx:4d} | {x:10.2f} | {y:10.2f} | {row:3d} | {col:3d}\n")
        f.write("\n")
        
        c1_tl = corners1[0, 0]
        c1_tr = corners1[cols-1, 0]
        c1_bl = corners1[cols*(rows-1), 0]
        c1_br = corners1[-1, 0]
        
        c2_tl = corners2[0, 0]
        c2_tr = corners2[cols-1, 0]
        c2_bl = corners2[cols*(rows-1), 0]
        c2_br = corners2[-1, 0]
        
        f.write("## KEY CORNERS COMPARISON\n")
        f.write(f"{'Corner':<10} | {'Cam1 (x,y)':<20} | {'Cam2 (x,y)':<20} | {'Distance (px)':<12}\n")
        f.write(f"{'-'*70}\n")
        for name, c1, c2 in [
            ('TL (0,0)', c1_tl, c2_tl),
            ('TR (0,W)', c1_tr, c2_tr),
            ('BL (H,0)', c1_bl, c2_bl),
            ('BR (H,W)', c1_br, c2_br),
        ]:
            dist = np.linalg.norm(c1 - c2)
            f.write(f"{name:<10} | [{c1[0]:7.2f}, {c1[1]:7.2f}] | [{c2[0]:7.2f}, {c2[1]:7.2f}] | {dist:11.2f}\n")
        f.write("\n")
        
        vec1_top = c1_tr - c1_tl
        vec1_left = c1_bl - c1_tl
        vec2_top = c2_tr - c2_tl
        vec2_left = c2_bl - c2_tl
        
        f.write("## VECTORS (for orientation check)\n")
        f.write(f"Camera 1:\n")
        f.write(f"  top_vec  (TR - TL) : [{vec1_top[0]:8.3f}, {vec1_top[1]:8.3f}] | norm={np.linalg.norm(vec1_top):7.2f}\n")
        f.write(f"  left_vec (BL - TL) : [{vec1_left[0]:8.3f}, {vec1_left[1]:8.3f}] | norm={np.linalg.norm(vec1_left):7.2f}\n")
        f.write(f"Camera 2:\n")
        f.write(f"  top_vec  (TR - TL) : [{vec2_top[0]:8.3f}, {vec2_top[1]:8.3f}] | norm={np.linalg.norm(vec2_top):7.2f}\n")
        f.write(f"  left_vec (BL - TL) : [{vec2_left[0]:8.3f}, {vec2_left[1]:8.3f}] | norm={np.linalg.norm(vec2_left):7.2f}\n")
        f.write("\n")
        
        def safe_cos(v1, v2):
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            return np.dot(v1, v2) / (n1 * n2)
        
        cos_top = safe_cos(vec1_top, vec2_top)
        cos_left = safe_cos(vec1_left, vec2_left)
        
        f.write("## COSINE SIMILARITY (orientation check)\n")
        f.write(f"cos(top_vectors)  : {cos_top:+.4f}  {'← aligned' if cos_top > 0.9 else '← opposite' if cos_top < -0.9 else '← ambiguous'}\n")
        f.write(f"cos(left_vectors) : {cos_left:+.4f}  {'← aligned' if cos_left > 0.9 else '← opposite' if cos_left < -0.9 else '← ambiguous'}\n")
        f.write(f"cos(mean)         : {(cos_top + cos_left)/2:+.4f}\n")
        f.write("\n")
        
        cos_mean = (cos_top + cos_left) / 2
        f.write("## AUTO-RECOMMENDATION\n")
        if cos_mean >= 0.9:
            f.write("Orientation: KEEP as-is (vectors aligned)\n")
        elif cos_mean <= -0.9:
            f.write("Orientation: FLIP camera2 corners (vectors opposite)\n")
        else:
            f.write("Orientation: AMBIGUOUS — review manually or reject frame\n")


def corner_orientation(corners1: np.ndarray, corners2: np.ndarray, 
                          rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, str]:

    c1_tl, c1_tr = corners1[0, 0], corners1[cols-1, 0]
    c1_bl = corners1[cols*(rows-1), 0]
    c2_tl, c2_tr = corners2[0, 0], corners2[cols-1, 0]
    c2_bl = corners2[cols*(rows-1), 0]
    
    def norm(v): 
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros_like(v)
    
    v1_top = norm(c1_tr - c1_tl)
    v1_left = norm(c1_bl - c1_tl)
    v2_top = norm(c2_tr - c2_tl)
    v2_left = norm(c2_bl - c2_tl)
    
    cos_top = np.dot(v1_top, v2_top)
    cos_left = np.dot(v1_left, v2_left)
    cos_mean = (cos_top + cos_left) / 2.0
    
    if cos_mean <= -0.9:
        return 'flipped'
    
    return  'kept'


    
def calibrate_stereo(
    cam1_intrinsics_path: str,
    cam2_intrinsics_path: str,
    cam1_imgs: str,
    cam2_imgs: str,
    config_path: str = "config/calibration_config.yaml",
    visualize: bool = None,
    invert: bool = None,
    save_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute stereo calibration parameters from two cameras.

    Parameters
    ----------
    cam1_intrinsics_path : str
        Path to the file containing the intrinsic calibration parameters of the first camera.
    cam2_intrinsics_path : str
        Path to the file containing the intrinsic calibration parameters of the second camera.
    cam1_imgs : str
        Path to the images of the first camera. The path should be given as a glob pattern.
    cam2_imgs : str
        Path to the images of the second camera. The path should be given as a glob pattern.
    config_path : str, optional
        Path to the configuration file.
    visualize : bool, optional
        If True, the images with the detected checkerboard corners will be displayed.
        Overrides config if provided.
    invert : bool, optional
        If True, inverts the images before processing. Overrides config if provided.
    save_path : str, optional
        Path to save the visualized images and calibration results.

    Returns
    -------
    R : numpy.ndarray
        Rotation matrix of the second camera relative to the first camera.
    T : numpy.ndarray
        Translation vector of the second camera relative to the first camera.
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
    
    algo_config = calib_config['algorithm']
    calib_flags = get_calibration_flags(algo_config.get('calibration_flags', []))
    debug_enabled = config.get('debug', {}).get('enabled', False)
    debug_path = config.get('debug', {}).get('path', None)
    
    refinement_cfg = calib_config.get('corner_refinement', {})
    refinement_enabled = refinement_cfg.get('enabled', False)
    
    vis_config = config['visualization']
    visualize_enabled = vis_config['enabled']
    visualize_delay = vis_config['delay']
    
    # Get save_corners from config
    save_corners = config.get('calibration', {}).get('save_corners', False)
    
    outlier_config = calib_config.get('outlier_detection', {})
    # max_corner_distance = outlier_config.get('max_corner_distance_threshold', 100.0)
    
    output_config = config.get('calibration', {}).get('output', {})
    corners_dir_name = output_config.get('corners_dir', 'corner_coordinates')
    debug_cam1_dir = output_config.get('debug_cam1_dir', 'camera_0_corners')
    debug_cam2_dir = output_config.get('debug_cam2_dir', 'camera_1_corners')
    debug_cam1_refined_dir = output_config.get('debug_cam1_refined_dir', 'camera_0_refined_corners')
    debug_cam2_refined_dir = output_config.get('debug_cam2_refined_dir', 'camera_1_refined_corners')
    
    # Create save directories if save_path is provided
    if save_path is not None:
        os.makedirs(Path(save_path), exist_ok=True)
        if save_corners:
            corners_save_dir = Path(save_path, corners_dir_name)
            # corners_save_dir = Path(save_path, 'corner_coordinates')
            os.makedirs(corners_save_dir, exist_ok=True)
    

    debug_txt_dir = Path('debug_stereo')
    debug_txt_dir.mkdir(parents=True, exist_ok=True)


    if debug_enabled:
        if debug_path is not None:
            cam0_save_dir = Path(debug_path, debug_cam1_dir)  
            cam1_save_dir = Path(debug_path, debug_cam2_dir)  
            refined_cam0_save_dir = Path(debug_path, debug_cam1_refined_dir)  
            refined_cam1_save_dir = Path(debug_path, debug_cam2_refined_dir)
            
            
            os.makedirs(cam0_save_dir, exist_ok=True)
            os.makedirs(cam1_save_dir, exist_ok=True)
            os.makedirs(refined_cam0_save_dir, exist_ok=True)
            os.makedirs(refined_cam1_save_dir, exist_ok=True)


    # Load intrinsics of both cameras
    logger.info(f"Loading intrinsics from {cam1_intrinsics_path} and {cam2_intrinsics_path}")
    cam1_intrinsics = np.load(cam1_intrinsics_path)
    cam2_intrinsics = np.load(cam2_intrinsics_path)
    mtx1, dist1 = cam1_intrinsics['mtx'], cam1_intrinsics['dist']
    mtx2, dist2 = cam2_intrinsics['mtx'], cam2_intrinsics['dist']
    
    
    logger.info(f"Camera 1 matrix:\n{mtx1}")
    logger.info(f"Camera 1 distortion:\n{dist1}")
    logger.info(f"Camera 2 matrix:\n{mtx2}")
    logger.info(f"Camera 2 distortion:\n{dist2}")

    # Load and preprocess images
    c1_image_paths = sorted(glob.glob(cam1_imgs))
    c2_image_paths = sorted(glob.glob(cam2_imgs))
    
    if not c1_image_paths or not c2_image_paths:
        logger.error(f"No images found at {cam1_imgs} or {cam2_imgs}")
        raise FileNotFoundError(f"No images found at {cam1_imgs} or {cam2_imgs}")
    
    if len(c1_image_paths) != len(c2_image_paths):
        logger.warning(f"Number of images for each camera differs: {len(c1_image_paths)} vs {len(c2_image_paths)}")
    
    logger.info(f"Found {len(c1_image_paths)} images for camera 1 and {len(c2_image_paths)} images for camera 2")
    
    # Process and load images for both cameras
    c1_images = []
    for imname in c1_image_paths:
        im = cv.imread(imname)
        if im is None:
            logger.warning(f"Failed to read image: {imname}")
            continue
        
        # Preprocess the image
        im = preprocess_calibration_image(
            im, resize_factor, invert_images, contrast_alpha, contrast_beta
        )
        c1_images.append(im)
    
    c2_images = []
    for imname in c2_image_paths:
        im = cv.imread(imname)
        if im is None:
            logger.warning(f"Failed to read image: {imname}")
            continue
        
        # Preprocess the image
        im = preprocess_calibration_image(
            im, resize_factor, invert_images, contrast_alpha, contrast_beta
        )
        c2_images.append(im)
    
    if not c1_images or not c2_images:
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
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)  # ← ПРАВИЛЬНО!
    objp = world_scaling * objp
    
    
    
    # Get frame dimensions 
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
    
    # Initialize arrays for detected points
    imgpoints_left = []   # 2D points in left image plane
    imgpoints_right = []  # 2D points in right image plane
    refined_imgpoints_left = []  # Refined 2D points in left image
    refined_imgpoints_right = [] # Refined 2D points in right image
    objpoints = []        # 3D points in world space
    

    for i, (frame1, frame2, path1, path2) in enumerate(
            zip(c1_images, c2_images, c1_image_paths, c2_image_paths)):
        # Detect corners in left image
        ret1, corners1 = detect_checkerboard_corners(
            frame1, rows, columns, subpix_window_size, criteria, flags, alt_flags
        )
        
        # Detect corners in right image
        ret2, corners2 = detect_checkerboard_corners(
            frame2, rows, columns, subpix_window_size, criteria, flags, alt_flags
        )
        
        # Only proceed if corners are found in both images
        if ret1 and ret2:
            

            orient_status = corner_orientation(
                corners1, corners2, rows, columns
            )
            if orient_status == 'flipped':
                continue
            

            debug_file = debug_txt_dir / f'pair_{i:04d}.txt'
            write_corner_debug_txt(
                debug_file, i, path1, path2,
                corners1, corners2, rows, columns
            )
            logger.debug(f"Saved debug info to {debug_file}")

            
            # Create visualizations for original corners
            vis_frame1 = create_visualization_frame(frame1, corners1, rows, columns, ret1)
            vis_frame2 = create_visualization_frame(frame2, corners2, rows, columns, ret2)
            
            # Store detected points
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            
            # Refine corners using linear regression
            logger.info(f"Refining corners for image pair {i}")
            
            # refined_corners1 = refine_corners_with_linear_regression(corners1, rows, columns)
            # refined_corners2 = refine_corners_with_linear_regression(corners2, rows, columns)
            if refinement_enabled:
                logger.info(f"Refining corners for image pair {i}")
                refined_corners1 = refine_corners_with_linear_regression(corners1, rows, columns)
                refined_corners2 = refine_corners_with_linear_regression(corners2, rows, columns)
            else:
                refined_corners1 = corners1
                refined_corners2 = corners2
                     
            
            refined_imgpoints_left.append(refined_corners1)
            refined_imgpoints_right.append(refined_corners2)
            
            # Create visualizations for refined corners
            refined_vis_frame1 = create_refined_corners_visualization(
                frame1, corners1, refined_corners1
            )
            refined_vis_frame2 = create_refined_corners_visualization(
                frame2, corners2, refined_corners2
            )

            # Save visualization images if save_path is provided
            if debug_enabled and debug_path is not None:
                frame1_filename = os.path.basename(path1)
                frame2_filename = os.path.basename(path2)
                
                cv.imwrite(str(Path(cam0_save_dir, frame1_filename)), vis_frame1)
                cv.imwrite(str(Path(cam1_save_dir, frame2_filename)), vis_frame2)
                cv.imwrite(str(Path(refined_cam0_save_dir, frame1_filename)), refined_vis_frame1)
                cv.imwrite(str(Path(refined_cam1_save_dir, frame2_filename)), refined_vis_frame2)
            
            # Display if visualization is enabled
            if visualize_enabled:
                cv.imshow('Camera 1 Original Corners', vis_frame1)
                cv.imshow('Camera 2 Original Corners', vis_frame2)
                cv.imshow('Camera 1 Refined Corners', refined_vis_frame1)
                cv.imshow('Camera 2 Refined Corners', refined_vis_frame2)
                
                key = cv.waitKey(visualize_delay)
                if key == 27:  # ESC key
                    logger.info("Visualization interrupted by user")
                    cv.destroyAllWindows()
                    visualize_enabled = False
            
            logger.info(f"Successfully detected corners in image pair {i}")

            # Save corner coordinates if requested
            if save_corners and save_path is not None:
                corners_file = Path(corners_save_dir, f'corners_{i:04d}.npz')
                np.savez(
                    corners_file,
                    cam1_original_corners=corners1,
                    cam2_original_corners=corners2,
                    cam1_refined_corners=refined_corners1,
                    cam2_refined_corners=refined_corners2,
                    object_points=objp
                )
                logger.info(f"Saved corner coordinates to {corners_file}")
        else:
            logger.warning(f"Failed to detect corners in image pair {i}")
            if visualize_enabled:
                cv.imshow('Failed detection - Camera 1', frame1)
                cv.imshow('Failed detection - Camera 2', frame2)
                key = cv.waitKey(visualize_delay)
                if key == 27:  # ESC key
                    logger.info("Visualization interrupted by user")
                    cv.destroyAllWindows()
                    visualize_enabled = False
    
    if visualize_enabled:
        cv.destroyAllWindows()
    

    orientation_logger.info("="*80)
    orientation_logger.info("SUMMARY")
    orientation_logger.info("="*80)
    orientation_logger.info(f"Total image pairs processed: {len(c1_images)}")
    orientation_logger.info(f"Successful detections: {len(objpoints)}")
    orientation_logger.info(f"Failed detections: {len(c1_images) - len(objpoints)}")
    orientation_logger.info("")
    orientation_logger.info("If all pairs show 'PASS' above, corner ordering is CORRECT!")
    orientation_logger.info("="*80)

    
    
    if not imgpoints_left or not imgpoints_right:
        logger.error("No chessboard corners detected in any image pairs")
        raise ValueError("No chessboard corners detected in any image pairs")
    
    # Run stereo calibration with refined corner points
    logger.info(f"Running stereo calibration with {len(refined_imgpoints_left)} image pairs")
    
    # Use CALIB_FIX_INTRINSIC flag to use the provided intrinsic parameters
    stereo_calib_flags = cv.CALIB_FIX_INTRINSIC


    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, 
        refined_imgpoints_left, 
        refined_imgpoints_right, 
        mtx1, dist1, mtx2, dist2,
        (width, height),
        criteria=criteria,
        flags=stereo_calib_flags
    )
    
    # Convert rotation matrix to Euler angles
    rotation = Rotation.from_matrix(R)
    euler_angles_rad = rotation.as_euler('xyz', degrees=False)
    euler_angles_deg = rotation.as_euler('xyz', degrees=True)
    
    # Log calibration results
    logger.info(f"STEREO CALIBRATION RESULTS:")
    logger.info(f"RMS reprojection error: {ret}")
    logger.info(f"Camera 1 matrix:\n{CM1}")
    logger.info(f"Camera 1 distortion:\n{dist1}")
    logger.info(f"Camera 2 matrix:\n{CM2}")
    logger.info(f"Camera 2 distortion:\n{dist2}")
    logger.info(f"Rotation matrix:\n{R}")
    logger.info(f"Euler angles (XYZ, radians): {euler_angles_rad}")
    logger.info(f"Euler angles (XYZ, degrees): {euler_angles_deg}")
    logger.info(f"Translation vector:\n{T}")
    
    return R, T, ret

def main(
    cam1_intrinsics_path: str,
    cam2_intrinsics_path: str,
    cam1_imgs: str,
    cam2_imgs: str,
    config_path: str = "config/calibration_config.yaml",
    save_path: str = None,
    visualize: bool = None,
    invert: bool = None
) -> None:
    
    """
    Compute stereo calibration parameters from two cameras and save them to a file.

    Parameters
    ----------
    cam1_intrinsics_path : str
        Path to the file containing the intrinsic calibration parameters of the first camera.
    cam2_intrinsics_path : str
        Path to the file containing the intrinsic calibration parameters of the second camera.
    cam1_imgs : str
        Path to the folder containing the images of the first camera. The path should be given as a glob pattern.
    cam2_imgs : str
        Path to the folder containing the images of the second camera. The path should be given as a glob pattern.
    config_path : str, optional
        Path to the configuration file.
    save_path : str, optional
        Path to the folder where the stereo calibration parameters will be saved. 
        If None, the parameters will not be saved. Defaults to None.
    visualize : bool, optional
        If True, the images with the detected checkerboard corners will be displayed. 
        Overrides config if provided.
    invert : bool, optional
        If True, inverts the images before processing. Overrides config if provided.
    """


    if save_path is not None:
        os.makedirs(Path(save_path), exist_ok=True)
    
    try:
        R, T, rms_error = calibrate_stereo(
            cam1_intrinsics_path,
            cam2_intrinsics_path,
            cam1_imgs,
            cam2_imgs,
            config_path=config_path,
            visualize=visualize,
            invert=invert,
            save_path=save_path
        )
        
        if save_path is not None:
            
            config = load_config(config_path)
            output_config = config.get('calibration', {}).get('output', {})
            
            stereo_name = output_config.get('stereo_filename', 'stereo.npz')
            report_name = output_config.get('report_filename', 'stereo_calibration_report.txt')
            
            # Convert rotation matrix to Euler angles for report
            rotation = Rotation.from_matrix(R)
            euler_angles_rad = rotation.as_euler('xyz', degrees=False)
            euler_angles_deg = rotation.as_euler('xyz', degrees=True)
            
            # Save calibration results
            stereo_file = Path(save_path, stereo_name) 
            # stereo_file = Path(save_path, 'stereo.npz')
            np.savez(stereo_file, R=R, T=T, euler_angles_rad=euler_angles_rad, euler_angles_deg=euler_angles_deg)
            logger.info(f"Saved stereo calibration results to {stereo_file}")
            
            # Save readable text report
            report_file = Path(save_path, report_name)
            # report_file = Path(save_path, 'stereo_calibration_report.txt')
            with open(report_file, 'w') as f:
                f.write(f"Stereo Calibration Results\n")
                f.write(f"="*50 + "\n\n")
                f.write(f"Camera 1: {cam1_intrinsics_path}\n")
                f.write(f"Camera 2: {cam2_intrinsics_path}\n\n")
                f.write(f"Calibration Quality:\n")
                f.write(f"  RMS Reprojection Error: {rms_error:.6f} pixels\n\n")
                f.write(f"Rotation Matrix:\n{R}\n\n")
                f.write(f"Euler Angles (XYZ convention):\n")
                f.write(f"  Radians: [{euler_angles_rad[0]:.6f}, {euler_angles_rad[1]:.6f}, {euler_angles_rad[2]:.6f}]\n")
                f.write(f"  Degrees: [{euler_angles_deg[0]:.6f}, {euler_angles_deg[1]:.6f}, {euler_angles_deg[2]:.6f}]\n\n")
                f.write(f"Translation Vector:\n{T}\n")
            logger.info(f"Saved stereo calibration report to {report_file}")
    
    except Exception as e:
        logger.error(f"Stereo calibration failed: {e}")
        raise

if __name__ == "__main__":
    # Fire(main)
    main(
        cam1_intrinsics_path = '/Users/cruelangel/Desktop/JARV/Ichthyander/calibration_results/left/intrinsics/intrinsics.npz',
        cam2_intrinsics_path = '/Users/cruelangel/Desktop/JARV/Ichthyander/calibration_results/right/intrinsics/intrinsics.npz',
        cam1_imgs = '/Users/cruelangel/Desktop/JARV/Ichthyander/calibration_data0/ir_left/*.png',
        cam2_imgs = '/Users/cruelangel/Desktop/JARV/Ichthyander/calibration_data0/ir_right/*.png',
        config_path = 'config/calibration_config_stereo_banner.yaml',
        save_path = 'calibration_results/stereo',
        visualize = False,
        invert = True
    )