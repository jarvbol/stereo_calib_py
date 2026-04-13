import numpy as np
import cv2 as cv
import logging
from typing import List, Tuple, Dict
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

def detect_checkerboard_corners(
    image: np.ndarray,
    rows: int,
    columns: int,
    subpix_window_size: int,
    criteria: Tuple,
    flags: int,
    alt_flags: int
) -> Tuple[bool, np.ndarray]:
    """
    Detect checkerboard corners in an image using OpenCV.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    rows : int
        Number of checkerboard rows.
    columns : int
        Number of checkerboard columns.
    subpix_window_size : int
        Window size for subpixel refinement.
    criteria : Tuple
        Termination criteria for cornerSubPix.
    flags : int
        Flags for findChessboardCorners.
    alt_flags : int
        Alternative flags for findChessboardCornersSB.
        
    Returns
    -------
    Tuple[bool, np.ndarray]
        Success flag and detected corners array.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # OpenCV patternSize convention: (points_per_row, points_per_column) = (columns, rows)
    ret, corners = cv.findChessboardCorners(gray, (columns, rows), None, flags=flags)
    
    # If standard method fails, try alternative method
    if not ret:
        logger.debug(f"Standard findChessboardCorners failed, trying findChessboardCornersSB")
        ret, corners = cv.findChessboardCornersSB(gray, (columns, rows), flags=alt_flags)
    
    if ret:
        # Refine corner positions
        conv_size = (subpix_window_size, subpix_window_size)
        corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
    
    return ret, corners

def create_visualization_frame(
    image: np.ndarray,
    corners: np.ndarray,
    rows: int,
    columns: int,
    success: bool
) -> np.ndarray:
    """
    Create a visualization frame showing detected corners.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    corners : np.ndarray
        Detected corners.
    rows : int
        Number of checkerboard rows.
    columns : int
        Number of checkerboard columns.
    success : bool
        Whether corner detection was successful.
        
    Returns
    -------
    np.ndarray
        Visualization frame.
    """
    vis_frame = image.copy()
    if success:
        # OpenCV patternSize convention: (columns, rows)
        cv.drawChessboardCorners(vis_frame, (columns, rows), corners, True)
    return vis_frame

def create_refined_corners_visualization(
    original_frame: np.ndarray,
    original_corners: np.ndarray,
    refined_corners: np.ndarray
) -> np.ndarray:
    """
    Create a visualization showing both original and refined corners.
    
    Parameters
    ----------
    original_frame : np.ndarray
        Original image.
    original_corners : np.ndarray
        Original detected corners.
    refined_corners : np.ndarray
        Refined corner positions.
        
    Returns
    -------
    np.ndarray
        Visualization frame with both corner sets.
    """
    # First upscale the image for better visualization
    scale_factor = 4.0
    h, w = original_frame.shape[:2]
    upscaled_frame = cv.resize(original_frame, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv.INTER_LINEAR)
    
    # Scale the corner coordinates to match the upscaled image
    original_corners_scaled = original_corners.copy()
    refined_corners_scaled = refined_corners.copy()
    
    # Scale the coordinates
    original_corners_scaled = original_corners_scaled.reshape(-1, 2) * scale_factor
    refined_corners_scaled = refined_corners_scaled.reshape(-1, 2) * scale_factor
    
    # Draw the original corners as red circles
    for corner in original_corners_scaled:
        cv.circle(upscaled_frame, tuple(corner.astype(int)), 3, (0, 0, 255), -1)  # Small red circles
        
    # Draw the refined corners as green circles
    for corner in refined_corners_scaled:
        cv.circle(upscaled_frame, tuple(corner.astype(int)), 3, (0, 255, 0), -1)  # Small green circles
    
    # Draw lines connecting original and refined corners
    for j in range(len(original_corners_scaled)):
        pt1 = tuple(original_corners_scaled[j].astype(int))
        pt2 = tuple(refined_corners_scaled[j].astype(int))
        cv.line(upscaled_frame, pt1, pt2, (255, 0, 255), 1)  # Thin magenta lines
    
    # Add legend
    legend_y_start = 30
    cv.putText(upscaled_frame, "Original corners", (20, legend_y_start), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(upscaled_frame, "Refined corners", (20, legend_y_start + 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return upscaled_frame

def preprocess_calibration_image(
    image: np.ndarray,
    resize_factor: float,
    invert: bool,
    contrast_alpha: float,
    contrast_beta: float
) -> np.ndarray:
    """
    Preprocess an image for calibration.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    resize_factor : float
        Resize factor for the image.
    invert : bool
        Whether to invert the image.
    contrast_alpha : float
        Contrast alpha value.
    contrast_beta : float
        Contrast beta value.
        
    Returns
    -------
    np.ndarray
        Preprocessed image.
    """
    # Resize the image
    if resize_factor != 1.0:
        image = cv.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_LINEAR)
    
    # Invert if needed
    if invert:
        image = cv.bitwise_not(image)
    
    # Adjust contrast
    if contrast_alpha != 1.0 or contrast_beta != 0:
        image = cv.convertScaleAbs(image, alpha=contrast_alpha, beta=contrast_beta)
    
    return image

def visualize_line_fitting(
    corners: np.ndarray, 
    rows: int, 
    columns: int,
    image: np.ndarray,
    image_number: int,
    save_path: str
) -> None:
    """
    Create debug visualizations showing row and column line fitting on the original image.
    
    Parameters
    ----------
    corners : np.ndarray
        Detected corners (n x 1 x 2 array).
    rows : int
        Number of checkerboard rows.
    columns : int
        Number of checkerboard columns.
    image : np.ndarray
        Original image to draw on.
    image_number : int
        Index of the image being processed.
    save_path : str
        Directory to save visualization images.
    """
    os.makedirs(save_path, exist_ok=True)
    
    reshaped = corners.reshape(rows, columns, 2)
    
    # Make a copy of the image for row visualization
    row_img = image.copy()
    
    # Draw each row with a different color
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
        (128, 128, 0)   # Dark Cyan
    ]
    
    # Visualize rows - In the reshaped array, rows are accessed as [r,:,:]
    for r in range(rows):
        # Get row points: all columns for this row
        row_points = reshaped[r, :, :]
        
        # Extract x and y coordinates
        x_coords = row_points[:, 0]
        y_coords = row_points[:, 1]
        
        # Draw row points
        color = colors[r % len(colors)]
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            cv.circle(row_img, (int(x), int(y)), 5, color, -1)
            
        # Fit line if possible and draw it
        if len(np.unique(x_coords)) > 1:
            model = LinearRegression()
            model.fit(x_coords.reshape(-1, 1), y_coords)
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Calculate line endpoints for visualization
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min = int(slope * x_min + intercept)
            y_max = int(slope * x_max + intercept)
            
            # Draw line
            cv.line(row_img, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Add text label for the row
            text_pos = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            cv.putText(row_img, f"Row {r}", text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Vertical line
            x_avg = int(np.mean(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cv.line(row_img, (x_avg, y_min), (x_avg, y_max), color, 2)
            cv.putText(row_img, f"Row {r}", (x_avg, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save row visualization
    cv.imwrite(os.path.join(save_path, f'rows_debug_img_{image_number}.png'), row_img)
    
    # Make a copy of the image for column visualization
    col_img = image.copy()
    
    # Visualize columns - In the reshaped array, columns are accessed as [:,c,:]
    for c in range(columns):
        # Get column points: all rows for this column
        col_points = reshaped[:, c, :]
        
        # Extract x and y coordinates
        x_coords = col_points[:, 0]
        y_coords = col_points[:, 1]
        
        # Draw column points
        color = colors[c % len(colors)]
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            cv.circle(col_img, (int(x), int(y)), 5, color, -1)
            
        # Fit line if possible and draw it
        if len(np.unique(y_coords)) > 1:
            model = LinearRegression()
            model.fit(y_coords.reshape(-1, 1), x_coords)
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Calculate line endpoints for visualization
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            x_min = int(slope * y_min + intercept)
            x_max = int(slope * y_max + intercept)
            
            # Draw line
            cv.line(col_img, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Add text label for the column
            text_pos = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            cv.putText(col_img, f"Col {c}", text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Horizontal line
            y_avg = int(np.mean(y_coords))
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            cv.line(col_img, (x_min, y_avg), (x_max, y_avg), color, 2)
            cv.putText(col_img, f"Col {c}", (x_min - 30, y_avg), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save column visualization
    cv.imwrite(os.path.join(save_path, f'columns_debug_img_{image_number}.png'), col_img)
    
    # Create a combined visualization with both rows and columns
    combined_img = image.copy()
    
    # Draw all corners
    corners_flat = corners.reshape(-1, 2)
    for corner in corners_flat:
        cv.circle(combined_img, (int(corner[0]), int(corner[1])), 3, (0, 0, 255), -1)  # Red circles
    
    # Draw row lines
    for r in range(rows):
        row_points = reshaped[r, :, :]
        x_coords = row_points[:, 0]
        y_coords = row_points[:, 1]
        
        # Draw the row points and connect them with a polyline
        pts = np.array([[int(x), int(y)] for x, y in zip(x_coords, y_coords)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(combined_img, [pts], False, (0, 255, 0), 2)  # Green lines for rows
    
    # Draw column lines
    for c in range(columns):
        col_points = reshaped[:, c, :]
        x_coords = col_points[:, 0]
        y_coords = col_points[:, 1]
        
        # Draw the column points and connect them with a polyline
        pts = np.array([[int(x), int(y)] for x, y in zip(x_coords, y_coords)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(combined_img, [pts], False, (255, 0, 0), 2)  # Blue lines for columns
    
    # Save combined visualization
    cv.imwrite(os.path.join(save_path, f'combined_debug_img_{image_number}.png'), combined_img)