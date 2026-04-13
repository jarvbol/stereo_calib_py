import numpy as np
import cv2 as cv
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict
import logging
import os

logger = logging.getLogger(__name__)

def refine_corners_with_linear_regression(corners: np.ndarray, 
                                         rows: int, 
                                         columns: int) -> np.ndarray:
    """
    Refine corner positions using simple linear regression on rows and columns.
    
    Parameters
    ----------
    corners : np.ndarray
        Detected corners (n x 1 x 2 array).
    rows : int
        Number of checkerboard rows.
    columns : int
        Number of checkerboard columns.
        
    Returns
    -------
    np.ndarray
        Refined corner positions.
    """
    # Store original shape for reshaping back at the end
    original_shape = corners.shape
    
    # Reshape corners properly according to checkerboard dimensions.
    reshaped = corners.reshape(rows, columns, 2)
    
    # Create empty array for refined corners with same shape
    refined_reshaped = np.zeros_like(reshaped)
    
    # Fit horizontal (image row) lines using linear regression.
    # Fit y = mx + b (x varies widely, y nearly constant).
    row_lines = []
    for r in range(rows):
        # Get all corners in image row r
        row_points = reshaped[r, :, :]
        
        # Extract x and y coordinates
        x_coords = row_points[:, 0].reshape(-1, 1)
        y_coords = row_points[:, 1]
        
        if len(np.unique(x_coords)) > 1:
            # Fit a line: y = mx + b
            model = LinearRegression()
            model.fit(x_coords, y_coords)
            slope = model.coef_[0]
            intercept = model.intercept_
            row_lines.append((slope, intercept, "y=mx+b"))  # y = mx + b
        else:
            # Degenerate: all corners in same column (x = constant)
            avg_x = np.mean(x_coords)
            row_lines.append((None, avg_x, "x=c"))  # x = constant
    
    # Fit vertical (image column) lines using linear regression.
    # Fit x = my + b (y varies widely, x nearly constant).
    col_lines = []
    for c in range(columns):
        # Get all corners in image column c
        col_points = reshaped[:, c, :]
        
        # Extract x and y coordinates
        y_coords = col_points[:, 1].reshape(-1, 1)
        x_coords = col_points[:, 0]
        
        if len(np.unique(y_coords)) > 1:
            # Fit a line: x = my + b
            model = LinearRegression()
            model.fit(y_coords, x_coords)
            slope = model.coef_[0]
            intercept = model.intercept_
            col_lines.append((slope, intercept, "x=my+b"))  # x = my + b
        else:
            # Degenerate: all corners in same row (y = constant)
            avg_y = np.mean(y_coords)
            col_lines.append((None, avg_y, "y=c"))  # y = constant
    
    # Compute intersection points of horizontal row lines and vertical column lines.
    for r in range(rows):
        for c in range(columns):
            row_info = row_lines[r]
            col_info = col_lines[c]
            
            row_slope, row_intercept, row_type = row_info
            col_slope, col_intercept, col_type = col_info
            
            # Handle different cases based on line types
            if row_type == "x=c" and col_type == "y=c":
                # Row is vertical (x=c), column is horizontal (y=c)
                x = row_intercept
                y = col_intercept
            elif row_type == "x=c":
                # Row is vertical (x=c), column is regular (x=my+b)
                x = row_intercept
                y = (x - col_intercept) / col_slope if col_slope != 0 else col_intercept
            elif col_type == "y=c":
                # Column is horizontal (y=c), row is regular (y=mx+b)
                y = col_intercept
                x = (y - row_intercept) / row_slope if row_slope != 0 else reshaped[r, c, 0]
            else:
                # Both are regular lines: y=mx+b and x=my+b
                # Solve: row_slope*x + row_intercept = y = (x - col_intercept)/col_slope
                # x = (row_intercept*col_slope + col_intercept) / (1 - row_slope*col_slope)
                if abs(1 - row_slope * col_slope) > 1e-6:  # avoid division by near-zero
                    x = (row_intercept * col_slope + col_intercept) / (1 - row_slope * col_slope)
                    y = row_slope * x + row_intercept
                else:
                    # Lines are nearly parallel, use original point
                    x, y = reshaped[r, c]
            
            refined_reshaped[r, c] = [x, y]
    
    # Reshape back to original shape
    refined_corners = refined_reshaped.reshape(original_shape)
    
    return refined_corners