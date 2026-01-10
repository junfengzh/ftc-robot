#!/usr/bin/env python3
"""
Core utility functions for keypoint detection in goal images.
Contains calibration, color detection, contour processing, and keypoint extraction functions.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def get_calibration_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """Initialize camera calibration parameters."""
    camera_matrix = np.array([
        [786.357, 0, 989.731],
        [0, 785.863, 501.663],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.array([
        -0.370767, 0.106516, 0.000118, -0.000542, -0.012277
    ], dtype=np.float32)
    
    return camera_matrix, dist_coeffs


def undistort_and_scale(frame: np.ndarray, camera_matrix: np.ndarray, 
                       dist_coeffs: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Undistort the frame and scale to target size."""
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
    return cv2.resize(undistorted, target_size)


def undistort_points(points: List[Tuple[float, float]], camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray, original_size: Tuple[int, int],
                     target_size: Optional[Tuple[int, int]] = None) -> List[Optional[Tuple[float, float]]]:
    """Undistort points from distorted image coordinates to undistorted coordinates.
    
    Args:
        points: List of (x, y) points in distorted image coordinates
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        original_size: Size of the original image (width, height)
        target_size: Optional target size to scale to (width, height)
        
    Returns:
        List of undistorted (x, y) points, or None if point falls outside image bounds
    """
    if not points:
        return []
    
    # Convert points to the format required by cv2.undistortPoints
    # Input shape should be (N, 1, 2) where N is number of points
    points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Undistort the points
    undistorted = cv2.undistortPoints(points_array, camera_matrix, dist_coeffs, P=camera_matrix)
    
    # Reshape back to (N, 2)
    undistorted_points = undistorted.reshape(-1, 2)
    
    # Check if points are within bounds and optionally scale
    result = []
    for point in undistorted_points:
        x, y = point
        
        # Check if point is within original image bounds
        if x < 0 or x >= original_size[0] or y < 0 or y >= original_size[1]:
            result.append(None)
            continue
        
        # Scale to target size if specified
        if target_size is not None:
            scale_x = target_size[0] / original_size[0]
            scale_y = target_size[1] / original_size[1]
            x = x * scale_x
            y = y * scale_y
        
        result.append((float(x), float(y)))
    
    return result
    return cv2.resize(undistorted, target_size)


def detect_color_contours(frame: np.ndarray, color: str, apply_region_mask: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    """Detect contours for specified color.
    
    For yellow: uses full frame
    For red/blue: uses top 40% of frame only (if apply_region_mask is True)
    
    Args:
        frame: Input frame
        color: Color to detect ('red', 'blue', or 'yellow')
        apply_region_mask: Whether to apply region mask for red/blue (default: True)
        
    Returns:
        Tuple of (contours, filtered_frame)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    
    # Create color mask based on color
    if color == "red":
        # Red has two ranges in HSV; OLD
        # lower_red1 = (170, 140, 70)
        # upper_red1 = (180, 255, 225)
        # lower_red2 = (0, 140, 70)
        # upper_red2 = (10, 255, 225)
        # NEW: More lenient
        lower_red1 = (170, 100, 30)
        upper_red1 = (180, 255, 225)
        lower_red2 = (0, 100, 30)
        upper_red2 = (10, 255, 225)
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "blue":
        lower_blue = (105, 140, 70)
        upper_blue = (115, 255, 225)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    elif color == "yellow":
        lower_yellow = (15, 50, 50)
        upper_yellow = (50, 255, 255)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    else:
        raise ValueError(f"Unsupported color: {color}")
    
    # Apply region mask for red and blue (top 40% only), but not for yellow (full frame)
    if color in ["red", "blue"] and apply_region_mask:
        region_mask = np.zeros((height, width), dtype=np.uint8)
        region_mask[:int(height * 0.4), :] = 255  # Only process top half
        mask = cv2.bitwise_and(mask, region_mask)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return contours, filtered_frame


def find_highest_point(contour: np.ndarray) -> Optional[Tuple[tuple, float]]:
    """Find the highest point in a contour."""
    contour_points = contour.squeeze()
    if len(contour_points.shape) >= 2:
        min_y_idx = np.argmin(contour_points[:, 1])
        highest_point = tuple(contour_points[min_y_idx])
        return highest_point, float(highest_point[1])
    return None


def find_top_straight_line(contour: np.ndarray, tolerance: float = 5.0) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Find the top straight line of a contour and return its endpoints and midpoint.
    
    Args:
        contour: The contour to analyze
        tolerance: Y-coordinate tolerance for considering points on the same horizontal line
    
    Returns:
        Tuple of (left_point, right_point, midpoint) if found, None otherwise
    """
    if contour is None or len(contour) < 3:
        return None
        
    contour_points = contour.squeeze()
    if len(contour_points.shape) != 2 or contour_points.shape[1] != 2:
        return None
    
    # Find the minimum Y coordinate (topmost point)
    min_y = np.min(contour_points[:, 1])
    
    # Find all points within tolerance of the minimum Y
    top_points = contour_points[np.abs(contour_points[:, 1] - min_y) <= tolerance]
    
    if len(top_points) < 2:
        return None
    
    # Sort by X coordinate to find leftmost and rightmost
    top_points = top_points[np.argsort(top_points[:, 0])]
    
    left_point = tuple(top_points[0].astype(int))
    right_point = tuple(top_points[-1].astype(int))
    
    # Calculate midpoint
    midpoint = (
        int((left_point[0] + right_point[0]) / 2),
        int((left_point[1] + right_point[1]) / 2)
    )
    
    return left_point, right_point, midpoint


def process_contours(contours: List[np.ndarray], min_area: float) -> Tuple[List[np.ndarray], Optional[Tuple[tuple, float]]]:
    """Process contours to find the highest point in the contour with largest area."""
    largest_contour = None
    largest_area = 0
    
    # Find the contour with the largest area that meets the minimum threshold
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area > largest_area:
            largest_area = area
            largest_contour = contour
    
    # Find the highest point in the largest contour
    if largest_contour is not None:
        result = find_highest_point(largest_contour)
        if result is not None:
            point, y = result
            return point
    
    return (None, None)


def filter_topmost_yellow_contour(contours: List[np.ndarray], frame_area: float) -> List[np.ndarray]:
    """Filter yellow contours to keep only the topmost one.
    
    Args:
        contours: List of contours to filter
        frame_area: Total area of the frame (width * height)
    
    Returns:
        List containing only the topmost contour, or empty list if none found
    """
    if not contours:
        return contours
    
    # Find the contour with the smallest (topmost) Y coordinate
    topmost_contour = None
    topmost_y = float('inf')
    
    min_area = frame_area * 0.0006  # Proportional minimum area threshold
    
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            result = find_highest_point(contour)
            if result is not None:
                point, y = result
                if y < topmost_y:
                    topmost_y = y
                    topmost_contour = contour
    
    return [topmost_contour] if topmost_contour is not None else []


def detect_yellow_in_contour_interiors(frame: np.ndarray, color_contours: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Detect yellow contours only within the interior regions of red and blue contours."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    frame_area = height * width
    
    # Create mask for yellow detection
    lower_yellow = (15, 50, 50)
    upper_yellow = (50, 255, 255)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Create interior mask from red and blue contours
    interior_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add interior regions of red contours with erosion to ensure true interior
    for contour in color_contours:
        if cv2.contourArea(contour) >= frame_area * 0.0013:  # Use larger contours only
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [contour], 255)
            
            # Check if contour touches frame edges (for U-shaped contours)
            x, y, w, h = cv2.boundingRect(contour)
            touches_left = (x <= 3)
            touches_right = (x + w >= width - 3)
            touches_top = (y <= 3)
            touches_bottom = (y + h >= height - 3)
            touches_edge = touches_left or touches_right or touches_top or touches_bottom
            
            if touches_edge:
                # Fill from the edge side(s) that the contour touches
                flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
                
                # Invert temp_mask to find exterior regions
                exterior_mask = cv2.bitwise_not(temp_mask)
                
                # Flood fill from midpoints of sides that are NOT touched to find exterior regions
                # We'll then invert to get the interior + U-bounded area
                filled_exterior = exterior_mask.copy()
                
                # Try flood filling from the midpoint of each side that isn't touched
                if not touches_top:
                    cv2.floodFill(filled_exterior, flood_mask, (width // 2, 0), 0)
                if not touches_bottom:
                    cv2.floodFill(filled_exterior, flood_mask, (width // 2, height - 1), 0)
                if not touches_left:
                    cv2.floodFill(filled_exterior, flood_mask, (0, height // 2), 0)
                if not touches_right:
                    cv2.floodFill(filled_exterior, flood_mask, (width - 1, height // 2), 0)
                
                # What remains in filled_exterior is the interior + U-bounded area
                temp_mask = cv2.bitwise_or(temp_mask, filled_exterior)
            
            interior_mask = cv2.bitwise_or(interior_mask, temp_mask)
    
    # Apply interior mask to yellow detection
    yellow_mask = cv2.bitwise_and(yellow_mask, interior_mask)
    
    # Fill gaps caused by text within yellow areas
    # Use closing to fill small holes and gaps within yellow regions
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
    
    # Find contours in the masked yellow regions
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    
    return contours, filtered_frame, yellow_mask, interior_mask


def draw_results(frame: np.ndarray, contours: List[np.ndarray], 
                highest_point: Optional[Tuple[tuple, float]], 
                color: Tuple[int, int, int], draw_top_line: bool = False,
                frame_area: Optional[float] = None) -> None:
    """Draw contours and highest point on the frame.
    
    Args:
        frame: The frame to draw on
        contours: List of contours to draw
        highest_point: Highest point to mark
        color: Color for drawing contours
        draw_top_line: Whether to find and draw top straight line (for yellow contours)
        frame_area: Total area of the frame (width * height), required if draw_top_line is True
    """
    if contours:
        cv2.drawContours(frame, contours, -1, color, 1)
        # Only draw highest point if not drawing top line (skip for yellow contours)
        if highest_point is not None and not draw_top_line:
            point, y = highest_point
            cv2.circle(frame, point, 4, (0, 255, 0), -1)
        
        # Draw top straight line for yellow contours
        if draw_top_line:
            if frame_area is None:
                frame_area = frame.shape[0] * frame.shape[1]
            for contour in contours:
                if cv2.contourArea(contour) >= frame_area * 0.0006:  # Only for reasonably sized contours
                    line_result = find_top_straight_line(contour)
                    if line_result is not None:
                        left_point, right_point, midpoint = line_result
                        
                        # Draw the top line in bright green
                        cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
                        
                        # Draw the midpoint as a large red circle
                        cv2.circle(frame, midpoint, 4, (0, 0, 255), -1)


def calculate_relative_angle(point: Tuple[int, int], width: int, focal_length: float = 131.06) -> float:
    """Calculate the relative angle of a point using the camera's focal length.
    
    Args:
        point: The (x,y) coordinates of the point
        width: The width of the frame
        focal_length: Camera's focal length in pixels (default: fx from calibration)
    
    Returns:
        angle: The angle in degrees from the center
    """
    x, _ = point
    center_x = width / 2
    angle = np.arctan2(x - center_x, focal_length)
    return np.degrees(angle)


def show_color_contours(frame: np.ndarray, category_id: int, 
                       target_size: Tuple[int, int], window_name: str = "Color Contours") -> None:
    """Display frame with contours used in the detection pipeline.
    
    Shows:
    - Red/Blue contours in their respective colors
    - Yellow contours in yellow
    - Goal top point (highest colored point) in green
    - Tag top point (midpoint of yellow top line) in red
    - Top yellow line in bright green
    
    Args:
        frame: Original frame
        category_id: 1 for Blue Goal, 2 for Red Goal
        target_size: Target size for processing
        window_name: Name of the window to display
    """
    # Scale frame
    processed_frame = cv2.resize(frame, target_size)
    vis_frame = processed_frame.copy()
    
    frame_area = target_size[0] * target_size[1]
    
    # Determine color based on category_id
    color = "blue" if category_id == 1 else "red"
    color_bgr = (255, 0, 0) if category_id == 1 else (0, 0, 255)
    
    # Detect colored contours (red or blue) with region mask for goal-top detection
    color_contours_masked, _ = detect_color_contours(processed_frame, color, apply_region_mask=True)
    color_highest = process_contours(color_contours_masked, frame_area * 0.002)
    
    # Draw masked colored contours (used for goal-top detection)
    if color_contours_masked:
        cv2.drawContours(vis_frame, color_contours_masked, -1, color_bgr, 1)
    
    # Draw goal top point
    if color_highest is not None and color_highest != (None, None):
        cv2.circle(vis_frame, color_highest, 5, (0, 255, 0), -1)
        cv2.putText(vis_frame, "Goal-Top", (color_highest[0] + 10, color_highest[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Detect colored contours without region mask for yellow interior detection
    color_contours_full, _ = detect_color_contours(processed_frame, color, apply_region_mask=False)
    
    # Detect yellow contours within full colored contours
    yellow_contours, _, yellow_mask, interior_mask = detect_yellow_in_contour_interiors(processed_frame, color_contours_full)
    
    # Filter to keep only the topmost yellow contour
    filtered_yellow_contours = filter_topmost_yellow_contour(yellow_contours, frame_area)
    
    # Draw yellow contours
    if filtered_yellow_contours:
        cv2.drawContours(vis_frame, filtered_yellow_contours, -1, (0, 255, 255), 1)
    
    # Draw tag top point and line
    for contour in filtered_yellow_contours:
        if cv2.contourArea(contour) > frame_area * 0.0001:
            line_result = find_top_straight_line(contour)
            if line_result is not None:
                left_point, right_point, midpoint = line_result
                
                # Draw the top line in bright green
                cv2.line(vis_frame, left_point, right_point, (0, 255, 0), 1)
                
                # Draw the midpoint as a large red circle
                cv2.circle(vis_frame, midpoint, 5, (0, 0, 255), -1)
                cv2.putText(vis_frame, "Tag-Top", (midpoint[0] + 10, midpoint[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break
    
    # Convert grayscale interior mask to BGR for side-by-side display
    interior_mask_bgr = cv2.cvtColor(interior_mask, cv2.COLOR_GRAY2BGR)
    
    # Create side-by-side display
    combined_view = np.hstack([vis_frame, interior_mask_bgr])
    
    # Add labels to distinguish the two views
    cv2.putText(combined_view, "Contours", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_view, "Interior Mask", (target_size[0] + 10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the combined frame
    cv2.imshow(window_name, combined_view)


def detect_goal_keypoints(frame: np.ndarray, category_id: int,
                         target_size: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Detect goal top and tag top keypoints from a frame.
    
    This is the ultimate, comprehensive function that detects both the goal-top 
    (topmost colored point) and tag-top (middle of yellow line) keypoints.
    
    Args:
        frame: Original frame
        category_id: 1 for Blue Goal, 2 for Red Goal
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        target_size: Target size for processing
        
    Returns:
        Tuple of ((goal_top_x, goal_top_y), (tag_top_x, tag_top_y)) or None
    """
    # Scale frame without undistortion
    processed_frame = cv2.resize(frame, target_size)
    
    frame_area = target_size[0] * target_size[1]

    # Determine color based on category_id
    color = "blue" if category_id == 1 else "red"
    
    # Detect colored contours (red or blue) with region mask for goal-top detection
    color_contours_masked, _ = detect_color_contours(processed_frame, color, apply_region_mask=True)
    color_highest = process_contours(color_contours_masked, frame_area * 0.002)
    
    goal_top = color_highest
    
    # Detect colored contours without region mask for yellow interior detection
    color_contours_full, _ = detect_color_contours(processed_frame, color, apply_region_mask=False)
    
    # Detect yellow contours within colored contours
    yellow_contours, _, _, _ = detect_yellow_in_contour_interiors(processed_frame, color_contours_full)
    
    # Filter to keep only the topmost yellow contour
    filtered_yellow_contours = filter_topmost_yellow_contour(yellow_contours, frame_area)
    
    # Get the midpoint of the top line of the yellow contour
    tag_top = (None, None)
    for contour in filtered_yellow_contours:
        if cv2.contourArea(contour) > frame_area * 0.0001:
            line_result = find_top_straight_line(contour)
            if line_result is not None:
                _, _, midpoint = line_result
                tag_top = midpoint
                break
    
    return goal_top, tag_top
