#!/usr/bin/env python3
"""
Test script that uses live camera feed to detect and display goal keypoints.

Goal top: topmost red point found
Tag top: middle of the yellow point line
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

# Import functions from keypoint_detection_functions.py
from keypoint_detection_functions import (
    detect_goal_keypoints,
    show_color_contours,
    show_approx_poly_contours
)


def visualize_keypoints(frame: np.ndarray, detected_goal_top: Tuple[int, int], 
                       detected_tag_top: Tuple[int, int],
                       target_size: Tuple[int, int]) -> np.ndarray:
    """Visualize detected keypoints on the frame.
    
    Args:
        frame: Original frame from camera
        detected_goal_top: Detected goal top coordinates (scaled resolution) or (None, None)
        detected_tag_top: Detected tag top coordinates (scaled resolution) or (None, None)
        target_size: Target processing size (320x240)
        
    Returns:
        Frame with visualizations drawn
    """
    # Scale up coordinates to original resolution for visualization
    scale_x = frame.shape[1] / target_size[0]
    scale_y = frame.shape[0] / target_size[1]
    
    # Create a copy to draw on
    vis_frame = frame.copy()
    
    # Handle detected goal_top
    if detected_goal_top != (None, None):
        det_goal_orig = (int(detected_goal_top[0] * scale_x), int(detected_goal_top[1] * scale_y))
        cv2.circle(vis_frame, det_goal_orig, 8, (0, 255, 0), -1)
        cv2.putText(vis_frame, "Goal-Top", (det_goal_orig[0] + 10, det_goal_orig[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Display coordinates
        coord_text = f"({detected_goal_top[0]:.0f}, {detected_goal_top[1]:.0f})"
        cv2.putText(vis_frame, coord_text, (det_goal_orig[0] + 10, det_goal_orig[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(vis_frame, "Goal-Top: NOT DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Handle detected tag_top
    if detected_tag_top != (None, None):
        det_tag_orig = (int(detected_tag_top[0] * scale_x), int(detected_tag_top[1] * scale_y))
        cv2.circle(vis_frame, det_tag_orig, 8, (0, 255, 0), -1)
        cv2.putText(vis_frame, "Tag-Top", (det_tag_orig[0] + 10, det_tag_orig[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Display coordinates
        coord_text = f"({detected_tag_top[0]:.0f}, {detected_tag_top[1]:.0f})"
        cv2.putText(vis_frame, coord_text, (det_tag_orig[0] + 10, det_tag_orig[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(vis_frame, "Tag-Top: NOT DETECTED", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return vis_frame


def main():
    # Default to Red Goal (category_id=2), can be changed with command line argument
    category_id = 2  # Red Goal
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'blue':
            category_id = 1  # Blue Goal
        elif sys.argv[1].lower() == 'red':
            category_id = 2  # Red Goal
        else:
            print("Usage: python test_opencv_2.py [blue|red]")
            print("Defaulting to Red Goal")
    
    target_size = (320, 240)
    color_name = "Blue" if category_id == 1 else "Red"
    
    print(f"Starting camera feed for {color_name} Goal detection...")
    print(f"Target processing size: {target_size}")
    print("Press 'q' to quit")
    print("Press 'c' to show color contours")
    print("Press 'a' to show approxPolyDP contours")
    print("Press 'b' to toggle between Blue and Red goal detection")
    
    # Open camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Try to set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    show_contours = False
    show_approx = False
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Detect keypoints
            detected_goal_top, detected_tag_top = detect_goal_keypoints(frame, category_id, target_size)
            
            # Visualize keypoints on frame
            vis_frame = visualize_keypoints(frame, detected_goal_top, detected_tag_top, target_size)
            
            # Display the resulting frame
            window_title = f"{color_name} Goal Detection"
            cv2.imshow(window_title, vis_frame)
            
            # Display and update contours every frame if enabled
            if show_contours:
                contour_frame = show_color_contours(frame, category_id, target_size, f"{color_name} Goal - Color Contours")
            
            # Display and update approxPolyDP every frame if enabled
            if show_approx:
                approx_frame = show_approx_poly_contours(frame, category_id, target_size, f"{color_name} Goal - ApproxPolyDP")
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_contours = not show_contours
                if not show_contours:
                    cv2.destroyWindow(f"{color_name} Goal - Color Contours")
                print(f"Color contours: {'ON' if show_contours else 'OFF'}")
            elif key == ord('a'):
                show_approx = not show_approx
                if not show_approx:
                    cv2.destroyWindow(f"{color_name} Goal - ApproxPolyDP")
                print(f"ApproxPolyDP contours: {'ON' if show_approx else 'OFF'}")
            elif key == ord('b'):
                # Toggle between Blue and Red
                category_id = 2 if category_id == 1 else 1
                color_name = "Blue" if category_id == 1 else "Red"
                print(f"Switched to {color_name} Goal detection")
                # Destroy old windows
                cv2.destroyAllWindows()
                show_contours = False
                show_approx = False
    
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera feed closed.")


if __name__ == "__main__":
    main()
