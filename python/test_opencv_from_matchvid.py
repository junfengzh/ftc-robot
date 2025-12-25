#!/usr/bin/env python3
"""
Test script that extracts keypoint annotations from goalannotations.json
and calculates the deviation between annotated coordinates and detected coordinates.

Goal top: topmost red point found
Tag top: middle of the yellow point line
"""

import json
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import functions from keypoint_detection_functions.py
from keypoint_detection_functions import (
    get_calibration_matrices,
    detect_goal_keypoints,
    undistort_points
)


def load_annotations(json_path: str) -> Dict:
    """Load the COCO format annotations from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_keypoints(annotation: Dict) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Extract Goal-Top and Tag-Top keypoints from an annotation.
    
    Args:
        annotation: Single annotation dict with keypoints
        
    Returns:
        Tuple of ((goal_top_x, goal_top_y), (tag_top_x, tag_top_y)) or None if invalid
    """
    keypoints = annotation.get('keypoints', [])
    if len(keypoints) < 6:  # Need at least 2 keypoints (x, y, visibility for each)
        return None
    
    # Keypoints format: [x1, y1, v1, x2, y2, v2, ...]
    # First keypoint is Goal-Top
    goal_top = (keypoints[0], keypoints[1])
    goal_top_visible = keypoints[2] > 0
    
    # Second keypoint is Tag-Top
    tag_top = (keypoints[3], keypoints[4])
    tag_top_visible = keypoints[5] > 0
    
    if not (goal_top_visible and tag_top_visible):
        return None
    
    return goal_top, tag_top


def calculate_deviation(detected: Tuple[int, int], annotated: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between detected and annotated points."""
    dx = detected[0] - annotated[0]
    dy = detected[1] - annotated[1]
    return np.sqrt(dx**2 + dy**2)


def load_frame(frames_dir: Path, filename: str) -> Optional[np.ndarray]:
    """Load a frame from the roboflow frames directory.
    
    Args:
        frames_dir: Path to the roboflow frames directory
        filename: Name of the image file
        
    Returns:
        The loaded frame or None if failed
    """
    frame_path = frames_dir / filename
    if not frame_path.exists():
        return None
    
    frame = cv2.imread(str(frame_path))
    return frame


def visualize_keypoints(frame: np.ndarray, detected_goal_top: Tuple[int, int], 
                       detected_tag_top: Tuple[int, int],
                       undistorted_goal_top: Optional[Tuple[float, float]],
                       undistorted_tag_top: Optional[Tuple[float, float]],
                       target_size: Tuple[int, int],
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       title: str) -> None:
    """Visualize detected and annotated keypoints on the frame.
    
    Args:
        frame: Original frame (1920x1080)
        detected_goal_top: Detected goal top coordinates (scaled resolution) or (None, None)
        detected_tag_top: Detected tag top coordinates (scaled resolution) or (None, None)
        undistorted_goal_top: Undistorted annotated goal top coordinates (scaled resolution) or None
        undistorted_tag_top: Undistorted annotated tag top coordinates (scaled resolution) or None
        target_size: Target processing size (320x240)
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        title: Window title
    """
    # Undistort the frame first
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    # Scale up coordinates to original resolution for visualization
    scale_x = 1920 / target_size[0]
    scale_y = 1080 / target_size[1]
    
    # Create a copy to draw on
    vis_frame = undistorted_frame.copy()
    
    # Handle annotated goal_top (if within bounds)
    if undistorted_goal_top is not None:
        ann_goal_orig = (int(undistorted_goal_top[0] * scale_x), int(undistorted_goal_top[1] * scale_y))
        cv2.circle(vis_frame, ann_goal_orig, 8, (0, 0, 255), -1)
        cv2.putText(vis_frame, "A-Goal", (ann_goal_orig[0] + 10, ann_goal_orig[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Handle detected goal_top (if detected)
        if detected_goal_top != (None, None):
            det_goal_orig = (int(detected_goal_top[0] * scale_x), int(detected_goal_top[1] * scale_y))
            cv2.circle(vis_frame, det_goal_orig, 8, (0, 255, 0), -1)
            cv2.putText(vis_frame, "D-Goal", (det_goal_orig[0] + 10, det_goal_orig[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(vis_frame, det_goal_orig, ann_goal_orig, (255, 255, 0), 2)
        else:
            # Draw X mark for failed detection
            cv2.putText(vis_frame, "X DET FAILED", (ann_goal_orig[0] + 10, ann_goal_orig[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Annotated point outside bounds
        cv2.putText(vis_frame, "GOAL: OUTSIDE BOUNDS", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Handle annotated tag_top (if within bounds)
    if undistorted_tag_top is not None:
        ann_tag_orig = (int(undistorted_tag_top[0] * scale_x), int(undistorted_tag_top[1] * scale_y))
        cv2.circle(vis_frame, ann_tag_orig, 8, (0, 0, 255), -1)
        cv2.putText(vis_frame, "A-Tag", (ann_tag_orig[0] + 10, ann_tag_orig[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Handle detected tag_top (if detected)
        if detected_tag_top != (None, None):
            det_tag_orig = (int(detected_tag_top[0] * scale_x), int(detected_tag_top[1] * scale_y))
            cv2.circle(vis_frame, det_tag_orig, 8, (0, 255, 0), -1)
            cv2.putText(vis_frame, "D-Tag", (det_tag_orig[0] + 10, det_tag_orig[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(vis_frame, det_tag_orig, ann_tag_orig, (255, 255, 0), 2)
        else:
            # Draw X mark for failed detection
            cv2.putText(vis_frame, "X DET FAILED", (ann_tag_orig[0] + 10, ann_tag_orig[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Annotated point outside bounds
        cv2.putText(vis_frame, "TAG: OUTSIDE BOUNDS", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow(title, vis_frame)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def main():
    # Paths
    project_dir = Path(__file__).parent
    json_path = project_dir / "test_data" / "goalannotations.json"
    frames_dir = project_dir / "test_data" / "roboflow_frames"
    
    # Load annotations
    print("Loading annotations...")
    data = load_annotations(str(json_path))
    
    # Get camera calibration
    camera_matrix, dist_coeffs = get_calibration_matrices()
    target_size = (320, 240)
    
    # Scale factors for converting from original 1920x1080 to target_size
    scale_x = target_size[0] / 1920
    scale_y = target_size[1] / 1080
    
    # Statistics
    total_annotations = 0
    successful_detections = 0
    goal_top_detections = 0
    tag_top_detections = 0
    goal_top_outside = 0
    tag_top_outside = 0
    goal_top_deviations = []
    tag_top_deviations = []
    
    # Track frame data for visualization
    frame_data = []  # List of tuples: (combined_deviation, frame, detected, scaled_annotated, filename, color)
    
    # Create image_id to image mapping
    images_by_id = {img['id']: img for img in data['images']}
    
    print(f"Processing {len(data['annotations'])} annotations...")
    
    # Process each annotation
    for annotation in data['annotations']:
        # Only process Red Goal annotations (category_id == 2)
        category_id = annotation['category_id']
        if category_id != 2:
            continue
        
        total_annotations += 1
        
        # Get keypoints
        keypoints = extract_keypoints(annotation)
        if keypoints is None:
            print(f"  Annotation {annotation['id']}: Missing or invalid keypoints")
            continue
        
        annotated_goal_top, annotated_tag_top = keypoints
        
        # Get image info
        image_id = annotation['image_id']
        image_info = images_by_id.get(image_id)
        if image_info is None:
            print(f"  Annotation {annotation['id']}: Image not found")
            continue
        
        # Get the filename from the annotations
        filename = image_info.get('file_name', '')
        if not filename:
            print(f"  Annotation {annotation['id']}: No filename in image info")
            continue
        
        # Load frame from roboflow frames directory
        frame = load_frame(frames_dir, filename)
        if frame is None:
            print(f"  Annotation {annotation['id']}: Failed to load frame {filename}")
            continue
        
        # Detect keypoints
        detected_goal_top, detected_tag_top = detect_goal_keypoints(frame, category_id, camera_matrix, dist_coeffs, target_size)
        
        # Check if both detections failed
        if detected_goal_top == (None, None) and detected_tag_top == (None, None):
            print(f"  Annotation {annotation['id']}: Both detections failed for {filename}")
            continue
        
        # Undistort annotated coordinates (they are in raw image space, need to be in undistorted space)
        # Original annotated points are in 1920x1080 space
        undistorted_points = undistort_points(
            [annotated_goal_top, annotated_tag_top],
            camera_matrix,
            dist_coeffs,
            original_size=(1920, 1080),
            target_size=target_size
        )
        
        undistorted_goal_top = undistorted_points[0]
        undistorted_tag_top = undistorted_points[1]
        
        # Track outside cases
        if undistorted_goal_top is None:
            goal_top_outside += 1
        if undistorted_tag_top is None:
            tag_top_outside += 1
        
        # Check if annotated points fell outside the image after undistortion
        if undistorted_goal_top is None and undistorted_tag_top is None:
            print(f"  Annotation {annotation['id']}: Both annotated points outside image bounds after undistortion - OUTSIDE")
            continue
        
        # Calculate deviations only for successfully detected points and valid undistorted annotated points
        goal_deviation = None
        tag_deviation = None
        combined_deviation = 0
        
        if detected_goal_top != (None, None) and undistorted_goal_top is not None:
            goal_deviation = calculate_deviation(detected_goal_top, undistorted_goal_top)
            goal_top_deviations.append(goal_deviation)
            goal_top_detections += 1
            combined_deviation += goal_deviation
        
        if detected_tag_top != (None, None) and undistorted_tag_top is not None:
            tag_deviation = calculate_deviation(detected_tag_top, undistorted_tag_top)
            tag_top_deviations.append(tag_deviation)
            tag_top_detections += 1
            combined_deviation += tag_deviation
        
        successful_detections += 1
        
        # Store frame data for visualization
        color = "Blue" if category_id == 1 else "Red"
        frame_data.append((combined_deviation, frame, detected_goal_top, detected_tag_top,
                          undistorted_goal_top, undistorted_tag_top, filename, color,
                          goal_deviation, tag_deviation))
        
        print(f"  Annotation {annotation['id']} ({filename}, {color} Goal):")
        if goal_deviation is not None:
            print(f"    Goal-Top deviation: {goal_deviation:.2f} pixels")
            if undistorted_goal_top is not None:
                print(f"    Detected Goal-Top: ({int(detected_goal_top[0])}, {int(detected_goal_top[1])}), Annotated (undistorted): ({undistorted_goal_top[0]:.1f}, {undistorted_goal_top[1]:.1f})")
        elif undistorted_goal_top is None:
            print(f"    Goal-Top: Annotated point OUTSIDE image bounds")
        else:
            print(f"    Goal-Top: Detection FAILED")
        
        if tag_deviation is not None:
            print(f"    Tag-Top deviation: {tag_deviation:.2f} pixels")
            if undistorted_tag_top is not None:
                print(f"    Detected Tag-Top: ({int(detected_tag_top[0])}, {int(detected_tag_top[1])}), Annotated (undistorted): ({undistorted_tag_top[0]:.1f}, {undistorted_tag_top[1]:.1f})")
        elif undistorted_tag_top is None:
            print(f"    Tag-Top: Annotated point OUTSIDE image bounds")
        else:
            print(f"    Tag-Top: Detection FAILED")
        
        if goal_deviation is not None and tag_deviation is not None:
            print(f"    Combined deviation: {combined_deviation:.2f} pixels")
        elif goal_deviation is not None or tag_deviation is not None:
            print(f"    Partial deviation: {combined_deviation:.2f} pixels")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total annotations: {total_annotations}")
    print(f"Successful detections (at least one point): {successful_detections}")
    print(f"Detection rate (at least one point): {successful_detections/total_annotations*100:.1f}%")
    
    # Calculate valid test cases (excluding outside bounds)
    goal_top_valid = total_annotations - goal_top_outside
    tag_top_valid = total_annotations - tag_top_outside
    
    print(f"\nGoal-Top Statistics:")
    print(f"  Valid test cases: {goal_top_valid} (excluded {goal_top_outside} outside bounds)")
    print(f"  Successful detections: {goal_top_detections}/{goal_top_valid} ({goal_top_detections/goal_top_valid*100:.1f}%)" if goal_top_valid > 0 else "  No valid test cases")
    
    print(f"\nTag-Top Statistics:")
    print(f"  Valid test cases: {tag_top_valid} (excluded {tag_top_outside} outside bounds)")
    print(f"  Successful detections: {tag_top_detections}/{tag_top_valid} ({tag_top_detections/tag_top_valid*100:.1f}%)" if tag_top_valid > 0 else "  No valid test cases")
    
    if goal_top_deviations:
        print(f"\nGoal-Top Deviations:")
        print(f"  Mean: {np.mean(goal_top_deviations):.2f} pixels")
        print(f"  Median: {np.median(goal_top_deviations):.2f} pixels")
        print(f"  Std Dev: {np.std(goal_top_deviations):.2f} pixels")
        print(f"  Min: {np.min(goal_top_deviations):.2f} pixels")
        print(f"  Max: {np.max(goal_top_deviations):.2f} pixels")
        print(f"  Total: {np.sum(goal_top_deviations):.2f} pixels")
    
    if tag_top_deviations:
        print(f"\nTag-Top Deviations:")
        print(f"  Mean: {np.mean(tag_top_deviations):.2f} pixels")
        print(f"  Median: {np.median(tag_top_deviations):.2f} pixels")
        print(f"  Std Dev: {np.std(tag_top_deviations):.2f} pixels")
        print(f"  Min: {np.min(tag_top_deviations):.2f} pixels")
        print(f"  Max: {np.max(tag_top_deviations):.2f} pixels")
        print(f"  Total: {np.sum(tag_top_deviations):.2f} pixels")
    
    if goal_top_deviations and tag_top_deviations:
        combined_deviations = goal_top_deviations + tag_top_deviations
        print(f"\nCombined Deviations:")
        print(f"  Mean: {np.mean(combined_deviations):.2f} pixels")
        print(f"  Median: {np.median(combined_deviations):.2f} pixels")
        print(f"  Total: {np.sum(combined_deviations):.2f} pixels")
    
    print("="*60)
    
    # Display all frames in order
    if frame_data:
        print("\n" + "="*60)
        print("DISPLAYING FRAMES WITH LARGEST ERRORS")
        print("="*60)
        
        # Sort by combined deviation (descending)
        frame_data.sort(key=lambda x: x[0], reverse=True)
        
        num_to_show = len(frame_data)
        print(f"\nShowing all {num_to_show} frames sorted by error (largest first)...")
        print("(Press any key to move to next frame)\n")
        
        for i in range(num_to_show):
            combined_dev, frame, det_goal, det_tag, ann_goal, ann_tag, filename, color, goal_dev, tag_dev = frame_data[i]
            print(f"Frame {i+1}/{num_to_show}: {filename} ({color} Goal)")
            
            if combined_dev > 0:
                print(f"  Combined deviation: {combined_dev:.2f} pixels")
            
            if goal_dev is not None:
                print(f"  Goal-Top deviation: {goal_dev:.2f} pixels")
            else:
                print(f"  Goal-Top deviation: FAILED")
            
            if tag_dev is not None:
                print(f"  Tag-Top deviation: {tag_dev:.2f} pixels")
            else:
                print(f"  Tag-Top deviation: FAILED")
            
            title = f"Error Rank {i+1}/{num_to_show}: {filename}"
            if combined_dev > 0:
                title += f" (Combined: {combined_dev:.1f}px)"
            
            visualize_keypoints(frame, det_goal, det_tag, ann_goal, ann_tag,
                              target_size, camera_matrix, dist_coeffs, title)
        
        cv2.destroyAllWindows()
        print("\nVisualization complete.")


if __name__ == "__main__":
    main()
