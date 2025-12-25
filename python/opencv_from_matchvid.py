#!/usr/bin/env python3
import cv2
import numpy as np
import time
import sys
import threading
from typing import Tuple, List, Optional

# Import core utility functions
from keypoint_detection_functions import (
    get_calibration_matrices,
    undistort_and_scale,
    detect_color_contours,
    process_contours,
    filter_topmost_yellow_contour,
    draw_results,
    detect_yellow_in_contour_interiors,
    calculate_relative_angle,
    detect_goal_keypoints
)

# Global variables for threading
angle_lock = threading.Lock()
current_angle = None
frame_lock = threading.Lock()
display_frames = {}
tracking_color = "red"
should_exit = False
video_paused = False
skip_frames = 0

def init_video(video_path: str) -> Tuple[cv2.VideoCapture, int, int, float]:
    """Initialize the video capture with the match video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        sys.exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video resolution: {width}x{height}")
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.1f} seconds")
    
    return cap, width, height, fps


def add_frame_info(frame: np.ndarray, fps: float, width: int, height: int, label: str, 
                   frame_num: int = 0, total_frames: int = 0, timestamp: float = 0) -> None:
    """Add FPS, resolution, label, and video progress information to the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Resolution: {width}x{height}", (30, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (30, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {timestamp:.1f}s", (30, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


class VideoProcessingThread(threading.Thread):
    """Separate thread for video processing and angle calculation."""
    
    def __init__(self, cap, camera_matrix, dist_coeffs, target_size, video_fps):
        threading.Thread.__init__(self)
        self.cap = cap
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.target_size = target_size
        self.video_fps = video_fps
        self.daemon = True
        
        # Video control variables
        self.frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        
        # Processing variables
        self.MIN_CONTOUR_AREA = 400
        
    def run(self):
        """Main video processing loop."""
        global current_angle, display_frames, tracking_color, should_exit, video_paused, skip_frames
        
        print("Video processing thread started")
        
        while not should_exit:
            if video_paused:
                time.sleep(0.1)
                continue
            
            # Handle skip frames request
            if skip_frames > 0:
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_pos = min(current_pos + skip_frames, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                self.current_frame_num = int(new_pos)
                skip_frames = 0
                print(f"Skipped to frame {self.current_frame_num}")
                
            # Capture frame from video
            ret, frame = self.cap.read()
            if not ret:
                # End of video - loop back to beginning
                print("End of video reached, looping...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_num = 0
                continue
                
            self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = self.current_frame_num / self.video_fps
            
            # Undistort and scale frame
            processed_frame = undistort_and_scale(frame, self.camera_matrix, self.dist_coeffs, self.target_size)
            red and blue contours for yellow detection
            red_contours_full, _ = detect_color_contours(processed_frame, "red")
            blue_contours_full, _ = detect_color_contours(processed_frame, "blue")processed_frame, "red")
            blue_contours_full, _ = detect_color_contours
            # Process yellow contours within full-frame red/blue interiors
            yellow_contours, yellow_filtered = detect_yellow_in_contour_interiors(processed_frame, red_contours_full, blue_contours_full)
            # Filter to keep only the topmost yellow contour
            filtered_yellow_contours = filter_topmost_yellow_contour(yellow_contours)
            valid_yellow_contours, yellow_highest = process_contours(filtered_yellow_contours, 0)  # No minimum area for yellow
            draw_results(yellow_filtered, valid_yellow_contours, yellow_highest, (0, 255, 255), draw_top_line=True)
            
            # Then process red contours with region masking for display
            red_contours, red_filtered = detect_color_contours(processed_frame, "red")
            valid_red_contours, red_highest = process_contours(red_contours, self.MIN_CONTOUR_AREA)
            draw_results(red_filtered, valid_red_contours, red_highest, (0, 0, 255))
            
            # Process blue contours with region masking for display
            blue_contours, blue_filtered = detect_color_contours(processed_frame, "blue")
            valid_blue_contours, blue_highest = process_contours(blue_contours, self.MIN_CONTOUR_AREA)
            draw_results(blue_filtered, valid_blue_contours, blue_highest, (255, 0, 0))
            
            # Overlay yellow contours onto red and blue filtered streams
            draw_results(red_filtered, valid_yellow_contours, yellow_highest, (0, 255, 255), draw_top_line=True)
            draw_results(blue_filtered, valid_yellow_contours, yellow_highest, (0, 255, 255), draw_top_line=True)
            
            # Add information to frames
            width, height = self.target_size
            
            # Calculate angle for tracked color and update global variable
            with angle_lock:
                if tracking_color == "red" and red_highest is not None:
                    point, _ = red_highest
                    current_angle = calculate_relative_angle(point, width)
                elif tracking_color == "blue" and blue_highest is not None:
                    point, _ = blue_highest
                    current_angle = calculate_relative_angle(point, width)
                else:
                    current_angle = None
            
            # Update display frames for main thread
            with frame_lock:
                display_frames = {
                    'raw': cv2.resize(frame, self.target_size),
                    'processed': processed_frame,
                    'red_filtered': red_filtered,
                    'blue_filtered': blue_filtered,
                    'yellow_filtered': yellow_filtered,
                    'frame_info': {
                        'current_frame': self.current_frame_num,
                        'total_frames': self.total_frames,
                        'timestamp': timestamp,
                        'angle': current_angle
                    }
                }
            
            # Control playback speed
            time.sleep(self.frame_delay)
        
        print("Video processing thread ended")

def main():
    global tracking_color, should_exit, video_paused, skip_frames
    
    # Video file path
    video_path = "/Users/rick/StudioProjects/FtcRoboController/python/test_data/matchvid.mp4"
    
    # Initialize video and calibration
    cap, orig_width, orig_height, video_fps = init_video(video_path)
    camera_matrix, dist_coeffs = get_calibration_matrices()
    target_size = (320, 240)
    
    print(f"Tracking {tracking_color} objects")
    print("Controls:")
    print("  'q' or ESC - quit")
    print("  'r' - track red objects")
    print("  'b' - track blue objects")
    print("  'SPACE' - pause/unpause video")
    print("  'f' - skip forward 5 seconds")
    print("  'F' - skip forward 30 seconds")
    print("  'l' - toggle video looping")

    # Start video processing thread
    video_thread = VideoProcessingThread(cap, camera_matrix, dist_coeffs, target_size, video_fps)
    video_thread.start()
    
    print("Main thread started - handling display and controls")
    
    try:
        while not should_exit:
            # Display frames if available
            with frame_lock:
                if display_frames:
                    # Show all streams
                    cv2.imshow('Raw Video Stream', display_frames.get('raw'))
                    cv2.imshow('Undistorted Stream', display_frames.get('processed'))
                    cv2.imshow('Red Filtered Stream', display_frames.get('red_filtered'))
                    cv2.imshow('Blue/Cyan Filtered Stream', display_frames.get('blue_filtered'))
                    cv2.imshow('Yellow Filtered Stream', display_frames.get('yellow_filtered'))
                    
                    # Print angle information if tracking something
                    frame_info = display_frames.get('frame_info', {})
                    if frame_info.get('angle') is not None:
                        angle = frame_info['angle']
                        timestamp = frame_info['timestamp']
                        print(f"Time: {timestamp:.1f}s - {tracking_color.capitalize()} object angle: {angle:.2f}°")
            
            # Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC key
                should_exit = True
                break
            elif key == ord('r'):
                tracking_color = "red"
                print("Switching to track red objects")
            elif key == ord('b'):
                tracking_color = "blue"
                print("Switching to track blue objects")
            elif key == ord(' '):  # SPACE key
                video_paused = not video_paused
                print(f"Video {'paused' if video_paused else 'resumed'}")
            elif key == ord('f'):  # Skip forward 5 seconds
                skip_frames = int(5 * video_fps)
                print(f"Skipping forward 5 seconds ({skip_frames} frames)")
            elif key == ord('F'):  # Skip forward 30 seconds
                skip_frames = int(30 * video_fps)
                print(f"Skipping forward 30 seconds ({skip_frames} frames)")
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        should_exit = True
    
    # Clean up
    print("Shutting down...")
    should_exit = True
    
    # Wait for video thread to finish
    if video_thread.is_alive():
        video_thread.join(timeout=2.0)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Video analysis ended")

if __name__ == "__main__":
    main()