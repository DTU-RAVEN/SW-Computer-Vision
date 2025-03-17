import cv2
import numpy as np
from tqdm import tqdm

# Disable OpenCL to prevent potential compatibility issues
cv2.ocl.setUseOpenCL(False)

def create_mosaic_with_stitcher(video_path, frame_skip=10):
    """
    Creates a 2D mosaic from a drone video using OpenCV's Stitcher in SCANS mode.
    
    This function performs the following steps:
    1. Reads frames from the input video at regular intervals
    2. Collects suitable frames for stitching
    3. Uses OpenCV's Stitcher in SCANS mode for image stitching
    4. Post-processes the result to remove black borders
    
    The SCANS mode is specifically chosen over PANORAMA mode because:
    - It's better suited for overhead/aerial imagery
    - Handles cases where there's no single projection plane
    - More robust for drone footage with varying angles
    
    :param video_path: Path to the input video file
    :param frame_skip: Number of frames to skip between captures (reduces computation)
    :return: Stitched mosaic as a NumPy array (BGR format)
    :raises: 
        - IOError: If video file cannot be opened
        - ValueError: If insufficient frames for stitching
        - RuntimeError: If stitching process fails
    """
    # Initialize video capture and verify file accessibility
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    frames = []
    frame_index = 0

    # Frame collection phase with progress tracking
    with tqdm(total=total_frames, desc="Reading frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Systematic frame sampling to reduce computational load
            # while maintaining sufficient overlap for stitching
            if frame_index % frame_skip == 0:
                # Note: Frame resizing can be enabled here if memory is a concern
                # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                frames.append(frame)
            
            frame_index += 1
            pbar.update(1)

    cap.release()
    print(f"Collected {len(frames)} frames for stitching.")

    # Verify minimum frame requirement for stitching
    if len(frames) < 2:
        raise ValueError("Not enough frames to stitch. Consider reducing 'frame_skip' value.")
    
    # Initialize stitcher in SCANS mode
    # SCANS mode is preferred for aerial footage as it handles:
    # - Multiple viewpoints
    # - Varying camera angles
    # - Non-planar scenes
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    
    # Perform the stitching operation
    print("Stitching...")
    status, stitched = stitcher.stitch(frames)
    
    if status != cv2.Stitcher_OK:
        # Common failure cases:
        # - Insufficient feature matches
        # - Too much camera movement
        # - Poor image overlap
        raise RuntimeError(f"Stitching failed with status code: {status}")
    
    print("Stitching done. Cropping the mosaic...")
    
    # Post-processing: Remove black borders from the stitched image
    # This is done by:
    # 1. Converting to grayscale
    # 2. Finding non-zero (non-black) pixels
    # 3. Computing the bounding rectangle
    # 4. Cropping to that rectangle
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        stitched = stitched[y:y+h, x:x+w]
    
    return stitched

if __name__ == "__main__":
    # Example usage and configuration
    video_file = '/Users/fredmac/Documents/DTU-FredMac/Drone/archive/Berghouse.mp4'
    
    # frame_skip parameter considerations:
    # - Lower values (e.g., 3) give more frames but increase processing time
    # - Higher values speed up processing but might miss important details
    mosaic = create_mosaic_with_stitcher(video_file, frame_skip=3)
    
    # Save the final mosaic
    cv2.imwrite("mosaic3.jpg", mosaic)
    print("Stitched mosaic saved")
