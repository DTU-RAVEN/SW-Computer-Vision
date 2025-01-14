import cv2
import numpy as np

def create_mosaic_with_stitcher(video_path, frame_skip=10):
    """
    Creates a 2D mosaic from a drone video using OpenCV's Stitcher in SCANS mode.
    
    :param video_path: Path to the input video file.
    :param frame_skip: Capture 1 frame every `frame_skip` frames for stitching.
    :return: A stitched mosaic as a NumPy array (BGR image).
    """
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    frames = []
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 2. Sample frames: pick every 'frame_skip'-th frame
        if frame_index % frame_skip == 0:
            # Optionally resize if you want smaller frames
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            frames.append(frame)
        
        frame_index += 1

    cap.release()
    print(f"Collected {len(frames)} frames for stitching.")

    if len(frames) < 2:
        raise ValueError("Not enough frames to stitch. Try reducing 'frame_skip'.")
    
    # 3. Create the stitcher object in SCANS mode
    #    (If this fails, ensure you have a recent version of OpenCV contrib)
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    
    # 4. Stitch the frames
    print("Stitching...")
    status, stitched = stitcher.stitch(frames)
    
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed with status code: {status}")
    
    print("Stitching done. Cropping the mosaic...")
    
    # 5. Crop the result to remove black borders
    # Convert to grayscale to find non-zero region
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        stitched = stitched[y:y+h, x:x+w]
    
    return stitched

if __name__ == "__main__":
    # Example usage
    video_file = '/Users/fredmac/Documents/DTU-FredMac/archive/SurenenTrail.mp4'
    video_file = '/Users/fredmac/Documents/DTU-FredMac/archive/VerticalFlyOver.mp4'
    
    # Adjust 'frame_skip' as needed. 
    # If your drone footage moves quickly, you might want frame_skip=5 or less.
    # If it’s slow-moving or very large, you can try 20 or more to reduce overlap.
    mosaic = create_mosaic_with_stitcher(video_file, frame_skip=3)
    
    # 6. Save or display the result
    cv2.imwrite("advanced_stitched_mosaic.jpg", mosaic)
    print("Stitched mosaic saved as advanced_stitched_mosaic.jpg")
