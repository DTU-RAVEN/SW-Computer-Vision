"""
OpenDroneMap (ODM) Video Processing Script

This script processes drone video footage using OpenDroneMap to generate mapping data.
It extracts frames from a video file and processes them through ODM to create
orthophotos, point clouds, and other mapping products.
"""
import os
import subprocess
import cv2
import shutil

# === User-Defined Parameters ===
"""
Configuration Parameters:
    video_path: Path to the input drone video file
    project_dir: Directory where ODM will store its output
    frame_rate: Number of frames to extract per second from video
    use_docker: Boolean flag to determine if ODM should run in Docker
    odm_settings: Dictionary containing ODM processing parameters:
        - optimize_disk_space: Removes intermediate files during processing
        - resize_to: Maximum image size in pixels
        - feature_quality: Quality of feature detection (low/medium/high)
        - min_num_features: Minimum features to extract per image
        - force_ccd: Camera CCD width in mm
        - force_focal: Camera focal length in mm
        - skip_3dmodel: Skip 3D model generation to save time
        - time_limit: Maximum processing time in seconds
"""
video_path = '/videos/mapping_video.mp4'
project_dir = "odm_project"
frame_rate = 4.0 
use_docker = True
odm_settings = {
    "optimize_disk_space": True,
    "resize_to": "2400",
    "feature_quality": "high",
    "min_num_features": "8000",
    "force_ccd": "6.17",
    "force_focal": "4.6",
    "skip_3dmodel": True,
    "time_limit": "1200",
}
# ================================

def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video file at specified intervals.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory where extracted frames will be saved
        frame_rate (float): Number of frames to extract per second
    
    Returns:
        int: Number of frames extracted
    
    The function calculates the appropriate frame interval based on the video's
    native FPS and the desired extraction rate, then saves frames as JPEG files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate) if frame_rate > 0 else 1
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_file, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    return extracted_count

def run_opendronemap(images_dir, project_dir, use_docker=True, project_name="odm_project"):
    """
    Executes OpenDroneMap processing on a set of images.
    
    Args:
        images_dir (str): Directory containing input images
        project_dir (str): Directory for ODM output
        use_docker (bool): Whether to run ODM via Docker
        project_name (str): Name of the ODM project
    
    The function constructs and executes either a Docker command or native ODM
    command based on the use_docker parameter. It applies all settings from
    the odm_settings dictionary to configure the processing parameters.
    
    Docker Command Structure:
    - Mounts input and output volumes
    - Sets project path
    - Applies all ODM parameters defined in odm_settings
    
    Native Command Structure:
    - Uses local ODM installation
    - Directly specifies image and project paths
    - Applies identical parameters as Docker version
    """
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    
    # Construct Docker command for ODM execution with given settings
    docker_command = [
        "docker", "run", "-it", "--rm",
        "-v", f"{os.path.abspath(images_dir)}:/code/images",
        "-v", f"{os.path.abspath(project_dir)}:/code/{project_name}",
        "opendronemap/opendronemap",
        "--project-path", f"/code/{project_name}",
        "--time", odm_settings["time_limit"],
        "--optimize-disk-space" if odm_settings["optimize_disk_space"] else "",
        "--resize-to", odm_settings["resize_to"],
        "--feature-quality", odm_settings["feature_quality"],
        "--min-num-features", odm_settings["min_num_features"],
        "--force-ccd", odm_settings["force_ccd"],
        "--force-focal", odm_settings["force_focal"],
        "--skip-3dmodel" if odm_settings["skip_3dmodel"] else ""
    ]
    # Filter out empty strings that occur if conditions are not met
    docker_command = list(filter(None, docker_command))
    
    native_command = [
        "odm_app.py",
        "--project-path", project_dir,
        "--images", images_dir,
        "--time", odm_settings["time_limit"],
        "--optimize-disk-space" if odm_settings["optimize_disk_space"] else "",
        "--resize-to", odm_settings["resize_to"],
        "--feature-quality", odm_settings["feature_quality"],
        "--min-num-features", odm_settings["min_num_features"],
        "--force-ccd", odm_settings["force_ccd"],
        "--force-focal", odm_settings["force_focal"],
        "--skip-3dmodel" if odm_settings["skip_3dmodel"] else ""
    ]
    native_command = list(filter(None, native_command))
    
    if use_docker:
        print("Running OpenDroneMap via Docker...")
        subprocess.run(docker_command, check=True)
    else:
        print("Running OpenDroneMap natively...")
        subprocess.run(native_command, check=True)

"""
Main Execution Flow:
1. Set up project directory structure
2. Clean up any existing project data to prevent conflicts
3. Extract frames from the input video at specified rate
4. Process extracted frames through OpenDroneMap
"""
# Main execution flow
images_dir = os.path.join(project_dir, "images")

if os.path.exists(project_dir):
    print(f"Warning: project directory '{project_dir}' already exists.")
    print("Removing old directory to avoid mixing old and new data...")
    shutil.rmtree(project_dir)

os.makedirs(images_dir, exist_ok=True)

print("Extracting frames from video...")
extracted_count = extract_frames_from_video(video_path, images_dir, frame_rate)
print(f"Extracted {extracted_count} frames.")

print("Starting OpenDroneMap processing...")
run_opendronemap(images_dir, project_dir, use_docker=use_docker)
print("OpenDroneMap processing completed.")
