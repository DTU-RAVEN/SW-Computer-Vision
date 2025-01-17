import os
import subprocess
import cv2
import shutil

# === User-Defined Parameters ===
# Path to the drone video file
video_path = '/Users/fredmac/Documents/DTU-FredMac/Drone/archive/Berghouse shorter.mp4'

# Directory to store ODM output (will be created if it doesn't exist)
project_dir = "odm_project"

# Frame extraction rate: number of frames per second to extract from the video
frame_rate = 4.0  # Adjust as needed

# Flag to use Docker for running OpenDroneMap
use_docker = True

# ODM parameters (you can adjust these as needed)
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
