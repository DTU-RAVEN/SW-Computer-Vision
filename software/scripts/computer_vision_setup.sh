# Create virtual environment for Python
cd  workspaces/computer-vision
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Activate ROS2 Foxy
cd ..
cd ..
source opt/ros/foxy/setup.bash

cd workspaces/computer-vision