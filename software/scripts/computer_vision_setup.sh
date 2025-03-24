#!/bin/bash

# Create virtual environment for Python
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt