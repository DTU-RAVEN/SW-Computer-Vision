#!/bin/bash

# Create virtual environment for Python
sudo apt install python3.8 python3.8-venv python3.8-dev
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt