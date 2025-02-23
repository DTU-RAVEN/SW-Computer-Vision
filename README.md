# DTU RAVEN's Computer Vision repository

## Info

This repository contains the scripts needed for object detection on DTU RAVEN's drone, designated for competing in SUAS 2025. The modules use ROS2 for information sharing between them.

## Setup

After cloning the repository, press `CTRL + SHIFT + B` and select `Setup Virtual Environment` in the dropdown menu. This will create a virtual python environment and download and install the pip packages required for this repository.

## Run software

To run the software, press `CTRL + SHIFT + B` and select `Run` in the drowdown menu.

## Folder structure
.
├─ software/
│  ├─ scripts/
│  │  └─ (Shell scripts for setup, linting, and more)
│  └─ src/
│     ├─ application/
│     ├─ config/
│     ├─ videos/
│     └─ (Python modules for camera, object detection, mapping, etc.)
├─ README.md
└─ requirements.txt
