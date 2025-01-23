#!/bin/bash

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Find all requirements files and store them in an array
requirements_files=(requirements_*.txt)

# Check if any requirements files exist
if [ ${#requirements_files[@]} -eq 0 ]; then
    echo "No requirements_*.txt files found!"
    exit 1
fi

# Display available options
echo "Available requirements files:"
for i in "${!requirements_files[@]}"; do
    echo "[$i] ${requirements_files[$i]}"
done

# Get user selection
read -p "Select requirements file number: " selection

# Validate selection
if [ "$selection" -ge 0 ] && [ "$selection" -lt "${#requirements_files[@]}" ]; then
    selected_file="${requirements_files[$selection]}"
    echo "Installing requirements from: $selected_file"
    pip install -r "$selected_file"
else
    echo "Invalid selection!"
    exit 1
fi