#!/bin/bash

# Set the base path to the current working directory
export BASE_PATH=$(pwd)
echo "Base path: $BASE_PATH"
export PYTHONPATH=$BASE_PATH

# Directory containing the .xyz files
XYZ_DIR="$BASE_PATH/converted_xyz"

# Directory to store output
OUTPUT_DIR="$XYZ_DIR/output"

# Model files
MODELS="$BASE_PATH/pre_trained/hands2.pt $BASE_PATH/pre_trained/hands.pt $BASE_PATH/pre_trained/manmade.pt"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each .xyz file in the directory
for xyz_file in "$XYZ_DIR"/*.xyz; do
  # Extract the base name of the file (without path and extension)
  base_name=$(basename "$xyz_file" .xyz)
  
  # Print the name of the file being processed
  echo "Processing file: $xyz_file"

  # Define a unique output directory or file name for this input file
  unique_output_dir="$OUTPUT_DIR/${base_name}_output"
  mkdir -p "$unique_output_dir"

  # Run the Python script with the specified arguments
  python -u "$BASE_PATH/orient_pointcloud.py" \
  --pc "$xyz_file" \
  --export_dir "$unique_output_dir" \
  --models $MODELS \
  --iters 10 \
  --propagation_iters 5 \
  --number_parts 30 \
  --minimum_points_per_patch 100 \
  --curvature_threshold 0.01 \
  --diffuse \
  --weighted_prop \
  --estimate_normals
done
