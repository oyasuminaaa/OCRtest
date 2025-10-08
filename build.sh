#!/usr/bin/env bash

# This script is executed by Render to set up the Python environment.

# 1. Install Python dependencies from requirements.txt
echo "--- Installing Python dependencies from requirements.txt ---"
pip install -r requirements.txt

# Note: tesseract-ocr and related libraries are now installed via the 'apt' buildpack
# defined in the render.yaml file.
echo "--- Build process finished successfully. ---"