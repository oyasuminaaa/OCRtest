#!/usr/bin/env bash
# exit on error
set -o errexit

# ติดตั้ง Tesseract OCR และ dependencies อื่นๆ
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

# ติดตั้ง Python packages
pip install -r requirements.txt
