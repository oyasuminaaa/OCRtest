#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "--- Installing Tesseract OCR and Thai language data ---"
# Install Tesseract, essential libraries (like libglib2.0-0, libgl1 for OpenCV)
# คำสั่ง apt-get เป็นการติดตั้งแพ็กเกจบนระบบ Linux ของ Render
apt-get update -y
apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tha \
    libglib2.0-0 \
    libgl1

echo "--- Cleaning up APT caches ---"
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "--- Installing Python dependencies from requirements.txt ---"
# เนื่องจาก rootDir ถูกตั้งค่าเป็น ./backend Render จะรันคำสั่งนี้จากโฟลเดอร์ backend
# ดังนั้นจึงต้องหาไฟล์ requirements.txt ในโฟลเดอร์ปัจจุบัน (backend)
pip install -r requirements.txt

echo "Build process finished successfully ---"