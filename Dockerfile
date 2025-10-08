# ใช้ Python 3.11 เป็น base image
FROM python:3.11-slim

# กำหนด Working Directory ภายใน container
WORKDIR /app

# ติดตั้ง Tesseract OCR และลบแคชเพื่อลดขนาด image
RUN apt-get update && apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# คัดลอกไฟล์ requirements.txt เข้าไปใน container
COPY requirements.txt .

# ติดตั้ง Python packages
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดของแอปพลิเคชันทั้งหมดเข้าไป
COPY ./backend ./backend
COPY ./frontend ./frontend

# ไม่ต้องใช้ build.sh อีกต่อไป สามารถลบไฟล์นี้ทิ้งได้
# COPY build.sh . 

# กำหนดคำสั่งสำหรับรันแอปพลิเคชัน
# Render จะกำหนดค่า PORT ให้อัตโนมัติ เราจึงใช้ $PORT ที่นี่
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "backend.app:app"]
