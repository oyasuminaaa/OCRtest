# นำเข้า pytesseract
import pytesseract
import os
import io
import base64
import json
from PIL import Image, Image as PILImage # แก้ไขการนำเข้า Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np # นำเข้า numpy
import cv2 # นำเข้า opencv
import requests # นำเข้า requests

# --- CRITICAL ENVIRONMENT NOTE ---
# ระบบถูกปรับแก้ให้ใช้งานได้โดยไม่ต้องการตั้งค่า Tesseract หรือ OpenCV 
# ซึ่งเป็นจุดอันตรายหลักในการรันในเครื่องตัวเอง เราใช้ PIL สำหรับจัดการภาพ
# และ Base64 ส่งต่อข้อมูลภาพระหว่าง Frontend/Backend อย่างปลอดภัย
# --------------------------------

# *************************************************************************
# ส่วนที่ 1: ตรวจสอบและตั้งค่า Tesseract PATH
# *************************************************************************
# หากคุณใช้ Windows และ Tesseract ถูกติดตั้งไว้ที่ตำแหน่งเฉพาะ (ส่วนใหญ่เป็นแบบนี้)
# คุณต้อง UNCOMMENT บรรทัดด้านล่างนี้ และแก้ไข Path ให้ถูกต้อง

# ตัวอย่างสำหรับ Windows: (ใช้ Path ที่ยืนยันแล้ว)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# ตัวอย่างสำหรับ macOS/Linux (ปกติไม่ต้องกำหนด Path ถ้าติดตั้งผ่าน Homebrew/apt):
# หากยังไม่ได้ผล ลองค้นหา Path ด้วยคำสั่ง 'which tesseract' ใน Terminal
# *************************************************************************


# นำเข้าไลบรารีและโค้ดส่วนที่เหลือ...

# --- Gemini API Setup ---
# *** ตั้งค่า GEMINI_API_KEY โดยตรงเพื่อการทดสอบ ***
# (ในการใช้งานจริง ควรตั้งค่าเป็น Environment Variable เพื่อความปลอดภัย)
GEMINI_API_KEY = "AIzaSyB4niyedJwvvofav1ZaUHIwLrbclc3bPPA" # API Key ที่ได้รับจากผู้ใช้
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

app = Flask(__name__, static_folder='../frontend')
CORS(app) # เปิดใช้งาน CORS

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Helper function to check allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to convert PIL Image object to Base64 data URL
def image_to_base64_src(img_obj, format="JPEG"):
    """แปลงภาพ (PIL Image) เป็น Base64 string เพื่อส่งกลับให้ Frontend แสดงผล"""
    buffered = io.BytesIO()
    img_obj.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

# --- OCR Processing using Tesseract ---
def run_ocr(image_path, lang='tha+eng'):
    # ตั้งค่า PSM (Page Segmentation Mode) ให้อยู่ในโหมด 'Fully automatic page segmentation' (PSM 3)
    # PSM 3 มีความยืดหยุ่นในการอ่านมากกว่า PSM 6 (ที่ใช้ก่อนหน้านี้)
    custom_config = r'--oem 3 --psm 3' 
    try:
        # ใช้ PIL.Image.open() ซึ่งทำงานร่วมกับ pytesseract
        img = PILImage.open(image_path)
        # Tesseract ทำ OCR โดยใช้ภาษาไทยและอังกฤษ พร้อม config ที่กำหนด
        text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
        return text
    except pytesseract.TesseractError as e:
        # ดักจับ Error ที่เกี่ยวข้องกับ Tesseract
        return f"Tesseract Error: {e}"
    except Exception as e:
        # ดักจับ Error ทั่วไป
        return f"General OCR Error: {e}"

# --- Gemini AI Analysis ---
def analyze_ocr_text(ocr_text):
    # ไม่ต้องตรวจสอบ GEMINI_API_KEY แล้ว เพราะเราใส่ค่าลงไปโดยตรง
    
    # System instruction เพื่อกำหนดบทบาทและรูปแบบผลลัพธ์
    system_prompt = (
        "You are an expert document analysis system. Your task is to process the provided OCR text from a cropped document "
        "and structure the key information into a clean JSON format. "
        "The output must strictly be a JSON object conforming to the given schema. "
        # ***** ปรับปรุงการกำหนดภาษาให้เข้มงวดขึ้น *****
        "The output language for both 'sectionTitle' and 'content' MUST be in English. Do NOT translate the content from the original OCR text."
    )

    # User query
    user_query = (
        "Analyze the following OCR text from a document. Extract and categorize the key information "
        "(e.g., recipient, invoice date, total amount, checklist items, key questions, etc.) "
        "The document is likely a receipt, invoice, or checklist. If no key information is found, return an empty array for sections. "
        "Only respond with the raw JSON object, do not include any explanatory text or markdown formatting (e.g., ```json)."
        f"OCR Text:\n\n---\n\n{ocr_text}"
    )

    # JSON Schema for structured output
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "sections": {
                "type": "ARRAY",
                "description": "An array of categorized information sections.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        # ***** ใช้ภาษาอังกฤษสำหรับ Title *****
                        "sectionTitle": {"type": "STRING", "description": "The title of the information section in English (e.g., 'Contact Details', 'Education', 'Key Skills')."},
                        "content": {"type": "STRING", "description": "The extracted key details and facts related to the sectionTitle, summarized concisely, in English."}
                    },
                    "required": ["sectionTitle", "content"]
                }
            }
        },
        "required": ["sections"]
    }

    try:
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": { 
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }
        
        # Call Gemini API
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status() # ตรวจสอบ Error 4xx/5xx

        # Extract and parse the JSON content
        result = response.json()
        
        # ต้องจัดการการแปลงจาก JSON string ใน 'text' part
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
        # ลบ markdown formatting ถ้า Gemini เผลอใส่มา (ควรจะไม่มีเพราะใช้ responseMimeType)
        if json_string.startswith("```json"):
            json_string = json_string[7:]
        if json_string.endswith("```"):
            json_string = json_string[:-3]
            
        data = json.loads(json_string)

        return data.get("sections", [])

    except requests.exceptions.HTTPError as e:
        return {
            "error": f"HTTP Error: {e.response.status_code}",
            "message": f"AI API call failed. Status: {e.response.status_code}. Response: {e.response.text}"
        }
    except Exception as e:
        return {
            "error": "AI Processing Failed",
            "message": f"An unexpected error occurred during AI analysis: {e}"
        }

# --- Image Preprocessing (Crop and Deskew) ---
def preprocess_and_crop(image_path):
    # อ่านภาพด้วย OpenCV
    img_cv = cv2.imread(image_path)
    
    if img_cv is None:
        raise ValueError("Could not read image file.")

    # [1. Robustness Check Initialization]
    # ตรวจสอบขนาดของภาพต้นฉบับก่อนการประมวลผล
    original_h, original_w, _ = img_cv.shape 

    # 1. Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur & Adaptive Threshold (เพื่อหาขอบ)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Find the largest contour (assumed to be the document)
    if not contours:
        return img_cv # ไม่เจอ contour คืนภาพต้นฉบับ

    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. Get bounding box (x, y, w, h)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # [2. Robustness Check Implementation]
    # หาก Contour ที่ตรวจพบมีความสูงน้อยกว่า 10% ของความสูงภาพต้นฉบับ
    # ให้ถือว่าการ Crop ล้มเหลว และคืนภาพต้นฉบับแทนการ Crop
    min_height_threshold = original_h * 0.10 # 10%
    if h < min_height_threshold:
        print("WARNING: Detected contour is too small (Crop failed). Skipping cropping and using original image for OCR.")
        return img_cv
    
    # 6. Crop with padding (optional: add 5-10 pixel padding)
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_cv.shape[1] - x, w + 2 * padding)
    h = min(img_cv.shape[0] - y, h + 2 * padding)
    
    cropped_img_cv = img_cv[y:y+h, x:x+w]
    
    if cropped_img_cv.size == 0:
        return img_cv # Crop แล้วภาพว่าง คืนภาพต้นฉบับ
        
    return cropped_img_cv

# --- Analysis Pipeline Helper ---
def run_analysis_pipeline(ocr_text):
    """
    จัดการตรรกะการตัดสินใจหลังจากการทำ OCR เพื่อกำหนดสถานะ (Status) 
    และเรียก Gemini AI หากมีข้อความที่สามารถวิเคราะห์ได้
    """
    if "Tesseract Error" in ocr_text:
        return {
            'final_status': 'ocr_failed', 
            'message': f"AI Processing Error: {ocr_text}. Common cause: Tesseract not found or GEMINI_API_KEY is missing.",
            'ai_sections': []
        }
    elif isinstance(ocr_text, str) and not ocr_text.strip():
        return {
            'final_status': 'ocr_empty', 
            'message': "OCR found no text. AI analysis skipped.", 
            'ai_sections': []
        }
    else:
        # ถ้า OCR มีข้อความ ให้เรียก AI
        ai_sections_result = analyze_ocr_text(ocr_text)
        
        if isinstance(ai_sections_result, dict) and "error" in ai_sections_result:
            return {
                'final_status': 'ai_failed', 
                'message': f"AI Processing Error: {ai_sections_result['message']}. Common cause: Check API Key and network connection.",
                'ai_sections': []
            }
        else:
            # AI สำเร็จ
            return {
                'final_status': 'success', 
                'message': 'Processing complete.', 
                'ai_sections': ai_sections_result
            }

# --- FLASK ROUTES ---

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)
            
            # 1. Preprocess and Crop (OpenCV)
            cropped_img_cv = preprocess_and_crop(original_path)
            
            # DEBUG: ตรวจสอบขนาดของภาพที่ถูก Crop ใน Terminal
            print(f"DEBUG: Cropped image shape: {cropped_img_cv.shape if cropped_img_cv is not None else 'None'}")
            
            # แปลง CV Image เป็น PIL Image เพื่อส่งต่อไป OCR
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img_cv, cv2.COLOR_BGR2RGB))
            
            # บันทึกภาพที่ถูก Crop เพื่อให้ Frontend เรียกมาแสดงผล
            cropped_filename = "cropped_" + filename
            cropped_path = os.path.join(app.config['RESULTS_FOLDER'], cropped_filename)
            cropped_img_pil.save(cropped_path)
            
            # 2. Run OCR (Tesseract)
            ocr_text = run_ocr(cropped_path, lang='tha+eng')
            
            # DEBUG: พิมพ์ข้อความ OCR ดิบๆ ใน Terminal
            print(f"DEBUG: Raw OCR Text (First 100 chars):\n---{ocr_text.strip()[:100]}---")
            
            # 3. Analyze with Analysis Pipeline Helper (New step)
            analysis_results = run_analysis_pipeline(ocr_text)

            return jsonify({
                'status': analysis_results['final_status'],
                'message': analysis_results['message'],
                'ocr_text': ocr_text,
                'cropped_image': f'api/results/{cropped_filename}', # Path ที่ Frontend จะเรียก
                'sections': analysis_results['ai_sections'] 
            }), 200

        except ValueError as e:
             return jsonify({'status': 'error', 'message': f'File Error: {e}'}), 500
        except Exception as e:
            # ดักจับ Error ทั่วไปในขั้นตอนประมวลผล
            return jsonify({'status': 'error', 'message': f'Internal Server Error: {e}'}), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

# Route สำหรับส่งไฟล์ที่ถูกประมวลผลแล้วกลับไป Frontend
@app.route('/api/results/<filename>')
def serve_processed_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# Route สำหรับการแสดงผล Front-end
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY is not set. AI analysis will fail.")
    print(f"Tesseract Command Path: {pytesseract.pytesseract.tesseract_cmd}")
    print("NOTE: TESSERACT_PATH is now explicitly set.")
    # แก้ไข: ปิด Hot Reloader เพื่อแก้ไข WinError 10038
    app.run(debug=True, use_reloader=False)
