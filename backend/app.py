import pytesseract
import os
import io
import base64
import json
import time
import uuid
from PIL import Image, Image as PILImage 
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np 
import cv2
import requests

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

GEMINI_API_KEY = "AIzaSyB4niyedJwvvofav1ZaUHIwLrbclc3bPPA" 
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

app = Flask(__name__, static_folder='../frontend')
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_ocr(image_path, lang):
    
    if lang == 'eng':
        custom_config = r'--oem 1 --psm 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!@#$%^&*()-_+=/'
    elif lang == 'tha':
        custom_config = r'--oem 1 --psm 1'
    else:
        custom_config = r'--oem 1 --psm 3'
        
    try:
        img = PILImage.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
        return text
    except pytesseract.TesseractError as e:
        return f"Tesseract Error: {e}"
    except Exception as e:
        return f"General OCR Error: {e}"

def analyze_ocr_text(ocr_text):
    system_prompt = (
        "You are an expert document analysis system. Your task is to process the provided OCR text from a cropped document "
        "and structure the key information into a clean JSON format. "
        "The output language for both 'sectionTitle' and 'content' MUST match the primary language of the OCR text (Thai or English). Do NOT translate the content; simply extract it. 'sectionTitle' must be descriptive in the same language as the content."
    )

    user_query = (
        "Analyze the following OCR text from a document. Extract and categorize the key information "
        "(e.g., recipient, invoice date, total amount, checklist items, key questions, etc.) "
        "The document is likely a receipt, invoice, or checklist. If no key information is found, return an empty array for sections. "
        "Only respond with the raw JSON object, do not include any explanatory text or markdown formatting (e.g., ```json)."
        f"OCR Text:\n\n---\n\n{ocr_text}"
    )

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "sections": {
                "type": "ARRAY",
                "description": "An array of categorized information sections.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "sectionTitle": {"type": "STRING", "description": "The title of the information section. The language should match the content's primary language (Thai or English)."},
                        "content": {"type": "STRING", "description": "The extracted key details and facts related to the sectionTitle, summarized concisely, matching the content's original language (Thai or English)."}
                    },
                    "required": ["sectionTitle", "content"]
                }
            }
        },
        "required": ["sections"]
    }

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": { 
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    max_retries = 3
    delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()

            result = response.json()
            json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')

            if json_string.startswith("```json"):
                json_string = json_string.split('\n', 1)[-1]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            
            data = json.loads(json_string)
            return data.get("sections", [])

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            else:
                return {
                    "error": f"HTTP Error: {e.response.status_code}",
                    "message": f"AI API call failed. Status: {e.response.status_code}. Response: {e.response.text}"
                }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            else:
                return {
                    "error": "AI Processing Failed",
                    "message": f"An unexpected error occurred during AI analysis: {e}"
                }
    
    return {
        "error": "AI Processing Failed",
        "message": "AI analysis failed after multiple retries due to network or server issues."
    }

def preprocess_and_crop(image_path):
    
    try:
        n = np.fromfile(image_path, np.uint8)
        img_cv = cv2.imdecode(n, cv2.IMREAD_COLOR) 
    except Exception as e:
        raise ValueError(f"Error reading image file with numpy/imdecode: {e}. Check if the path is accessible and valid.")

    if img_cv is None:
        raise ValueError("Could not decode image data. Check if the file is corrupted or not a valid image format.")

    original_h, original_w, _ = img_cv.shape

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    max_area = 0

    min_area_threshold = original_h * original_w * 0.10

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area and area > min_area_threshold:
            max_area = area
            largest_contour = c
    
    cropped_img_cv_for_visual = img_cv
    
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        
        x_end = min(original_w, x + w + 2 * padding)
        y_end = min(original_h, y + h + 2 * padding)
        
        cropped_img_cv_for_visual = img_cv[y:y_end, x:x_end]
        
    
    if cropped_img_cv_for_visual.size == 0:
        cropped_img_cv_for_visual = img_cv
        
    gray_for_ocr = cv2.cvtColor(cropped_img_cv_for_visual, cv2.COLOR_BGR2GRAY)
    blurred_for_ocr = cv2.medianBlur(gray_for_ocr, 3) 
    
    processed_for_ocr_cv = cv2.adaptiveThreshold(
        blurred_for_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )

    return cropped_img_cv_for_visual, processed_for_ocr_cv

def run_analysis_pipeline(ocr_text):
    
    if "Tesseract Error" in ocr_text:
        return {
            'final_status': 'ocr_failed', 
            'message': f"AI Processing Error: {ocr_text}. Tesseract may not be installed or the Path is incorrect.",
            'ai_sections': []
        }
    elif isinstance(ocr_text, str) and not ocr_text.strip():
        return {
            'final_status': 'ocr_empty', 
            'message': "OCR found no text (blank or unrecognizable). AI analysis skipped.", 
            'ai_sections': []
        }
    else:
        ai_sections_result = analyze_ocr_text(ocr_text)
        
        if isinstance(ai_sections_result, dict) and "error" in ai_sections_result:
            return {
                'final_status': 'ai_failed', 
                'message': f"AI Processing Error: {ai_sections_result['message']}",
                'ai_sections': []
            }
        else:
            return {
                'final_status': 'success', 
                'message': 'Processing complete.', 
                'ai_sections': ai_sections_result
            }

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            _, file_extension = os.path.splitext(file.filename)
            safe_ascii_filename = str(uuid.uuid4()) + file_extension.lower()
            
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_ascii_filename)
            file.save(original_path)
            
            cropped_img_cv_for_visual, processed_for_ocr_cv = preprocess_and_crop(original_path)
            
            cropped_img_pil_visual = Image.fromarray(cv2.cvtColor(cropped_img_cv_for_visual, cv2.COLOR_BGR2RGB))
            cropped_img_pil_ocr = Image.fromarray(processed_for_ocr_cv)
            
            cropped_filename = "cropped_" + safe_ascii_filename
            cropped_path = os.path.join(app.config['RESULTS_FOLDER'], cropped_filename)
            cropped_img_pil_visual.save(cropped_path)
            
            ocr_temp_path = os.path.join(app.config['RESULTS_FOLDER'], "ocr_temp_" + safe_ascii_filename)
            cropped_img_pil_ocr.save(ocr_temp_path)
            
            ocr_lang = 'tha+eng'
            filename_lower = file.filename.lower()
            
            if 'eng' in filename_lower:
                ocr_lang = 'eng'
            elif 'ไทย' in filename_lower or 'tha' in filename_lower:
                ocr_lang = 'tha'

            ocr_text = run_ocr(ocr_temp_path, lang=ocr_lang)
            
            os.remove(ocr_temp_path)
            os.remove(original_path)
            
            analysis_results = run_analysis_pipeline(ocr_text)

            return jsonify({
                'status': analysis_results['final_status'],
                'message': analysis_results['message'],
                'ocr_text': ocr_text,
                'cropped_image': f'api/results/{cropped_filename}',
                'sections': analysis_results['ai_sections'] 
            }), 200

        except ValueError as e:
            print(f"File Read/Decode Error: {e}")
            return jsonify({'status': 'error', 'message': f'File Error: {e}'}), 500
        except Exception as e:
            print(f"General Internal Server Error: {e}")
            return jsonify({'status': 'error', 'message': f'Internal Server Error: {e}'}), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

@app.route('/api/results/<filename>')
def serve_processed_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)