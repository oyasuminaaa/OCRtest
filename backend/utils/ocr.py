import pytesseract
import cv2
import json
import requests
import os
import time
import numpy as np
from PIL import Image

def process_document_ocr(img_pil_cropped, tesseract_cmd):
    """
    1. Runs Tesseract OCR on the cropped PIL image.
    2. Sends raw text to Gemini API for structured data extraction.
    
    Args:
        img_pil_cropped (Image.Image): ภาพที่ถูกตัดแล้วในรูปแบบ PIL Image.
        tesseract_cmd (str): Path to the Tesseract executable.

    Returns:
        tuple: (raw_ocr_text, structured_sections_json_parsed, error_message)
    """
    
    raw_ocr_text = ""
    sections = []
    error_message = None

    try:
        # 1. Tesseract OCR Pre-processing
        # แปลง PIL Image เป็น Grayscale และใช้ Thresholding
        gray = img_pil_cropped.convert("L")
        # แปลงกลับเป็น OpenCV format เพื่อใช้ Thresholding ขั้นสูง (Optional)
        img_np_gray = np.array(gray)
        _, binary = cv2.threshold(img_np_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # ตั้งค่า Tesseract Command Path (ถูกตั้งใน app.py แล้ว)
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd 
        
        # ดึงข้อความ (ใช้ lang='tha+eng' เพื่อรองรับภาษาไทย)
        raw_ocr_text = pytesseract.image_to_string(binary, lang='tha+eng')

    except Exception as e:
        error_message = f"Tesseract OCR failed. Check Tesseract installation/path: {str(e)}"
        print(f"OCR Error: {error_message}")
        return raw_ocr_text, sections, error_message

    # 2. Gemini AI Analysis (LLM Call)
    if not raw_ocr_text or len(raw_ocr_text.strip()) < 10:
        return raw_ocr_text, sections, "OCR resulted in empty or very short text. Cannot proceed with AI analysis."
        
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            error_message = "GEMINI_API_KEY environment variable is not set. Cannot perform AI analysis."
            print(error_message)
            return raw_ocr_text, sections, error_message
            
        system_prompt = (
            "You are a document categorization AI. Analyze the provided OCR text and extract key information "
            "into a structured JSON format. Identify distinct logical sections (e.g., 'Invoice Header', 'Item List', 'Total Summary', 'Recipient Details')."
            "If the document is a résumé, identify 'Contact Information', 'Experience', and 'Education'."
        )
        
        user_query = f"Analyze the following OCR text and structure it into distinct sections as per the JSON schema. OCR Text:\n\n---\n{raw_ocr_text}\n---"

        # Define the JSON schema for structured output
        response_schema = {
            "type": "ARRAY",
            "description": "An array of structured sections extracted from the document text.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "sectionTitle": {"type": "STRING", "description": "A concise title for the extracted section (e.g., 'Candidate Name', 'Total Amount')."},
                    "content": {"type": "STRING", "description": "The detailed, cleaned text content belonging to this section, formatted for readability."}
                },
                "propertyOrdering": ["sectionTitle", "content"]
            }
        }
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }

        # LLM API Call with Exponential Backoff
        max_retries = 3
        delay = 1 

        for attempt in range(max_retries):
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                
                json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
                
                if json_string:
                    try:
                        sections = json.loads(json_string)
                        return raw_ocr_text, sections, None # Success
                    except json.JSONDecodeError:
                        error_message = "AI returned text, but it could not be parsed as valid JSON."
                        print(f"JSON Decode Error: {json_string}")
                        break
                else:
                    error_message = "AI analysis returned no content."
                    break

            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    error_message = "AI service is busy (Rate Limit Exceeded) after multiple retries."
                    break
            else:
                error_message = f"AI API call failed with status code {response.status_code}: {response.text}"
                break
        
        return raw_ocr_text, sections, error_message

    except Exception as e:
        error_message = f"Internal error during AI API communication: {str(e)}"
        print(f"LLM Communication Error: {error_message}")
        return raw_ocr_text, sections, error_message
