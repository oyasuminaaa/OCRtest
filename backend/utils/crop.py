import cv2
import numpy as np

def order_points(pts):
    """
    จัดเรียงจุดมุม 4 จุดให้อยู่ในลำดับ: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 1. คำนวณผลรวม (Sum) และผลต่าง (Difference) ของพิกัด
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # บนซ้าย (min sum)
    rect[2] = pts[np.argmax(s)] # ล่างขวา (max sum)
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # บนขวา (min diff)
    rect[3] = pts[np.argmax(diff)] # ล่างซ้าย (max diff)
    
    return rect

def four_point_transform(image, pts):
    """
    ทำ Perspective Transform โดยใช้จุดมุม 4 จุดที่จัดเรียงแล้ว
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # คำนวณความกว้างใหม่
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # คำนวณความสูงใหม่
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # จุดปลายทาง (Destination points)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # ทำ Perspective Transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def crop_document(img_np):
    """
    ฟังก์ชันหลักในการหาเอกสาร, ตัด, และแก้ไขมุมเอียง (Perspective Correction)
    
    Args:
        img_np (np.array): ภาพต้นฉบับในรูปแบบ NumPy Array (CV2 format)

    Returns:
        np.array: ภาพที่ถูกตัดและแก้ไขมุมเอียงแล้ว (หรือภาพต้นฉบับถ้าหาขอบไม่ได้)
    """
    
    try:
        # 1. การประมวลผลเบื้องต้น: Grayscale, Blur, Thresholding
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Thresholding และ Closing เพื่อรวมขอบ
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)) 
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 2. Canny Edge Detection
        edged = cv2.Canny(closed, 75, 200)

        # 3. ค้นหา Contour ที่ใหญ่ที่สุด
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        document_contour = None
        min_area_threshold = img_np.shape[0] * img_np.shape[1] * 0.15 

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                continue

            # หา 4 มุมโดยใช้ approxPolyDP
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) 
            
            if len(approx) == 4:
                document_contour = approx
                break
        
        # 4. ทำ Perspective Transform หรือ Fallback
        if document_contour is not None:
            # Case A: พบ 4 มุม - ทำ Perspective Transform (Ideal)
            return four_point_transform(img_np, document_contour.reshape(4, 2))
        elif len(contours) > 0:
            # Case B: ไม่พบ 4 มุม แต่พบ Contour ใหญ่ - ใช้ Bounding Box (Fallback)
            x, y, w, h = cv2.boundingRect(contours[0])
            padding = 10 
            x_p = max(0, x - padding)
            y_p = max(0, y - padding)
            w_p = min(img_np.shape[1], x + w + padding) - x_p
            h_p = min(img_np.shape[0], y + h + padding) - y_p
            
            return img_np[y_p:y_p+h_p, x_p:x_p+w_p]
        
        # Case C: Fallback - คืนภาพต้นฉบับ
        return img_np
        
    except Exception as e:
        print(f"OpenCV processing error: {e}")
        # คืนภาพต้นฉบับในกรณีที่เกิดข้อผิดพลาดในการประมวลผล
        return img_np
