import cv2
import pytesseract
import numpy as np

# Optional: Set tesseract path if not in environment
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary inverse for white text on black background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, binary

def detect_lines(binary_img):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary_img.shape[0] // 40))
    vertical_lines = cv2.erode(binary_img, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary_img.shape[1] // 40, 1))
    horizontal_lines = cv2.erode(binary_img, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)

    table_mask = cv2.add(vertical_lines, horizontal_lines)
    return table_mask

def find_table_cells(table_mask):
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 20:  # filter out noise
            cells.append((x, y, w, h))
    cells = sorted(cells, key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right
    return cells

def extract_text_from_cells(cells, image, draw_boxes=False):
    results = []
    for (x, y, w, h) in cells:
        cell_img = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cell_img, config="--psm 7").strip()
        results.append(((x, y, w, h), text))
        if draw_boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return results

def draw_and_save(image, results, output_path='output_with_boxes.png'):
    for ((x, y, w, h), text) in results:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imwrite(output_path, image)
    print(f"Saved debug output to: {output_path}")

# === MAIN ===
img_path = 'columns_detected.png'  # change to your actual image file path
image, binary = preprocess_image(img_path)
table_mask = detect_lines(binary)
cells = find_table_cells(table_mask)
results = extract_text_from_cells(cells, image.copy(), draw_boxes=True)

# Print cell texts
for i, ((x, y, w, h), text) in enumerate(results):
    print(f"Cell {i+1}: {text}")

# Save final output image with boxes
draw_and_save(image.copy(), results)
