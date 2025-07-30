import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Global variables to store bounding box coordinates and image states
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
original_image_full_res = None # Stores the full-resolution image from PDF
display_image = None           # Stores the image currently displayed (potentially resized)
image_clone_for_display = None # Used for dynamic drawing on the *displayed* image

DPI = 300 # Set a higher DPI for better quality of the *original* image (before display resizing)
MAX_DISPLAY_DIMENSION = 1200 # Max dimension (width or height) for the displayed image

# Global variables for output settings (moved from mouse_callback parameters)
GLOBAL_OUTPUT_FILENAME = ""
GLOBAL_SHOW_OUTPUT = False # Set to True to show the cropped image

# Scaling factor:
scale_factor_x = 1.0
scale_factor_y = 1.0

def mouse_callback(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, image_clone_for_display, display_image, \
           original_image_full_res, scale_factor_x, scale_factor_y, \
           GLOBAL_OUTPUT_FILENAME, GLOBAL_SHOW_OUTPUT 

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        if display_image is not None:
            image_clone_for_display = display_image.copy()
        else:
            print("Error: display_image is None at LBUTTONDOWN. Cannot start drawing.")
            drawing = False # Prevent drawing if image is not ready

    # Handle mouse movement event while drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing: # Only draw if the mouse button is pressed down
            if image_clone_for_display is not None:
                display_image[:] = image_clone_for_display[:] 
                cv2.rectangle(display_image, (ix, iy), (x, y), (0, 255, 0), 2) # Green rectangle, thickness 2
                cv2.imshow("Image", display_image)
            else:
                print("Warning: image_clone_for_display is None during MOUSEMOVE. Stopping drawing.")
                drawing = False # Stop drawing if the clone is invalid

    # Handle mouse button up event
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        if display_image is not None:
            cv2.rectangle(display_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow("Image", display_image)
        else:
            print("Error: display_image is None at LBUTTONUP. Cannot finalize drawing.")
            return

        # Ensure coordinates are positive and in correct order for display coordinates
        disp_x1, disp_y1 = min(ix, fx), min(iy, fy)
        disp_x2, disp_y2 = max(ix, fx), max(iy, fy)

        # Convert display coordinates back to original image coordinates using scale factors
        orig_x1 = int(disp_x1 / scale_factor_x)
        orig_y1 = int(disp_y1 / scale_factor_y)
        orig_x2 = int(disp_x2 / scale_factor_x)
        orig_y2 = int(disp_y2 / scale_factor_y)

        # Validate selection dimensions before cropping
        if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
            print("Warning: Invalid or zero-sized selection. Cannot crop.")
            return 

        # Ensure coordinates are within the bounds of the original full-resolution image
        if original_image_full_res is None:
            print("Error: Original full-resolution image is not loaded. Cannot crop.")
            return

        h_orig, w_orig = original_image_full_res.shape[:2]
        orig_x1 = max(0, orig_x1)
        orig_y1 = max(0, orig_y1)
        orig_x2 = min(w_orig, orig_x2)
        orig_y2 = min(h_orig, orig_y2)

        # Crop the ORIGINAL full-resolution image using the scaled coordinates
        cropped_image = original_image_full_res[orig_y1:orig_y2, orig_x1:orig_x2]

        # Check if the cropped image is valid (not empty)
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            print("Warning: Cropped image has zero width or height. Not saving.")
            return

        cv2.imwrite(GLOBAL_OUTPUT_FILENAME, cropped_image)
        print(f"Cropped image saved as {GLOBAL_OUTPUT_FILENAME}")

        # Optionally display the cropped image
        if GLOBAL_SHOW_OUTPUT:
            display_cropped = None
            if cropped_image.shape[0] > 800 or cropped_image.shape[1] > 800:
                display_cropped, _, _ = resize_image_for_display(cropped_image, max_dim=800)
            else:
                display_cropped = cropped_image.copy() # Use a copy if not resized

            if display_cropped is not None and display_cropped.shape[0] > 0 and display_cropped.shape[1] > 0:
                cv2.imshow("Cropped Image", display_cropped)
            else:
                print("Warning: Cropped image for display is empty or invalid. Cannot show.")

def resize_image_for_display(image, max_dim=1000):
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Invalid image provided to resize_image_for_display.")
        return None, 1.0, 1.0

    h, w = image.shape[:2]
    new_h, new_w = h, w
    width_scale, height_scale = 1.0, 1.0

    if max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))

        if new_w == 0: new_w = 1
        if new_h == 0: new_h = 1

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        width_scale = new_w / w
        height_scale = new_h / h
        return resized_image, width_scale, height_scale
    else:
        return image.copy(), width_scale, height_scale # Return a copy to avoid modifying original


def annotate(pdf_path, out_filename= "extracted_table.png"):
    global original_image_full_res, display_image, scale_factor_x, scale_factor_y, DPI, MAX_DISPLAY_DIMENSION, GLOBAL_OUTPUT_FILENAME
    GLOBAL_OUTPUT_FILENAME = out_filename

    try:
        print(f"Converting {pdf_path} to image...")
        pages = convert_from_path(pdf_path, dpi=DPI, grayscale=False)

        if not pages:
            raise ValueError("No pages found in the PDF. Is the PDF empty or corrupted?")

        pil_image = pages[0] # Get the first page as a PIL Image

        original_image_full_res = np.array(pil_image)
        original_image_full_res = cv2.cvtColor(original_image_full_res, cv2.COLOR_RGB2BGR)

        if original_image_full_res is None or original_image_full_res.shape[0] == 0 or original_image_full_res.shape[1] == 0:
            raise ValueError("Failed to convert PDF page to a valid OpenCV image.")

        print(f"PDF converted to original image size: {original_image_full_res.shape[1]}x{original_image_full_res.shape[0]} pixels.")

        display_image, scale_factor_x, scale_factor_y = resize_image_for_display(
            original_image_full_res, # Pass directly, as resize_image_for_display makes a copy if not resized
            max_dim=MAX_DISPLAY_DIMENSION
        )

        if display_image is None:
            raise ValueError("Initial display image could not be generated.")

        print(f"Image resized for display to: {display_image.shape[1]}x{display_image.shape[0]} pixels.")

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please make sure the PDF file exists and the path is correct.")
        exit()
    except Exception as e:
        print(f"An error occurred during PDF conversion or image processing: {e}")
        print("Make sure Poppler is installed and accessible in your system's PATH.")
        exit()

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    print("Drag your mouse to draw a rectangle. Press 'c' or 'q' to close all windows.")

    while True:
        cv2.imshow("Image", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') or key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("All OpenCV windows closed.")
    
if __name__ == "__main__":
    pdf_path = "../pdfs/Bank_Statement_Template_1_TemplateLab.pdf" 
    
    annotate(pdf_path=pdf_path, out_filename= "table_image.png")