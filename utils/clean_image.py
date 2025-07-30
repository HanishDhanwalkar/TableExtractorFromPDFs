# import cv2
# import numpy as np

# def clean_image(in_file="table_image.png", out_file="clean_text_only_table.png"):
#     """
#     Cleans a table image by removing horizontal and vertical lines,
#     removing small noise, and making the text darker/bolder.

#     Args:
#         in_file (str): Path to the input image file (e.g., "table_image.png").
#         out_file (str): Path to save the cleaned output image (e.g., "clean_text_only_table.png").
#     """
#     img = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE)

#     if img is None:
#         print(f"Error: Could not read image '{in_file}'. Please ensure the file exists.")
#         return
#     _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

#     # --- Remove Horizontal Lines ---
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
#     horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
#     no_horizontal = cv2.subtract(binary, horizontal_lines) # Subtract the detected horizontal lines

#     # --- Remove Vertical Lines ---
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
#     vertical_lines = cv2.morphologyEx(no_horizontal, cv2.MORPH_OPEN, vertical_kernel) # isolate vertical lines from the image
#     no_lines = cv2.subtract(no_horizontal, vertical_lines) # Subtract the detected vertical lines

#     # --- Remove Specks/Noise ---
#     kernel_clean = np.ones((2, 2), np.uint8)
#     clean = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, kernel_clean)

#     # --- Darken Text (Thicken Text) ---
#     # Define a dilation kernel. Dilation expands white regions.
#     dilate_kernel = np.ones((3, 3), np.uint8)
#     dark_text = cv2.dilate(clean, dilate_kernel, iterations=1)

#     # --- Invert Back to Original Color Scheme ---
#     final = 255 - dark_text

#     # Save the resulting image
#     cv2.imwrite(out_file, final)
#     print(f"Cleaned image saved to '{out_file}'")

# if __name__ == "__main__":
#     clean_image()


import cv2

import numpy as np



def clean_image(in_file = "table_image.png", out_file="clean_text_only_table.png"):

    img = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE)



    # Binarize - text becomes white (255), background black (0)

    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)



    # Remove horizontal lines

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    no_horizontal = cv2.subtract(binary, horizontal_lines)



    # Remove vertical lines

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    vertical_lines = cv2.morphologyEx(no_horizontal, cv2.MORPH_OPEN, vertical_kernel)

    no_lines = cv2.subtract(no_horizontal, vertical_lines)



    # remove specks/noise

    kernel_clean = np.ones((1, 1), np.uint8)

    clean = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, kernel_clean)



    # Darken text — we are still in inverted mode (white text), so no need to invert yet

    dilate_kernel = np.ones((2, 2), np.uint8)

    dark_text = cv2.dilate(clean, dilate_kernel, iterations=2)



    # Invert back — now text becomes black, background white

    final = 255 - dark_text

    cv2.imwrite(out_file, final)


if __name__ == "__main__":
    clean_image()

