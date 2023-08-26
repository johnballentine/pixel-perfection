import sys
import cv2
import numpy as np

def add_alpha_channel(input_image_path):
    # Read the input image
    bicubic_jpeg = cv2.imread(input_image_path)

    if bicubic_jpeg is None:
        print("Error: Unable to read input image.")
        return

    # Add alpha channel
    alpha_channel = np.ones(bicubic_jpeg.shape[:2], dtype=bicubic_jpeg.dtype) * 255
    final_image_with_alpha = cv2.merge([bicubic_jpeg, alpha_channel])

    # Save the processed image
    output_image_path = input_image_path.replace('.', '_out.')
    cv2.imwrite(output_image_path, final_image_with_alpha)
    print(f"Processed image saved as '{output_image_path}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input_image")
    else:
        input_image_path = sys.argv[1]
        add_alpha_channel(input_image_path)
