import os
import cv2
import numpy as np
import argparse

def add_background(image, bgcolor):
    ''' Overlays an image with a transparent background on top of a background color.
        "image" is an OpenCV image with a transparency alpha channel
        "bgcolor" is a BGR color
        
        returns a BGR OpenCV image with no alpha channel'''

    # Split the image into BGRA channels
    B,G,R,A = cv2.split(image)

    # Create a 3-channel color image (RGB)
    fill_color_img = np.zeros([image.shape[0], image.shape[1], 3]).astype('uint8')
    fill_color_img[:,:] = bgcolor
  
    # Calculate the weighting of each pixel based on transparency
    A = A / 255.0
    fill_weight = 1 - A
    image_weight = A

    # Split fill color into channels
    fill_B, fill_G, fill_R = cv2.split(fill_color_img)
    blend_B = (B * image_weight + fill_B * fill_weight).astype(np.uint8)
    blend_G = (G * image_weight + fill_G * fill_weight).astype(np.uint8)
    blend_R = (R * image_weight + fill_R * fill_weight).astype(np.uint8)

    # Merge channels into final image but only keep BGR channels
    return cv2.merge([blend_B, blend_G, blend_R])

# ... (existing imports and add_background function)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add a background color to a transparent image.')
    parser.add_argument('input_image', type=str, help='Path to the input image with alpha channel.')
    parser.add_argument('--output_image', type=str, help='Path to save the output image.')
    parser.add_argument('--bgcolor', nargs=3, type=int, default=[255, 255, 255], help='Background color in BGR format.')
    parser.add_argument('--scale', type=float, default=4.0, help='Scaling factor for upscaling the image.')

    args = parser.parse_args()

    # Determine the output filename if not specified
    if args.output_image is None:
        base_name, ext = os.path.splitext(args.input_image)
        args.output_image = f"{base_name}_bg{ext}"

    # Read the input image
    input_image = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)

    # Check if image is loaded properly
    if input_image is None:
        print("Error: Couldn't load the input image.")
        exit(1)

    # Apply background
    output_image = add_background(input_image, tuple(args.bgcolor))

    # Upscale the image using nearest-neighbor interpolation, if scaling factor is not 1.0
    if args.scale != 1.0:
        output_image = cv2.resize(output_image, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)

    # Save the output image
    cv2.imwrite(args.output_image, output_image)
