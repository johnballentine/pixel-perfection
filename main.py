import cv2
import os
import numpy as np
import random

def rewrite_filename_with_string(filename, extra_string):
    # Split the filename into base and extension
    base, extension = os.path.splitext(filename)

    # Create the new filename with the added string and extension
    new_filename = f"{base}_{extra_string}{extension}"

    return new_filename

def upscale_nearest(image, scale_factor):
    # Upscale the image using INTER_NEAREST interpolation
    rescaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    return rescaled_image

def upscale_bicubic(image, scale_factor):
    # Upscale the image using INTER_CUBIC interpolation
    rescaled_image = cv2.resize(image, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_CUBIC)
    return rescaled_image

def upscale_bilinear(image, scale_factor):
    # Upscale the image using INTER_LINEAR interpolation
    rescaled_image = cv2.resize(image, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_LINEAR)
    return rescaled_image

def increase_canvas(image, new_dimensions, max_random_offset=0):
    enlarged_image = np.zeros((*new_dimensions, 4), dtype=np.uint8)
    
    yoff = (new_dimensions[0] - image.shape[0]) // 2
    xoff = (new_dimensions[1] - image.shape[1]) // 2
    if max_random_offset > 0:
        random_yoff = int(random.gauss(0, max_random_offset / 2))  
        random_xoff = int(random.gauss(0, max_random_offset / 2))
        yoff += random_yoff
        xoff += random_xoff

    enlarged_image[yoff:yoff+image.shape[0], xoff:xoff+image.shape[1]] = image

    return enlarged_image

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


def add_noise(image, max_amount):

    ''' max_amount is between 0 and 100 percent '''

    # Normalize max_amount to range from 0 to 0.1
    max_amount /= 4000

    # Normalize the image to [0, 1]
    image_normalized = image / 255.0

    # Generate a noise matrix with mean 0 and standard deviation of max_amount
    noise = np.random.normal(0, max_amount, image_normalized.shape)

    # Add the noise to the image
    noisy_image = image_normalized + noise

    # Clip the pixel values to be between 0 and 1
    noisy_image_clipped = np.clip(noisy_image, 0, 1)

    # Denormalize the image to original scale
    noisy_image_denormalized = (noisy_image_clipped * 255).astype(np.uint8)

    return noisy_image_denormalized

def add_jpeg_artifacts(image, quality):
    # Encode the image as JPEG with specified quality level
    retval, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]) 

    # Decode back into an OpenCV image
    jpeg_image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    return jpeg_image


if __name__ == "__main__":
    input_directory = "data"
    files_in_directory = os.listdir(input_directory)
    first_image = [file for file in files_in_directory if file.lower().endswith(('.png', '.jpg', '.jpeg'))][0]

    # Specify the input image path
    input_image_path = os.path.join(input_directory, first_image)
    scale_factor = 2
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    upscaled_image = upscale_nearest(input_image, scale_factor)
    canvas_scaled_image = increase_canvas(upscaled_image, (129, 129), 5)

    red_filled_image = add_background(canvas_scaled_image, (0,0,255))  # Red color
    red_filled_image_bicubic = add_jpeg_artifacts(upscale_bicubic(red_filled_image, 2), 20)

    output_directory = "./output"
    output_filename = rewrite_filename_with_string(first_image, "canvas_scaled_red_filled")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, red_filled_image_bicubic)