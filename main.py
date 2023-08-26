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

    ''' max_amount is between 0 and 100 percent. Noise might not be visible'''

    # Normalize max_amount to range from 0 to 0.1
    max_amount /= 10000

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

def random_color(saturation=30):
    # Ensure saturation is in range [0, 255]
    saturation = np.clip(saturation, 0, 255)

    # Create a HSV color with random hue, specified saturation, and maximum brightness
    hsv_color = np.uint8([[[np.random.randint(0, 180), saturation, 255]]])

    # Convert the HSV color to BGR color
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

    # Return the first element of the 1x1 image which is the color
    return bgr_color[0][0]


def generate(image,
             image_size=(128,128),
             scale_factor=2,
             max_random_offset=2,
             max_noise=40,
             jpeg_quality=40):
    # Upscale the image using nearest neighbor interpolation
    upscaled_image = upscale_nearest(image, scale_factor)

    # Increase the canvas size with specified offset
    canvas_scaled_image = increase_canvas(upscaled_image, image_size, max_random_offset)

    # Fill transparency (if any) with a random color
    color_filled_image = add_background(canvas_scaled_image, random_color())

    # Upscale the color filled image using bicubic interpolation
    color_filled_image_bicubic = upscale_bicubic(color_filled_image, scale_factor)

    # Add some noise to the image
    noisy_image = add_noise(color_filled_image_bicubic, max_noise)

    # Add JPEG artifacts
    final_image = add_jpeg_artifacts(noisy_image, jpeg_quality)

    return final_image


if __name__ == "__main__":
    input_directory = "data"
    output_directory = "./output"
    files_in_directory = os.listdir(input_directory)

    image_size = (256, 256)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for filename in files_in_directory:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Specify the input image path
            input_image_path = os.path.join(input_directory, filename)

            # Read the input image
            input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

            # Increase the canvas size of the original image with 0 offset
            input_image_canvas = increase_canvas(input_image, image_size, 0)

            # Write the original image with increased canvas size to the output directory with filename modified to include "_label"
            label_filename = rewrite_filename_with_string(filename, "label")
            cv2.imwrite(os.path.join(output_directory, label_filename), input_image_canvas)

            # Process the input image
            output_image = generate(input_image)

            # Write the output image to the output directory with filename modified to include "_processed"
            processed_filename = rewrite_filename_with_string(filename, "processed")
            cv2.imwrite(os.path.join(output_directory, processed_filename), output_image)