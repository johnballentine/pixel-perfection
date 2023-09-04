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

def upscale_bicubic(image, target_size):
    # Upscale the image to the target size using INTER_CUBIC interpolation
    rescaled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
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
        random_yoff = random.randint(-max_random_offset, max_random_offset)
        random_xoff = random.randint(-max_random_offset, max_random_offset)

        # Clip the random offsets to ensure they are within bounds
        yoff = np.clip(yoff + random_yoff, 0, new_dimensions[0] - image.shape[0])
        xoff = np.clip(xoff + random_xoff, 0, new_dimensions[1] - image.shape[1])

    enlarged_image[yoff:yoff+image.shape[0], xoff:xoff+image.shape[1]] = image

    return enlarged_image

def rotate_image(image, angle):
    ''' Rotate an image "angle" degrees.
        "image" is the image to rotate.
        
        returns the rotated image'''
    
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Define the center of the image
    center = (w // 2, h // 2)
    
    # Perform rotation
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated


def add_background(image, bgcolor, angle=2):
    ''' Overlays an image with a transparent background on top of a background color.
        "image" is an OpenCV image with a transparency alpha channel
        "bgcolor" is a BGR color
        "angle" is the degree to which the image should be rotated
        
        returns a BGR OpenCV image with no alpha channel'''
    
    # First, rotate the image if angle is not 0
    if angle != 0:
        image = rotate_image(image, generate_skewed_random(min_val=(angle * -1), max_val=angle))
    
    # Split the image into BGRA channels
    B, G, R, A = cv2.split(image)

    # Create a 3-channel color image (RGB)
    fill_color_img = np.zeros([image.shape[0], image.shape[1], 3]).astype('uint8')
    fill_color_img[:, :] = bgcolor

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

def gaussian_blur(image, kernel_size=(9, 9), sigma=1.5):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def sharpen(image, alpha=7.0):
    sharpening_filter = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, sharpening_filter)
    
    # Blend the original and the sharpened image based on alpha
    blended = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
    
    return blended

def add_alpha_channel_if_missing(image):
    # Check if alpha channel exists
    if image.shape[2] == 3:
        # Add an alpha channel filled with 255 (opaque)
        alpha_channel = np.ones(image.shape[:2], dtype=image.dtype) * 255
        image = cv2.merge((image, alpha_channel))
    return image

def generate_skewed_random(min_val: int, max_val: int) -> int:
    # Skews a random number towards the max or the min

    # Generate a random number between 0 and 1
    random_val = random.random()
    
    # Apply the piecewise function
    if random_val < 0.5:
        transformed_val = random_val ** 2 / 2
    else:
        transformed_val = 1 - ((1 - random_val) ** 2) / 2

    # Map to the desired range
    return int(min_val + (max_val - min_val) * transformed_val)

def alter_vividness(image: np.ndarray, vividness: int = 10) -> np.ndarray:
    if len(image.shape) != 3 or image.shape[2] != 4:
        raise ValueError("Input should be a 3D numpy.ndarray with 4 channels (BGRA).")

    # Add randomness to vividness
    random_shift = generate_skewed_random(-(vividness), vividness)
    vividness += random_shift

    # Ensure vividness stays within 0-100
    vividness = max(100, min(50, vividness))

    # Create an empty array for the output image
    output_image = np.zeros_like(image)

    # Apply alteration to each color channel
    for channel in range(3):  # BGRA, so loop through B, G, R
        avg_value = np.mean(image[:, :, channel])
        output_image[:, :, channel] = np.clip(
            avg_value + (image[:, :, channel] - avg_value) * (1 + vividness / 100.0),
            0, 255
        ).astype(np.uint8)

    # Keep alpha channel unchanged
    output_image[:, :, 3] = image[:, :, 3]

    return output_image

def warp(image: np.ndarray, frequency: float = 1.0, amplitude: float = 2.0) -> np.ndarray:
    # Get the shape of the image
    rows, cols, _ = image.shape

    # Generate x and y coordinates
    x_indices, y_indices = np.indices((rows, cols))

    # Apply the ripple by modifying the y-coordinates based on a sinusoidal function of the x-coordinates
    y_indices_mod = y_indices + amplitude * np.sin(2 * np.pi * frequency * x_indices / cols)

    # Make sure we don't exceed the boundary of the image
    y_indices_mod = np.clip(y_indices_mod, 0, rows-1).astype(np.float32)

    # Prepare the final map
    map_y, map_x = np.broadcast_arrays(y_indices_mod, x_indices.astype(np.float32))

    # Remap the image
    output_image = cv2.remap(image, map_y, map_x, interpolation=cv2.INTER_LINEAR)

    return output_image

def generate(image,
             image_size=(128, 128),
             scale_factor=2,
             buffer_pixels=2,
             jpeg_quality=20):
    
    upscaled = upscale_nearest(image, scale_factor)
    padded = increase_canvas(upscaled, (34, 34), 7)
    bicubic_upscaled = upscale_bicubic(padded, (256,256))
    bicubic_blur = gaussian_blur(bicubic_upscaled)
    bicubic_vividness = alter_vividness(bicubic_blur)
    bicubic_background = add_background(bicubic_vividness, random_color())
    bicubic_warp = warp(bicubic_background)
    bicubic_jpeg = add_jpeg_artifacts(bicubic_warp, jpeg_quality)

    # Add alpha channel
    #alpha_channel = np.ones(bicubic_jpeg.shape[:2], dtype=bicubic_jpeg.dtype) * 255
    #final_image_with_alpha = cv2.merge([bicubic_jpeg, alpha_channel])

    return bicubic_jpeg


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
            #input_image_upscaled = upscale_nearest(input_image, 16)

            # Write the original image with increased canvas size to the output directory with filename modified to include "_label"
            label_filename = rewrite_filename_with_string(filename, "label")
            cv2.imwrite(os.path.join(output_directory, label_filename), input_image)

            # Process the input image
            output_image = generate(input_image)

            # Write the output image to the output directory with filename modified to include "_input"
            processed_filename = rewrite_filename_with_string(filename, "input")
            cv2.imwrite(os.path.join(output_directory, processed_filename), output_image)