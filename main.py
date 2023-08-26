import cv2
import os
import numpy as np

def rewrite_filename_with_string(filename, extra_string):
    # Split the filename into base and extension
    base, extension = os.path.splitext(filename)
    
    # Create the new filename with the added string and extension
    new_filename = f"{base}_{extra_string}{extension}"
    
    return new_filename


def upscale_integer(image, scale_factor):
    # Get the alpha channel
    alpha = image[:, :, 3]

    # Calculate the new width and height based on the scale factor
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Resize the image and the alpha channel using nearest-neighbor interpolation
    resized_img = cv2.resize(image[:, :, :3], (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    resized_alpha = cv2.resize(alpha, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Create the resized image with the merged alpha channel
    resized_img_with_alpha = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    resized_img_with_alpha[:, :, :3] = resized_img
    resized_img_with_alpha[:, :, 3] = resized_alpha

    return resized_img_with_alpha


def increase_canvas(image, scale_factor, canvas_color=(0, 0, 0, 0)):
    # Calculate the new width and height based on the scale factor
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Create a larger canvas
    enlarged_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    enlarged_image[:, :, 3] = 0  # Set the alpha channel to 0 (transparent)

    # Calculate the position to paste the original image
    paste_x = int((new_width - image.shape[1]) / 2)
    paste_y = int((new_height - image.shape[0]) / 2)

    # Paste the original image onto the larger canvas
    enlarged_image[paste_y:paste_y + image.shape[0], paste_x:paste_x + image.shape[1], :] = image

    return enlarged_image


if __name__ == "__main__":
    # Specify the input directory containing images
    input_directory = "data"

    # Get a list of all files in the input directory
    files_in_directory = os.listdir(input_directory)

    # Find the first image in the list
    first_image = None
    for file in files_in_directory:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            first_image = file
            break

    if first_image is not None:
        # Specify the input image path
        input_image_path = os.path.join(input_directory, first_image)

        # Set the scaling factor
        scale_factor = 2

        # Load the input image
        input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

        # Upscale the image using upscale_integer function
        upscaled_image = upscale_integer(input_image, scale_factor)

        # Specify the output directory and new filename
        output_directory = "output"
        output_filename = rewrite_filename_with_string(first_image, "canvas")

        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Save the upscaled image with the specified filename and path
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, upscaled_image)

        # Test the increase_canvas function
        canvas_scaled_image = increase_canvas(upscaled_image, 1.5, canvas_color=(255, 0, 0, 255))
        canvas_output_filename = rewrite_filename_with_string(first_image, "canvas_scaled")
        canvas_output_path = os.path.join(output_directory, canvas_output_filename)
        cv2.imwrite(canvas_output_path, canvas_scaled_image)

    else:
        print("No image files found in the input directory.")