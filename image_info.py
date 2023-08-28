from PIL import Image
import sys

def get_image_info(input_file_path):
    try:
        img = Image.open(input_file_path)
        width, height = img.size
        channels = len(img.getbands())
        color_mode = img.mode
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Number of channels: {channels}")
        print(f"Color mode: {color_mode}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file_path>")
    else:
        input_file_path = sys.argv[1]
        get_image_info(input_file_path)
