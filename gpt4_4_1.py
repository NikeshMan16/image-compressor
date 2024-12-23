# *******************************************************************************#
#  ***************************************************************************#
#   **************** ONLY THING CHANGED IN THIS GPT4_4.PY IS **************#
#     ************** FILE SAVED IN ORIGINAL DIMENSION *********************#
# **********ONLY CHange from gpt4_4 is calculation of ************************          ************compression time and optimization in code**********************


import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim
import math
import cv2

# Helper function to load an image
def load_image(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)

# Function to compress image by reducing resolution and color depth
def compress_image(image, scale_factor=0.5, color_depth=4):
    """
    Compress the image by reducing resolution and color depth.
    Args:
        image: Input image as a numpy array.
        scale_factor: Fraction by which resolution is reduced.
        color_depth: Number of bits per channel.

    Returns:
        Compressed image as a numpy array.
    """
    # Reduce resolution
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Reduce color depth
    max_value = 2 ** color_depth - 1
    compressed_image = np.round(resized_image / 255 * max_value) * (255 // max_value)

    # Resize back to original dimensions before saving
    compressed_image_resized = resize_to_original(compressed_image, image.shape)

    return compressed_image_resized.astype(np.uint8)

# Function to resize the compressed image to match the original dimensions
def resize_to_original(compressed, original_shape):
    return cv2.resize(compressed, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

# Function to calculate PSNR
def calculate_psnr(original, compressed):
    compressed_resized = resize_to_original(compressed, original.shape)
    mse = np.mean((original - compressed_resized) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Function to calculate SSIM
def calculate_ssim(original, compressed):
    compressed_resized = resize_to_original(compressed, original.shape)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed_resized, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, compressed_gray)
    return ssim_value

# Function to auto-adjust settings for compression
def auto_adjust_settings(original_image, target_psnr=30, target_ssim=0.95):
    """
    Adjusts scale factor and color depth to maximize compression while maintaining quality.
    
    Target PSNR and SSIM values are chosen based on perceptual quality requirements:
    - PSNR >= 30 ensures minimal visible noise for most images.
    - SSIM >= 0.95 ensures structural integrity and visual similarity.

    Args:
        original_image: Input image as a numpy array.
        target_psnr: User-defined minimum acceptable PSNR (default is 30).
        target_ssim: User-defined minimum acceptable SSIM (default is 0.95).

    Returns:
        Tuple (scale_factor, color_depth) for optimal settings.
    """
    scale_factor = 1.0
    color_depth = 8

    for sf in np.arange(1.0, 0.1, -0.1):
        for cd in range(8, 1, -1):
            compressed_image = compress_image(original_image, scale_factor=sf, color_depth=cd)
            psnr_value = calculate_psnr(original_image, compressed_image)
            ssim_value = calculate_ssim(original_image, compressed_image)

            if psnr_value >= target_psnr and ssim_value >= target_ssim:
                scale_factor = sf
                color_depth = cd

    return scale_factor, color_depth

# Main function
def main():
    image_path = input("Enter the path to the image file: ")

    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load the image.")
        return

    # Allow user to input custom PSNR and SSIM targets
    target_psnr = float(input("Enter the target PSNR  (default 30): ") or 30)
    target_ssim = float(input("Enter the target SSIM (default 0.95): ") or 0.95)

    # Start timer for compression
    start_time = time.time()

    # Automatically adjust settings
    scale_factor, color_depth = auto_adjust_settings(original_image, target_psnr=target_psnr, target_ssim=target_ssim)
    print(f"Auto-adjusted settings: Scale Factor = {scale_factor}, Color Depth = {color_depth}")

    # Compress the image
    compressed_image = compress_image(original_image, scale_factor, color_depth)

    # Save the compressed image as JPEG
    output_path = os.path.splitext(image_path)[0] + "_comp_dmn.jpg"
    compression_quality = 85  # Adjust JPEG quality (1-100, lower = more compression)
    cv2.imwrite(output_path, compressed_image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])

    # End timer for compression
    end_time = time.time()
    compression_time = end_time - start_time

    print(f"Compressed image saved to {output_path}")

    # Calculate and display quality metrics
    psnr_value = calculate_psnr(original_image, compressed_image)
    ssim_value = calculate_ssim(original_image, compressed_image)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.2f}")
    print(f"Time taken for compression: {compression_time:.2f} seconds")

if __name__ == "__main__":
    main()
