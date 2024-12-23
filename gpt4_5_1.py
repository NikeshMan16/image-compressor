import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import math
import struct

# Helper function to load an image
def load_image(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)

# Custom image decoder for basic JPEG
# Note: For this program, this is a simplified implementation; complex formats may need full parsing.
def decode_image(image_data):
    try:
        from io import BytesIO
        from PIL import Image
        image = Image.open(BytesIO(image_data))
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# Function to resize an image
def resize_image(image, new_width, new_height):
    """
    Resize an image to new dimensions using nearest neighbor interpolation.
    
    Args:
        image: Input image as a numpy array.
        new_width: Target width.
        new_height: Target height.

    Returns:
        Resized image as a numpy array.
    """
    height, width, channels = image.shape
    resized = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            src_x = int(j * width / new_width)
            src_y = int(i * height / new_height)
            resized[i, j] = image[src_y, src_x]

    return resized

# Function to convert an image to grayscale
def convert_to_grayscale(image):
    """
    Convert a color image to grayscale.

    Args:
        image: Input image as a numpy array.

    Returns:
        Grayscale image as a 2D numpy array.
    """
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

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
    resized_image = resize_image(image, new_w, new_h)

    # Reduce color depth
    max_value = 2 ** color_depth - 1
    compressed_image = np.round(resized_image / 255 * max_value) * (255 // max_value)

    # Resize back to original dimensions before saving
    compressed_image_resized = resize_to_original(compressed_image, image.shape)

    return compressed_image_resized.astype(np.uint8)

# Function to resize the compressed image to match the original dimensions
def resize_to_original(compressed, original_shape):
    return resize_image(compressed, original_shape[1], original_shape[0])

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
    original_gray = convert_to_grayscale(original)
    compressed_gray = convert_to_grayscale(compressed_resized)

    # SSIM calculation without skimage (manual implementation for simplicity)
    mean_orig = np.mean(original_gray)
    mean_comp = np.mean(compressed_gray)

    var_orig = np.var(original_gray)
    var_comp = np.var(compressed_gray)

    cov = np.mean((original_gray - mean_orig) * (compressed_gray - mean_comp))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim_value = ((2 * mean_orig * mean_comp + c1) * (2 * cov + c2)) / \
                 ((mean_orig ** 2 + mean_comp ** 2 + c1) * (var_orig + var_comp + c2))
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
    original_image_data = load_image(image_path)
    original_image = decode_image(original_image_data)
    if original_image is None:
        print("Error: Could not load the image.")
        return

    # Allow user to input custom PSNR and SSIM targets
    target_psnr = float(input("Enter the target PSNR (default 30): ") or 30)
    target_ssim = float(input("Enter the target SSIM (default 0.95): ") or 0.95)

    # Automatically adjust settings
    scale_factor, color_depth = auto_adjust_settings(original_image, target_psnr=target_psnr, target_ssim=target_ssim)
    print(f"Auto-adjusted settings: Scale Factor = {scale_factor}, Color Depth = {color_depth}")

    # Compress the image
    compressed_image = compress_image(original_image, scale_factor, color_depth)

    # Save the compressed image as JPEG
    output_path = os.path.splitext(image_path)[0] + "_cmp_5_1.jpg"
    compression_quality = 85  # Adjust JPEG quality (1-100, lower = more compression)
    with open(output_path, 'wb') as f:
        f.write(compressed_image)

    print(f"Compressed image saved to {output_path}")

    # Calculate and display quality metrics
    psnr_value = calculate_psnr(original_image, compressed_image)
    ssim_value = calculate_ssim(original_image, compressed_image)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.2f}")

if __name__ == "__main__":
    main()
