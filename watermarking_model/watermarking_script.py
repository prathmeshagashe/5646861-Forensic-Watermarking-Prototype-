import math
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --- Configuration Parameters / Constants ---
BLOCK_SIZE = 8  # Size of the square blocks for DCT (bs)
ALPHA_FACTOR = 8  # Scaling factor for DC coefficient modification (fact)
RANDOM_SEED = 123456  # Seed for reproducible random block selection (key)
WATERMARK_SIZE = (64, 64)  # Expected dimensions of the binary watermark image

# Directory for saving results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Helper Functions ---

def convert_to_binary_watermark(image_path: str, size: tuple = WATERMARK_SIZE) -> np.ndarray:


    wm_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if wm_img is None:
        raise FileNotFoundError(f"Watermark image not found at {image_path}")

    wm_img_resized = cv2.resize(wm_img, size, interpolation=cv2.INTER_AREA)
    # Threshold to ensure binary (0 or 1) based on pixel intensity (0-255)
    binary_wm = (wm_img_resized > 127).astype(np.uint8)
    return binary_wm


def calculate_ber(original_binary_wm: np.ndarray, extracted_binary_wm: np.ndarray) -> float:
    """
    Calculates the Bit Error Rate (BER) between two binary watermark arrays.

    Args:
        original_binary_wm (np.ndarray): The original binary watermark (0s and 1s).
        extracted_binary_wm (np.ndarray): The extracted binary watermark (0s and 1s).

    Returns:
        float: The Bit Error Rate, a value between 0.0 (no errors) and 1.0 (all errors).
    """
    # Ensure extracted watermark matches the original shape for comparison
    if original_binary_wm.shape != extracted_binary_wm.shape:
        print("Warning: Watermark dimensions mismatch for BER calculation. Resizing extracted.")
        # Resize the 0-255 representation, then re-binarize
        extracted_wm_255 = (extracted_binary_wm * 255).astype(np.uint8)
        extracted_wm_resized = cv2.resize(extracted_wm_255, original_binary_wm.shape[::-1],
                                          interpolation=cv2.INTER_AREA)
        extracted_binary_wm = (extracted_wm_resized > 127).astype(np.uint8)

    mismatched_bits = np.sum(original_binary_wm != extracted_binary_wm)
    total_bits = original_binary_wm.size
    ber = mismatched_bits / total_bits
    return ber


def calculate_ncc(original_img_255: np.ndarray, extracted_img_255: np.ndarray) -> float:
    """
    Calculates the Normalized Cross-Correlation (NCC) between two grayscale images (0-255).

    Args:
        original_img_255 (np.ndarray): The original image (0-255).
        extracted_img_255 (np.ndarray): The extracted image (0-255).

    Returns:
        float: The Normalized Cross-Correlation value, typically between -1.0 and 1.0.
    """
    # Ensure images are the same size. Resize extracted to match original if needed
    if original_img_255.shape != extracted_img_255.shape:
        extracted_img_255 = cv2.resize(extracted_img_255, original_img_255.shape[::-1],
                                       interpolation=cv2.INTER_LINEAR)

    # Convert to float for calculation to avoid overflow issues
    img1 = original_img_255.astype(np.float64)
    img2 = extracted_img_255.astype(np.float64)

    # Normalize images (subtract mean and divide by standard deviation)
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-6)  # Add epsilon for numerical stability
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-6)

    # Calculate cross-correlation as the sum of element-wise products
    ncc_val = np.sum(img1_norm * img2_norm) / img1_norm.size
    return ncc_val


# --- Core Watermarking Functions ---

def embed_watermark(original_image_path: str, watermark_binary_array: np.ndarray,
                    random_seed: int, alpha_factor: int = ALPHA_FACTOR) -> tuple[np.ndarray, float, float]:
    """
    Embeds a binary watermark into the DC coefficient of an image's DCT blocks.

    Args:
        original_image_path (str): The file path to the original cover image.
        watermark_binary_array (np.ndarray): The binary watermark array (0s and 1s) to embed.
        random_seed (int): The seed for the random number generator to select block locations.
        alpha_factor (int): Scaling factor for the watermark embedding strength.

    Returns:
        tuple[np.ndarray, float, float]: A tuple containing:
            - np.ndarray: The watermarked image (uint8, 0-255).
            - float: The PSNR (Peak Signal-to-Noise Ratio) value, indicating imperceptibility.
            - float: The SSIM (Structural Similarity Index) value, indicating imperceptibility.

    Raises:
        FileNotFoundError: If the original image file does not exist.
        ValueError: If the watermark is too large for the given image.
    """
    img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    original_height, original_width = img.shape

    # Pad image dimensions to be multiples of BLOCK_SIZE for block processing
    h_pad, w_pad = 0, 0
    if original_height % BLOCK_SIZE != 0:
        h_pad = BLOCK_SIZE - (original_height % BLOCK_SIZE)
    if original_width % BLOCK_SIZE != 0:
        w_pad = BLOCK_SIZE - (original_width % BLOCK_SIZE)
    img_padded = np.pad(img, ((0, h_pad), (0, w_pad)), 'constant').astype(np.float32)

    watermarked_img_float = np.zeros_like(img_padded, dtype=np.float32)

    np.random.seed(random_seed)  # Seed for reproducible random block selection
    num_blocks_h = int(img_padded.shape[0] / BLOCK_SIZE)
    num_blocks_w = int(img_padded.shape[1] / BLOCK_SIZE)

    num_watermark_bits = watermark_binary_array.size
    total_available_blocks = num_blocks_h * num_blocks_w

    if num_watermark_bits > total_available_blocks:
        raise ValueError(f"Watermark ({num_watermark_bits} bits) is too large "
                         f"for the image ({total_available_blocks} blocks). "
                         "Reduce watermark size or increase image size.")

    # Generate unique random block indices where watermark bits will be embedded
    selected_block_indices = np.random.choice(total_available_blocks, num_watermark_bits, replace=False)

    bit_counter = 0  # Counter for current watermark bit

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            current_block_idx = i * num_blocks_w + j
            block = img_padded[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE,
                    j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
            dct_block = cv2.dct(block)

            # Check if this block is one of the randomly selected ones for embedding
            if current_block_idx in selected_block_indices:
                bit_to_embed = watermark_binary_array.flatten()[bit_counter]  # Get the bit (0 or 1)

                dc_coeff_value = dct_block[0][0] / alpha_factor

                # Round to nearest integer for parity manipulation
                rounded_dc_coeff = math.floor(dc_coeff_value + 0.5)

                # Apply parity-based embedding:
                # If bit_to_embed is 1 (white), make rounded_dc_coeff odd
                # If bit_to_embed is 0 (black), make rounded_dc_coeff even
                if bit_to_embed == 1:  # Target odd parity
                    if rounded_dc_coeff % 2 == 0:  # If even, make odd
                        rounded_dc_coeff += 1
                else:  # Target even parity
                    if rounded_dc_coeff % 2 != 0:  # If odd, make even
                        rounded_dc_coeff += 1  # Adding 1 is generally safer than subtracting to avoid going negative for small values

                # Apply modified DC coefficient back to DCT block
                dct_block[0][0] = rounded_dc_coeff * alpha_factor
                bit_counter += 1  # Move to the next watermark bit

            # Perform Inverse DCT to get the spatial domain block
            idct_block = cv2.idct(dct_block)
            watermarked_img_float[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE,
            j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] = idct_block

    # Clip values to 0-255 range and convert to uint8 for proper image representation
    watermarked_img_uint8 = np.uint8(np.clip(watermarked_img_float, 0, 255))
    original_img_uint8 = np.uint8(np.clip(img_padded, 0, 255))  # Padded original for consistent comparison

    # Calculate imperceptibility metrics using only the original (unpadded) region
    psnr_value = psnr_metric(original_img_uint8[:original_height, :original_width],
                             watermarked_img_uint8[:original_height, :original_width])
    ssim_value = ssim_metric(original_img_uint8[:original_height, :original_width],
                             watermarked_img_uint8[:original_height, :original_width], data_range=255)

    return watermarked_img_uint8, psnr_value, ssim_value


def extract_watermark(watermarked_image_path: str, original_watermark_shape: tuple,
                      random_seed: int, alpha_factor: int = ALPHA_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts a binary watermark from a watermarked image based on DC coefficient parity.

    Args:
        watermarked_image_path (str): The file path to the watermarked image (can be attacked).
        original_watermark_shape (tuple): The original (height, width) shape of the binary watermark.
        random_seed (int): The same seed used during embedding for block selection.
        alpha_factor (int): The same scaling factor used during embedding.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: The extracted watermark as a binary array (0s and 1s).
            - np.ndarray: The extracted watermark as a 0-255 grayscale image (for visual saving).

    Raises:
        FileNotFoundError: If the watermarked image file does not exist.
    """
    img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {watermarked_image_path}")

    # Re-apply padding using original image dimensions (if any was applied during embedding)
    original_height_img, original_width_img = img.shape
    h_pad, w_pad = 0, 0
    if original_height_img % BLOCK_SIZE != 0:
        h_pad = BLOCK_SIZE - (original_height_img % BLOCK_SIZE)
    if original_width_img % BLOCK_SIZE != 0:
        w_pad = BLOCK_SIZE - (original_width_img % BLOCK_SIZE)
    img_padded = np.pad(img, ((0, h_pad), (0, w_pad)), 'constant').astype(np.float32)

    extracted_watermark_bits = np.zeros(original_watermark_shape).flatten()

    np.random.seed(random_seed)  # Use the same seed as embedding for block selection
    num_blocks_h = int(img_padded.shape[0] / BLOCK_SIZE)
    num_blocks_w = int(img_padded.shape[1] / BLOCK_SIZE)

    num_watermark_bits_expected = original_watermark_shape[0] * original_watermark_shape[1]
    selected_block_indices = np.random.choice(num_blocks_h * num_blocks_w,
                                              num_watermark_bits_expected, replace=False)

    bit_counter = 0

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            current_block_idx = i * num_blocks_w + j
            if current_block_idx in selected_block_indices:
                block = img_padded[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE,
                        j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
                dct_block = cv2.dct(block)

                dc_coeff_value = dct_block[0][0] / alpha_factor
                rounded_dc_coeff = math.floor(dc_coeff_value + 0.5)

                # Determine bit based on parity
                if rounded_dc_coeff % 2 == 1:  # Odd parity -> 1
                    extracted_watermark_bits[bit_counter] = 1
                else:  # Even parity -> 0
                    extracted_watermark_bits[bit_counter] = 0
                bit_counter += 1

    extracted_wm_2d_binary = extracted_watermark_bits.reshape(original_watermark_shape)
    # Convert binary (0/1) to 0-255 for visual representation and saving as image
    extracted_wm_255 = (extracted_wm_2d_binary * 255).astype(np.uint8)
    return extracted_wm_2d_binary, extracted_wm_255


# --- Attack Functions ---

def apply_scaling_bigger(img_arr: np.ndarray, scale_percent: int = 150) -> np.ndarray:
    """
    Scales an image up by a given percentage and then scales it back to its original size.
    Simulates a resize attack.
    """
    original_height, original_width = img_arr.shape
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)

    resized = cv2.resize(img_arr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Resize back to original dimensions to make it comparable for extraction
    resized_orig_dim = cv2.resize(resized, (original_width, original_height), interpolation=cv2.INTER_AREA)
    return resized_orig_dim


def apply_scaling_half(img_arr: np.ndarray, scale_percent: int = 50) -> np.ndarray:
    """
    Scales an image down by a given percentage and then scales it back to its original size.
    Simulates a resize attack.
    """
    original_height, original_width = img_arr.shape
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)

    resized = cv2.resize(img_arr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Resize back to original dimensions
    resized_orig_dim = cv2.resize(resized, (original_width, original_height), interpolation=cv2.INTER_AREA)
    return resized_orig_dim


def apply_cut_rows(img_arr: np.ndarray, rows_to_cut: int = 100) -> np.ndarray:
    """
    Cuts rows from the top and bottom of an image and then resizes it back to original dimensions.
    Simulates a cropping and resizing attack.
    """
    h, w = img_arr.shape
    if h < 2 * rows_to_cut:
        print(
            f"Warning: Image height ({h}) is too small to cut {rows_to_cut} rows from top and bottom. No cut applied.")
        return img_arr

    cut_img = img_arr[rows_to_cut:h - rows_to_cut, :]  # Cut rows from top and bottom
    # Resize back to original size; this will distort the image significantly
    resized_orig_dim = cv2.resize(cut_img, (w, h), interpolation=cv2.INTER_AREA)
    return resized_orig_dim


def apply_average_filter(img_arr: np.ndarray, kernel_size: tuple = (3, 3)) -> np.ndarray:
    """Applies an average (box) blur filter to the image."""
    return cv2.blur(img_arr, kernel_size)


def apply_median_filter(img_arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies a median blur filter to the image."""
    return cv2.medianBlur(img_arr, kernel_size)


def add_noise(img_arr: np.ndarray, noise_type: str = "gauss",
              gaussian_sigma: float = 20.0, sp_amount: float = 0.004) -> np.ndarray:
    """
    Adds various types of noise to the image.

    Args:
        img_arr (np.ndarray): Input image array (uint8).
        noise_type (str): Type of noise ('gauss', 's&p', 'speckle').
        gaussian_sigma (float): Standard deviation for Gaussian noise in pixel intensity.
        sp_amount (float): Amount of salt & pepper noise (proportion of pixels).

    Returns:
        np.ndarray: Image array with added noise (uint8).
    """
    row, col = img_arr.shape
    noisy_img = np.copy(img_arr).astype(np.float32)  # Work with float for noise addition

    if noise_type == "gauss":
        mean = 0
        gauss = np.random.normal(mean, gaussian_sigma, (row, col))
        noisy_img += gauss
    elif noise_type == "s&p":
        s_vs_p = 0.5  # Proportion of salt vs pepper
        # Salt mode
        num_salt = np.ceil(sp_amount * img_arr.size * s_vs_p).astype(int)
        coords_salt = [np.random.randint(0, i - 1, num_salt) for i in img_arr.shape]
        noisy_img[coords_salt[0], coords_salt[1]] = 255
        # Pepper mode
        num_pepper = np.ceil(sp_amount * img_arr.size * (1. - s_vs_p)).astype(int)
        coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in img_arr.shape]
        noisy_img[coords_pepper[0], coords_pepper[1]] = 0
    elif noise_type == "speckle":
        # Multiplicative noise: img = img + img * noise
        gauss = np.random.normal(loc=0, scale=0.1, size=img_arr.shape)  # Scale of 0.1 for speckle
        noisy_img += noisy_img * gauss
    else:
        print(f"Warning: Unknown noise type '{noise_type}'. No noise applied.")

    return np.uint8(np.clip(noisy_img, 0, 255))


def apply_jpeg_compression(img_arr: np.ndarray, quality: int) -> np.ndarray:
    """
    Applies JPEG compression to a grayscale NumPy image array.

    Args:
        img_arr (np.ndarray): Input image array (uint8).
        quality (int): JPEG compression quality (0-100, higher is better quality, less compression).

    Returns:
        np.ndarray: JPEG compressed image array (uint8).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # Encode to JPEG in memory
    _, encoded_img_bytes = cv2.imencode('.jpg', img_arr, encode_param)
    # Decode back from JPEG to get the compressed image
    decoded_img = cv2.imdecode(encoded_img_bytes, cv2.IMREAD_GRAYSCALE)
    return decoded_img


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- DEBUG INFO ---
    actual_cwd = os.getcwd()
    print(f"--- DEBUG INFO ---")
    print(f"Python's Current Working Directory (CWD): {actual_cwd}")
    print(f"--- END DEBUG INFO ---")

    # Define paths to your input images relative to the script's execution directory
    # If your project structure is: YourProjectFolder/watermarking_model/watermarking_script.py
    # And images are in: YourProjectFolder/watermarking_model/original_images/
    # And: YourProjectFolder/watermarking_model/watermark_images/
    # Then these paths are correct *if you run the script from the watermarking_model directory*.

    # Example paths - Adjust these to EXACTLY match your file names and locations
    # Based on the error you're seeing, ensure the extension here (.jpeg or .jpg or .png)
    # EXACTLY matches what's on your disk.
    ORIGINAL_IMAGE_PATH = "image.jpg"  # e.g., "original_images/lena.jpg"
    WATERMARK_IMAGE_PATH = "watermark.jpg"  # e.g., "watermark_images/my_watermark.png"

    # --- Step 1: Prepare Watermark ---
    print("\n--- Preparing Watermark ---")
    try:
        original_watermark_binary = convert_to_binary_watermark(WATERMARK_IMAGE_PATH)
        print(f"Original watermark loaded. Shape: {original_watermark_binary.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Please ensure the watermark image path is correct.")
        exit()  # Exit if watermark not found

    # --- Step 2: Watermark Embedding ---
    print("\n--- Embedding Watermark ---")
    try:
        watermarked_image_array, psnr_value, ssim_value = embed_watermark(
            ORIGINAL_IMAGE_PATH, original_watermark_binary, RANDOM_SEED, ALPHA_FACTOR
        )
        watermarked_image_path_output = os.path.join(RESULTS_DIR, "Watermarked_Image.jpg")
        cv2.imwrite(watermarked_image_path_output, watermarked_image_array)
        print(f"Watermarked image saved to: {watermarked_image_path_output}")
        print(f"PSNR (Imperceptibility): {psnr_value:.2f} dB")
        print(f"SSIM (Imperceptibility): {ssim_value:.4f}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Please ensure the original image path is correct.")
        exit()
    except ValueError as e:
        print(f"ERROR: {e}")
        exit()

    # --- Step 3: Clean Watermark Extraction & Verification ---
    print("\n--- Clean Extraction (No Attacks) ---")
    extracted_wm_binary_clean, extracted_wm_255_clean = extract_watermark(
        watermarked_image_path_output, original_watermark_binary.shape, RANDOM_SEED, ALPHA_FACTOR
    )
    clean_extracted_wm_path = os.path.join(RESULTS_DIR, "Watermark_Extracted_Clean.jpg")
    cv2.imwrite(clean_extracted_wm_path, extracted_wm_255_clean)
    print(f"Clean extracted watermark saved to: {clean_extracted_wm_path}")

    ber_clean = calculate_ber(original_watermark_binary, extracted_wm_binary_clean)
    ncc_clean = calculate_ncc(original_watermark_binary * 255, extracted_wm_255_clean)
    print(f"Clean BER: {ber_clean:.4f} (should be close to 0)")
    print(f"Clean NCC: {ncc_clean:.4f} (should be close to 1)")

    # --- Step 4: Robustness Testing (Attack Simulations) ---
    print("\n--- Initiating Robustness Testing with Various Attacks ---")

    # Dictionary of attack functions and their parameters
    attack_scenarios = {
        "Scaling Bigger (150%)": lambda img: apply_scaling_bigger(img, 150),
        "Scaling Half (50%)": lambda img: apply_scaling_half(img, 50),
        "Cut 100 Rows": apply_cut_rows,
        "Average Filter (3x3)": lambda img: apply_average_filter(img, (3, 3)),
        "Median Filter (3x3)": lambda img: apply_median_filter(img, 3),
        "Gaussian Noise (Sigma 20)": lambda img: add_noise(img, "gauss", gaussian_sigma=20),
        "Salt & Pepper Noise (0.004)": lambda img: add_noise(img, "s&p", sp_amount=0.004),
        "Speckle Noise": lambda img: add_noise(img, "speckle"),
        "JPEG Compression (Quality 80)": lambda img: apply_jpeg_compression(img, 80),
        "JPEG Compression (Quality 50)": lambda img: apply_jpeg_compression(img, 50),
    }

    robustness_results = {}

    for attack_name, attack_func in attack_scenarios.items():
        print(f"\nApplying Attack: {attack_name}")

        # Apply the attack to the watermarked image
        attacked_image_array = attack_func(watermarked_image_array)

        # Define path for the attacked image output
        sanitized_attack_name = attack_name.replace(' ', '_').replace('(', '').replace(')', '').replace('&',
                                                                                                        'and').replace(
            '%', '')
        attacked_image_path_output = os.path.join(RESULTS_DIR, f"Watermarked_Attacked_{sanitized_attack_name}.jpg")
        cv2.imwrite(attacked_image_path_output, attacked_image_array)
        print(f"  Attacked image saved to: {attacked_image_path_output}")

        # Extract watermark from the attacked image
        extracted_wm_binary_attacked, extracted_wm_255_attacked = extract_watermark(
            attacked_image_path_output, original_watermark_binary.shape, RANDOM_SEED, ALPHA_FACTOR
        )
        # Define path for the extracted watermark output
        extracted_wm_path_output = os.path.join(RESULTS_DIR,
                                                f"Watermark_Extracted_Attacked_{sanitized_attack_name}.jpg")
        cv2.imwrite(extracted_wm_path_output, extracted_wm_255_attacked)
        print(f"  Extracted watermark saved to: {extracted_wm_path_output}")

        # Calculate metrics
        ber_attacked = calculate_ber(original_watermark_binary, extracted_wm_binary_attacked)
        ncc_attacked = calculate_ncc(original_watermark_binary * 255, extracted_wm_255_attacked)

        print(f"  BER: {ber_attacked:.4f}")
        print(f"  NCC: {ncc_attacked:.4f}")
        robustness_results[attack_name] = {"BER": ber_attacked, "NCC": ncc_attacked}

    print("\n--- Summary of Attack Results ---")
    print(f"{'Attack':<40} {'BER':<10} {'NCC':<10}")
    print("-" * 60)
    for attack_name, metrics in robustness_results.items():
        print(f"{attack_name:<40} {metrics['BER']:<10.4f} {metrics['NCC']:<10.4f}")

    # --- Step 5: Speed Benchmarking (Conceptual Discussion) ---
    # For robust speed benchmarking, you would typically use Python's `time` module
    # and run the embedding and extraction processes many times (e.g., in a loop)
    # to get average timings across a larger dataset.
    # Example:
    # import time
    # start_time = time.time()
    # # Call watermark_image or extract_watermark here
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.4f} seconds")