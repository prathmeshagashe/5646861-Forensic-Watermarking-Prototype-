**FORENSIC WATERMARKING PROTOTYPE**

This project implements a forensic watermarking prototype in python, it has been developed for results and analysis of digital forensics and deepfake detection. 
The prototype embeds a binary watermark into an image using DCT (Discrete Cosine Transform) and DC coefficient parity. 

**📂 Project Structure**
watermarking_model/
│── watermarking_script.py   # Main Python script
│── image.jpg                # Sample input image
│── watermark.jpg            # Sample watermark logo
│── original_images/         # Store input images here
│── watermark_images/        # Store watermark logos here
│── results/                 # Generated outputs (watermarked images, attacked versions, extracted watermarks)

**⚙️ Requirements**
Install the following dependices for this prototype:
```bash
pip install opencv-python scikit-image numpy
```

**▶️ STEPS TO FOLLOW**

1. Place your input image in `original_images/` (e.g., `image.jpg`).
2. Run the script:
   ```bash
   python watermarking_script.py
   ```
3. The results will be stored in the "results/" folder:
   - `Watermarked_Image.jpg` – the embedded result  
   - `Watermark_Extracted_Clean.jpg` – extracted logo without attacks  
   - `Watermarked_Attacked_*.jpg` – attacked watermarked images  
   - `Watermark_Extracted_Attacked_*.jpg` – extracted watermarks under attacks
  
**📌 Notes**
- Change paths in `watermarking_script.py` to use your own input/watermark images.  
- Make sure the images are in grayscale (`cv2.IMREAD_GRAYSCALE` is used).  
- Watermark size is to be `64x64` bits (configurable in code).
