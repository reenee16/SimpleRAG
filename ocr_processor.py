import os
import io
import tempfile
import re
from typing import List, Dict, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    _OCR_OK = True
except ImportError:
    _OCR_OK = False


class OCRProcessor:
    
    def __init__(self, tesseract_path: Optional[str] = None):
        self._setup_tesseract(tesseract_path)
        self._setup_nltk()
    
    def _setup_tesseract(self, tesseract_path: Optional[str] = None):
        if not _OCR_OK:
            print("OCR libraries not available")
            return
        
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Set Tesseract path to: {tesseract_path}")
        elif os.name == 'nt': 
            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
                print(f"Set Tesseract path to: {default_path}")
        
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except Exception as e:
            print(f"Tesseract version check failed: {e}")
    
    def _setup_nltk(self):
        try:
            import nltk
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            print("NLTK resources loaded successfully")
        except Exception as e:
            print(f"NLTK setup failed: {e}")
    
    def extract_from_image(self, image_file, filename: str) -> str:
        if not _OCR_OK:
            raise RuntimeError("OCR libraries not installed. Install with: pip install pytesseract pillow")
        
        print(f"Starting OCR processing for image: {filename}")
        print(f"File size: {len(image_file.getvalue())} bytes")
        
        try:
            img = Image.open(image_file)
            print(f"Image opened successfully - Mode: {img.mode}, Size: {img.size}")
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("Image converted to RGB")
            configs_to_try = ['--psm 6','--psm 3','--psm 4','--psm 8','--psm 13']
            
            best_text = ""
            best_config = ""
            
            for config in configs_to_try:
                print(f"Trying OCR with config: {config}")
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    print(f"OCR result length: {len(text)} characters")
                    
                    if text.strip() and len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        best_config = config
                        print(f"Better result found with config: {config}")
                    
                except Exception as e:
                    print(f"OCR failed with config {config}: {e}")
            
            # Try without config
            try:
                text_no_config = pytesseract.image_to_string(img)
                if text_no_config.strip() and len(text_no_config.strip()) > len(best_text.strip()):
                    best_text = text_no_config
                    best_config = "no config"
                    print("Best result found without config")
            except Exception as e:
                print(f"OCR without config failed: {e}")
            
            print(f"Final best config: {best_config}")
            print(f"Final best text length: {len(best_text)} characters")
            
            if best_text.strip():
                print(f"OCR successful! Found {len(best_text.strip())} characters of text")
                return f"Image: {filename}\n{best_text}"
            else:
                print("No text found in image")
                return f"Image: {filename}\nNo text found in image"
        
        except Exception as e:
            print(f"Exception during OCR processing: {str(e)}")
            return f"Error processing image {filename}: {str(e)}"
    
    def extract_from_pdf(self, pdf_file, filename: str) -> str:
        if not _PDF_OK:
            raise RuntimeError("PyMuPDF not installed. Install with: pip install PyMuPDF")
        
        print(f"Starting PDF processing for: {filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name
        
        try:
            doc = fitz.open(tmp_path)
            pages = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if not text.strip() or len(text.strip()) < 10:
                    if _OCR_OK:
                        mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom for better OCR
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                        if ocr_text.strip():
                            text = ocr_text
                            print(f"Used OCR for page {page_num + 1}")
                        else:
                            print(f"No text found on page {page_num + 1} (even with OCR)")
                    else:
                        print(f"No text found on page {page_num + 1} (OCR not available)")
                
                if text.strip():
                    pages.append(f"Page {page_num + 1}:\n{text}")
            
            doc.close()
            return "\n".join(pages)
        
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
    def extract_from_file(self, uploaded_file) -> str:
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.pdf'):
            return self.extract_from_pdf(uploaded_file, uploaded_file.name)
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return self.extract_from_image(uploaded_file, uploaded_file.name)
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    
    def is_available(self) -> bool:
        return _OCR_OK
