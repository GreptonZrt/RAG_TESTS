"""
Test PDF Processing - OCR and Workflow Validation
Tests the PDF extraction capability on the Grepton document
"""

import sys
import os
from pathlib import Path

# Load .env configuration
from dotenv import load_dotenv
load_dotenv()

# Add workflow_parts to path
sys.path.insert(0, str(Path(__file__).parent))

from workflow_parts.data_loading import extract_text_from_pdf

def test_pdf_processing():
    """Test PDF extraction from the Grepton document"""
    
    pdf_path = r"c:\Users\jfeher\VSCodes\RAG_tests\data\Grepton_Konzorcia_SmartComm_módosítás_20240527.pdf"
    
    print("=" * 80)
    print("PDF FELDOLGOZÁS TESZT - Grepton Konzorcia SmartComm")
    print("=" * 80)
    print(f"\nPDF útvonal: {pdf_path}")
    print(f"Fájl létezik: {os.path.exists(pdf_path)}")
    
    if not os.path.exists(pdf_path):
        print("❌ Hiba: A PDF fájl nem található!")
        return False
    
    # 1. Próbálj szövegkinyerést (nem OCR)
    print("\n1️⃣ SZÖVEGKINYERÉS (PyMuPDF - direkta)")
    print("-" * 80)
    try:
        text = extract_text_from_pdf(pdf_path, use_ocr=False)
        
        if text.strip():
            print(f"✅ Szövegkinyerés sikeres!")
            print(f"Kinyert szöveg hossza: {len(text)} karakter")
            print(f"Első 500 karakter:\n{text[:500]}")
            print(f"\n...{text[len(text)-300:]}")  # Last 300 chars
            return True
        else:
            print("⚠️  Figyelmeztetés: Nem található szöveg (lehet, hogy scan-olva van)")
            
            # 2. Próbálj OCR-t
            print("\n2️⃣ OCR FELDOLGOZÁS")
            print("-" * 80)
            print("A PDF képként van mentve, OCR szükséges...")
            
            try:
                text_ocr = extract_text_from_pdf(pdf_path, use_ocr=True)
                if text_ocr.strip():
                    print(f"✅ OCR feldolgozás sikeres!")
                    print(f"OCR által kinyert szöveg hossza: {len(text_ocr)} karakter")
                    print(f"Első 500 karakter:\n{text_ocr[:500]}")
                    return True
                else:
                    print("❌ OCR is visszatért üres szöveggel")
                    return False
                    
            except ImportError as e:
                print(f"⚠️  OCR importálási hiba: {e}")
                print("Szükséges: pytesseract, Tesseract bináris, PIL")
                return False
            except Exception as e:
                print(f"❌ OCR feldolgozási hiba: {e}")
                return False
    
    except Exception as e:
        print(f"❌ Szövegkinyerési hiba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_processing()
    print("\n" + "=" * 80)
    if success:
        print("✅ A workflow képes feldolgozni a PDF-et!")
    else:
        print("❌ A workflow nem képes feldolgozni a PDF-et")
    print("=" * 80)
