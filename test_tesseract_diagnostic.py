"""
Tesseract Diagnostics - Problémamegoldás
Ellenőrzi a Tesseract telepítést és konfigurációját
"""

import os
import sys
import subprocess
from pathlib import Path

def check_tesseract_installation():
    """Ellenőrzi a Tesseract bináris telepítéseit"""
    
    print("=" * 80)
    print("TESSERACT DIAGNOSZTIKA")
    print("=" * 80)
    
    # 1. Ellenőrizd a rendszerint (PATH)
    print("\n1️⃣ TESSERACT BINÁRIS - RENDSZER PATH")
    print("-" * 80)
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Tesseract ELÉRHETŐ a rendszer PATH-ből!")
            print(result.stdout)
        else:
            print("❌ Tesseract NEM válaszol az egyéb parancsra")
            print(f"Hiba: {result.stderr}")
    except FileNotFoundError:
        print("❌ Tesseract NOT in PATH - szükséges az útvonal konfigurálása")
    except Exception as e:
        print(f"❌ Hiba a PATH ellenőrzésben: {e}")
    
    # 2. Tipikus Windows telepítési útvonalak
    print("\n2️⃣ KÖZÖNSÉGES WINDOWS TELEPÍTÉSI ÚTVONALAK")
    print("-" * 80)
    
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Tesseract-OCR\tesseract.exe",
        r"C:\Users\jfeher\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]
    
    tesseract_exe = None
    for path in common_paths:
        exists = os.path.exists(path)
        status = "✅ TALÁLT" if exists else "❌"
        print(f"{status}: {path}")
        if exists:
            tesseract_exe = path
    
    # 3. Összes Program Files keresés
    print("\n3️⃣ TELJES KERESÉS - PROGRAM FILES")
    print("-" * 80)
    
    try:
        for drive in ["C:", "D:", "E:"]:
            program_files = Path(drive) / "Program Files"
            if program_files.exists():
                for item in program_files.rglob("tesseract.exe"):
                    print(f"✅ TALÁLT: {item}")
                    tesseract_exe = str(item)
    except Exception as e:
        print(f"Keresési hiba: {e}")
    
    # 4. Python pytesseract konfigurálása
    print("\n4️⃣ PYTHON PYTESSERACT KONFIGURÁCIÓ")
    print("-" * 80)
    
    try:
        import pytesseract
        print("✅ pytesseract modul elérhető")
        
        # Aktuális beállítás
        current_cmd = pytesseract.pytesseract.pytesseract_cmd
        print(f"Aktuális pytesseract_cmd: {current_cmd if current_cmd else 'None (auto-detect)'}")
        
        # Próbálj a module-ból közvetlenül tesztelni
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ pytesseract verzió detektálva: {version}")
        except Exception as e:
            print(f"❌ pytesseract nem talál Tesseract-ot: {e}")
            
            if tesseract_exe:
                print(f"\n💡 JAVÍTÁS: Beállítása a Tesseract útvonalat...")
                pytesseract.pytesseract.pytesseract_cmd = tesseract_exe
                print(f"   Beállítva: {tesseract_exe}")
                
                try:
                    version = pytesseract.get_tesseract_version()
                    print(f"   ✅ Sikeresen detektálva: {version}")
                except Exception as e2:
                    print(f"   ❌ Még mindig hiba: {e2}")
    
    except ImportError:
        print("❌ pytesseract modul NEM telepítve!")
        print("   Telepítés: pip install pytesseract")
    
    # 5. Tesseract nyelvek
    print("\n5️⃣ TESSERACT NYELVEK")
    print("-" * 80)
    
    try:
        import pytesseract
        from PIL import Image
        import io
        
        if tesseract_exe:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_exe
        
        # Próbálj egy egyszerű OCR-t
        print("Próbálunk egy egyszerű OCR tesztet futtatni...")
        
        # Létrehozunk egy egyszerű képet
        from PIL import Image, ImageDraw, ImageFont
        
        # Egyszerű teszt kép
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Próbálj a default fonttal
            draw.text((10, 40), "Tesseract OCR Test", fill='black')
        except:
            # Ha nincs font, simán írj ki valamit
            draw.text((10, 40), "Test", fill='black')
        
        # Mentsd ideiglenesen
        test_img_path = "temp_test.png"
        img.save(test_img_path)
        
        try:
            text = pytesseract.image_to_string(img)
            print(f"✅ OCR TEST SIKERES!")
            print(f"   Felismert szöveg: {text.strip()[:100]}")
            
            # Elérhető nyelvek
            langs = pytesseract.get_languages()
            print(f"   Elérhető nyelvek: {', '.join(langs[:10])}")
        except Exception as e:
            print(f"❌ OCR teszt sikertelen: {e}")
        finally:
            if os.path.exists(test_img_path):
                os.remove(test_img_path)
    
    except Exception as e:
        print(f"OCR teszt hiba: {e}")
    
    # 6. Ajánlott megoldás
    print("\n6️⃣ AJÁNLOTT MEGOLDÁS")
    print("-" * 80)
    
    if tesseract_exe:
        print(f"Talált Tesseract: {tesseract_exe}")
        print("\nAdd ezt a .env fájlhoz:")
        print(f"TESSERACT_CMD={tesseract_exe}")
        print("\nVagy a kódban:")
        print(f"os.environ['TESSERACT_CMD'] = r'{tesseract_exe}'")
    else:
        print("❌ Tesseract NEM TELEPÍTVE!")
        print("\nTelepítés:")
        print("1. Letöltés: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Futtasd az installert (alapértelmezett: C:\\Program Files\\Tesseract-OCR)")
        print("3. Újra futtatni a tesztet")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_tesseract_installation()
