# Multi-Document RAG Workflow - Setup Guide

## Overview
Az 01_simple_rag workflow mostmár támogatja az alábbi dokumentumtípusok feldolgozását:
- ✅ PDF fájlok (szöveges tartalom)
- ✅ DOCX fájlok (Word dokumentumok)
- ⚠️ Képalapú PDF-ek (OCR szükséges)

## Telepítés

### Alapfüggőségek (már telepített)
```bash
pip install PyMuPDF numpy openai requests python-dotenv
pip install python-docx  # DOCX támogatás
pip install pytesseract pillow  # OCR támogatás (opcionális)
```

### OCR támogatás (opcionális - képalapú PDF-ekhez)

Ha képeket tartalmazó PDF-eket szeretnél feldolgozni, telepítened kell a Tesseract OCR motort:

#### Windows
1. Töltsd le az ingyenes Tesseract telepítőt: https://github.com/UB-Mannheim/tesseract/wiki
2. Futtasd a telepítőt az alapértelmezett helyre: `C:\Program Files\Tesseract-OCR`
3. Állítsd be a PATH környezeti változót:
```bash
# Vagy add hozzá manuálisan a pytesseract kódjában:
import pytesseract
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Linux/macOS
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## Használat

### Egyéb dokumentum feldolgozása
```bash
python workflows/01_simple_rag.py \
  --files "data/file1.pdf" "data/file2.docx" "data/file3.pdf" \
  --max 5 \
  --multi
```

### OCR engedélyezése (képalapú PDF-ekhez)
```bash
python workflows/01_simple_rag.py \
  --files "data/image_based.pdf" \
  --use-ocr \
  --max 5
```

### Validációs fájlok
- `data/val.json` - Az AI_Information.pdf validálása (angol)
- `data/val_multi.json` - A három Grepton dokumentum validálása (magyar)

## Fájlstruktúra

```
data/
├── AI_Information.pdf                                          # Szöveges PDF
├── CRA-2023-1067_...Integrations.pdf                          # Szöveges PDF  
├── GRE_INNOVITECH_..._modositas_3.docx                        # Word dokumentum
├── Grepton_Konzorcia_SmartComm_módosítás_20240527.pdf         # Képalapú PDF (OCR szükséges!)
├── val.json                                                    # Validáció (AI_Information)
└── val_multi.json                                              # Validáció (Grepton docs)
```

## Támogatott képformátumok az OCR-ben
- JPEG
- PNG
- TIFF
- BMP

## Megjegyzések

1. **DOCX feldolgozás**: Automatikus - függőségek: `python-docx`
2. **PDF feldolgozás**: Alapból szöveges tartalom, OCR-rel képek is kezelhetőek
3. **Teljesítmény**: Nagyobb dokumentumok feldolgozása lassabb lehet (OCR: 2-5 másodperc/oldal)
4. **Kódolás**: UTF-8 támogatott a dokumentumok feldolgozásakor

## Tesztelési eredmények

| Dokumentum | Típus | Extrahált szöveg | Módszer |
|-----------|-------|------------------|--------|
| AI_Information.pdf | Szöveges PDF | 33,499 char | Standard |
| CRA-2023-1067_...pdf | Szöveges PDF | 2,235 char | Standard |
| GRE_INNOVITECH...docx | Word doc | 1,641 char | python-docx |
| SmartComm_módosítás.pdf | Képalapú PDF | 0 char* | Szükséges OCR |

*OCR telepítés után működni fog

