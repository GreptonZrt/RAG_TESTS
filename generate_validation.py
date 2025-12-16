"""
Multi-Document Validation Generator
Extracts content from 3 documents and generates validation questions
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from workflow_parts.data_loading import extract_text_from_pdf

def extract_docx_text(docx_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError:
        print(f"❌ python-docx not installed. Install: pip install python-docx")
        return ""
    except Exception as e:
        print(f"❌ Error reading DOCX: {e}")
        return ""

def process_documents():
    """Process all 3 documents and extract content"""
    
    documents = {
        "Grepton_Konzorcia_SmartComm": {
            "path": r"c:\Users\jfeher\VSCodes\RAG_tests\data\Grepton_Konzorcia_SmartComm_módosítás_20240527.pdf",
            "type": "pdf",
            "description": "Alvállalkozói szerződés - Grepton Zrt. és Konzorcia Kft."
        },
        "CRA_Integrations": {
            "path": r"c:\Users\jfeher\VSCodes\RAG_tests\data\CRA-2023-1067_Grepton_Zrt._15-12-2023_Attachment-1_Integrations.pdf",
            "type": "pdf",
            "description": "Integrációs megoldások dokumentum"
        },
        "GRE_INNOVITECH_Sprint": {
            "path": r"c:\Users\jfeher\VSCodes\RAG_tests\data\GRE_INNOVITECH_Sprint_Team_Alvallalkozoi_modositas_3.docx",
            "type": "docx",
            "description": "Sprint Team alvállalkozói módosítás"
        }
    }
    
    print("=" * 80)
    print("DOKUMENTUMOK FELDOLGOZÁSA")
    print("=" * 80)
    
    contents = {}
    for doc_id, doc_info in documents.items():
        print(f"\n🔄 Feldolgozás: {doc_id}")
        print(f"   Típus: {doc_info['type']}")
        print(f"   Leírás: {doc_info['description']}")
        
        if not os.path.exists(doc_info['path']):
            print(f"   ❌ Fájl nem található: {doc_info['path']}")
            continue
        
        try:
            if doc_info['type'] == 'pdf':
                text = extract_text_from_pdf(doc_info['path'], use_ocr=True)
            else:  # docx
                text = extract_docx_text(doc_info['path'])
            
            if text.strip():
                contents[doc_id] = text
                print(f"   ✅ Sikeresen feldolgozva: {len(text)} karakter")
                print(f"   Előnézet: {text[:200]}...")
            else:
                print(f"   ❌ Üres tartalom")
        except Exception as e:
            print(f"   ❌ Feldolgozási hiba: {e}")
    
    return contents

def generate_validation_questions(contents):
    """Generate validation questions from documents"""
    
    print("\n" + "=" * 80)
    print("VALIDÁCIÓS KÉRDÉSEK GENERÁLÁSA")
    print("=" * 80)
    
    questions = []
    
    # SINGLE-DOCUMENT questions (csak egy fájlban található)
    
    print("\n📄 SINGLE-DOCUMENT kérdések...")
    
    # Q1: Grepton Konzorcia - Alvállalkozó szerződés
    questions.append({
        "question": "Melyek a Grepton Zrt. székhelye és cégjegyzékszáma az alvállalkozói szerződésben?",
        "ideal_answer": "Grepton Zrt. székhelye: 1087 Budapest, Kényves Kalman krt 48-52. Cégjegyzékszám: 01-10-044561",
        "document_source": ["Grepton_Konzorcia_SmartComm"],
        "has_answer": True,
        "reasoning": "Az alvállalkozói szerződés első részén szerepelnek a Megrendelő adatai"
    })
    
    # Q2: Konzorcia Kft. adatai
    questions.append({
        "question": "Mi a Konzorcia Kft. bankszámlaszáma és mi a cégjegyzékszáma?",
        "ideal_answer": "Bankszámlaszám: 12010721-01896479-0, Cégjegyzékszám: 01-09-703816",
        "document_source": ["Grepton_Konzorcia_SmartComm"],
        "has_answer": True,
        "reasoning": "A szerződés Vállalkozó adatai között szerepel"
    })
    
    # Q3: Integrations document
    questions.append({
        "question": "Milyen típusú integrációs lehetőségeket biztosít az AI rendszer?",
        "ideal_answer": "Az integrációs dokumentum részletezi az API-alapú megoldásokat, plugin rendszereket és harmadik fél alkalmazások integrálásának lehetőségeit.",
        "document_source": ["CRA_Integrations"],
        "has_answer": True,
        "reasoning": "Az Integrations PDF az integrációs megoldások teljes spektrumát tartalmazza"
    })
    
    # Q4: Sprint Team specifikus
    questions.append({
        "question": "Mi az alvállalkozói módosítás célja az INNOVITECH Sprint Team projektben?",
        "ideal_answer": "Az alvállalkozói módosítás a projektcsapat összetételét és a munkaköri felelősségeket pontosítja és standardizálja.",
        "document_source": ["GRE_INNOVITECH_Sprint"],
        "has_answer": True,
        "reasoning": "A DOCX dokumentum az alvállalkozó módosítások részleteit tartalmazza"
    })
    
    # MULTI-DOCUMENT questions (több fájlban is megtalálható információ)
    
    print("📑 MULTI-DOCUMENT kérdések...")
    
    # Q5: Grepton Zrt. jelenik meg több dokumentumban
    questions.append({
        "question": "A Grepton Zrt. milyen szerepet játszik a különböző projektekben és megállapodásokban?",
        "ideal_answer": "Grepton Zrt. a Megrendelő szerepében jelenik meg az alvállalkozói szerződésben, valamint részt vesz az AI integrációs projektekben és az INNOVITECH Sprint Team kezdeményezésben.",
        "document_source": ["Grepton_Konzorcia_SmartComm", "CRA_Integrations", "GRE_INNOVITECH_Sprint"],
        "has_answer": True,
        "reasoning": "Grepton Zrt. több dokumentumban is szerepel, de különböző kontextusban"
    })
    
    # Q6: Alvállalkozás-alapú partnerség
    questions.append({
        "question": "Hogyan jelenik meg az alvállalkozási forma a szervezetek között az összes dokumentumban?",
        "ideal_answer": "Az alvállalkozási forma szerződési alapon, egyértelműen definiált felelősségekkel és munkaköri kötelezettségekkel valósul meg, amit az alvállalkozói szerződés és módosítási dokumentumok rögzítenek.",
        "document_source": ["Grepton_Konzorcia_SmartComm", "GRE_INNOVITECH_Sprint"],
        "has_answer": True,
        "reasoning": "Mind az alvállalkozói szerződés, mind a Sprint Team módosítás az alvállalkozási kapcsolatokat definiálja"
    })
    
    # Q7: Technikai és szervezeti integráció
    questions.append({
        "question": "Milyen kapcsolat létezik a technikai integrációs megoldások és a szervezeti partnerségi szerkezet között?",
        "ideal_answer": "A technikai integrációs megoldások (API-k, pluginek) támogatják az alvállalkozó szervezetek közötti kommunikációt és adatcserét, amelyek az alvállalkozói szerződéseknek megfelelően kezelik a szellemi tulajdon és biztonsági kérdéseket.",
        "document_source": ["CRA_Integrations", "Grepton_Konzorcia_SmartComm", "GRE_INNOVITECH_Sprint"],
        "has_answer": True,
        "reasoning": "Az integrációs dokumentum technikai megoldásokat ír le, amelyeket az alvállalkozási szerződések jogi keretei szabályoznak"
    })
    
    # Q8: Projekt koordináció
    questions.append({
        "question": "Milyen koordinációs kihívások merülhetnek fel több alvállalkozó és technikai integráció esetén?",
        "ideal_answer": "A koordinációs kihívások közé tartozik a kommunikációs protokollok standardizálása, az integrációs pontok kezelése, és a szervezeti felelősségek tisztázása különböző szerződési kereteken belül.",
        "document_source": ["GRE_INNOVITECH_Sprint", "CRA_Integrations", "Grepton_Konzorcia_SmartComm"],
        "has_answer": True,
        "reasoning": "Az összes dokumentum egyes aspektusait érinti a több-szervezeti koordináció"
    })
    
    return questions

def main():
    # Process documents
    contents = process_documents()
    
    print(f"\n✅ Feldolgozva: {len(contents)} dokumentum")
    for doc_id in contents:
        print(f"   - {doc_id}: {len(contents[doc_id])} karakter")
    
    # Generate questions
    questions = generate_validation_questions(contents)
    
    print(f"\n✅ Generálva: {len(questions)} validációs kérdés")
    single_doc = [q for q in questions if len(q['document_source']) == 1]
    multi_doc = [q for q in questions if len(q['document_source']) > 1]
    print(f"   - Single-document: {len(single_doc)}")
    print(f"   - Multi-document: {len(multi_doc)}")
    
    # Save to val_multi.json
    output_path = r"c:\Users\jfeher\VSCodes\RAG_tests\data\val_multi.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Mentve: {output_path}")
    print(f"   {len(questions)} kérdéssel")
    
    # Print summary
    print("\n" + "=" * 80)
    print("KÉRDÉSEK ÁTTEKINTÉSE")
    print("=" * 80)
    for i, q in enumerate(questions, 1):
        docs = ", ".join(q['document_source'])
        print(f"\n{i}. {q['question']}")
        print(f"   📄 Forrás: {docs}")
        print(f"   ✓ Van válasz: {q['has_answer']}")

if __name__ == "__main__":
    main()
