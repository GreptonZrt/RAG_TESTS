# 📊 RAG Workflow Evaluation System - Gyors Útmutató

## 🎯 Mit csinál?

Az evaluation system **automatikusan nyomon követi az összes RAG workflow (11-14, 19) pontosságát és teljesítményét** egy felülírható CSV táblázatban.

```
Workflow 11  → Metrics → workflow_results.csv ✅
Workflow 12  → Metrics → workflow_results.csv ✅
Workflow 13  → Metrics → workflow_results.csv ✅
Workflow 14  → Metrics → workflow_results.csv ✅
Workflow 19  → Metrics → workflow_results.csv ✅
```

## 📁 Kimenet fájlok

| Fájl | Formátum | Tartalom |
|------|---------|----------|
| `workflow_results.csv` | CSV Táblázat | Bruttó adatok, könnyen feldolgozható |
| `workflow_results.md` | Markdown | Szép táblázat, előnézet |
| `workflow_results.html` | HTML Dashboard | Grafikus megjelenítés |
| `EVALUATION_RESULTS.md` | Dokumentáció | Részletes útmutató |

## ⚡ Gyors Start

### 1️⃣ Futtass egy workflow-t
```bash
python workflows/11_feedback_loop_rag.py --max 5
```
✅ Automatikusan elmenti az adatokat az evaluation system-be!

### 2️⃣ Megtekintheted az eredményeket
```bash
cat workflow_results.csv
```

### 3️⃣ Szép reportok létrehozása
```bash
python visualize_results.py
```

### 4️⃣ Vagy futtass mindent egyszerre
```bash
python run_all_workflows.py
```

## 📊 Mit nyomon követ?

| Metrika | Leírás |
|---------|--------|
| `queries_processed` | Feldolgozott kérdések száma |
| `avg_chunks_retrieved` | Átlag chunk-ok száma |
| `avg_response_length` | Átlagos válasz hossza |
| `avg_utility_rating` | Érdekes válaszok értékelése (1-5) |
| `avg_iterations` | Iterációk száma |

## 📈 Eredmények értelmezése

### Workflow 11 (Feedback Loop)
- **5.0 chunks** - Jó mennyiségű kontextus
- **2 queries** - Két kérdés feldolgozva

### Workflow 12 (Adaptive)
- **3.0 chunks** - Mérsékelten több chunk
- Query type-specifikus retrieval

### Workflow 13 (Self-RAG)
- **5.0 utility** - Tökéletes értékelés!
- **2.0 iterations** - Kétszeri finomítás

### Workflow 14 (Propositions)
- **7 propositions** - Atomi propozíciók
- Pontosabb chunk-level matching

### Workflow 19 (HyDE)
- **2.0 chunks** - Minimális de fontos
- Hipotikus dokumentum alapú

## 🔄 Felülírás működése

```
1. futtatás:  ├─ CSV létrehozása
              └─ WF11, WF12, WF13, WF14, WF19 adatok

2. futtatás:  ├─ WF11 adatok FELÜLÍRÁSA
              └─ Más workflow-k maradnak

3. futtatás:  ├─ WF12 adatok FELÜLÍRÁSA
              └─ Más workflow-k maradnak
```

**Mindig a legfrissebb adat marad a CSV-ben!**

## 🛠️ Technikai működés

1. Workflow futtatásakor:
   ```python
   metrics = create_metrics_from_results(results)
   tracker = ResultsTracker()
   tracker.add_result(workflow_id="11", ...)
   tracker.save_results()  # ← Felülírja az előző értékeket
   ```

2. Meglévő CSV-ből olvassa az előző adatokat
3. Frissíti az aktuális workflow adatait
4. Menti vissza (felülírva az előzőt)

## 💡 Hasznos parancsok

```bash
# Egyetlen workflow futtatása
python workflows/12_adaptive_rag.py --max 10

# Összes futtatása
python run_all_workflows.py

# Vizualizáció frissítése
python visualize_results.py

# Eredmények megtekintése
cat workflow_results.csv
cat workflow_results.md

# HTML megnyitása (Windows)
start workflow_results.html

# HTML megnyitása (macOS)
open workflow_results.html

# HTML megnyitása (Linux)
xdg-open workflow_results.html
```

## 📝 CSV formátum

```csv
workflow_id,workflow_name,timestamp,queries_processed,avg_chunks_retrieved,...
11,Feedback Loop RAG,2025-12-16T10:25:11.111631,2,5.0,...
12,Adaptive RAG,2025-12-16T10:25:57.867204,1,3.0,...
13,Self-RAG,2025-12-16T10:26:46.825335,1,2.0,...
14,Proposition Chunking RAG,2025-12-16T10:28:53.713104,1,0.0,...
19,HyDE RAG,2025-12-16T10:29:16.739574,1,2.0,...
```

## ✨ Előnyei

✅ **Automatikus** - Workflow futás közben fut  
✅ **Felülírható** - Mindig friss adatok  
✅ **Vizualizált** - HTML dashboard  
✅ **Összehasonlítható** - Könnyen összevethetők az eredmények  
✅ **Exportálható** - CSV, Markdown, HTML formátumok  

## 🐛 Gyakori kérdések

**K: Mit jelent az üres cella?**  
V: Az adott metrika nincs elérhető az adott workflow-nál

**K: Milyen sűrűn frissül az adat?**  
V: Minden workflow futás után azonnal

**K: Elvesznek az adatok?**  
V: Nem, csak felülírásra kerülnek (a timestamp mutatja az utolsó futást)

**K: Lehet több query-vel futtatni?**  
V: Igen: `--max 100` (az átlagok módosulnak)

---

📊 **RAG Workflow Evaluation System v1.0**  
Készült: 2025-12-16
