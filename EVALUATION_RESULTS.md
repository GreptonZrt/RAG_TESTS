# RAG Workflow Evaluation System

## 📊 Értékelési rendszer az összes RAG stratégiához

Ez az értékelési rendszer automatikusan nyomon követi az egyes RAG workflow-k (11-14, 19) teljesítményét és pontosságát, egy felülírható táblázatban.

## 📁 Fájlok

- **`workflow_results.csv`** - CSV táblázat az összes eredménnyel (felülírható)
- **`workflow_results.md`** - Markdown formátumú szép táblázat
- **`workflow_results.html`** - HTML vizualizáció (szép grafikus megjelenítés)

## 🎯 Felhasználás

### 1. Workflow futtatása
```bash
python workflows/11_feedback_loop_rag.py --max 5
python workflows/12_adaptive_rag.py --max 5
python workflows/13_self_rag.py --max 5
python workflows/14_proposition_chunking_rag.py --max 5
python workflows/19_hyde_rag.py --max 5
```

Minden workflow futtatása után:
- ✅ Automatikusan kiszámítja az értékelési metrikákat
- ✅ Felülírja az előző értékeket a CSV-ben
- ✅ Mutatja a teljes összehasonlítási táblázatot

### 2. Vizualizáció létrehozása
```bash
python visualize_results.py
```

Ez létrehozza:
- `workflow_results.html` - Szép grafikus dashbord
- `workflow_results.md` - Markdown táblázat

## 📈 Nyomon követett metrikák

Minden workflow-hoz az alábbi metrikák kerülnek nyomkövetésre:

| Metrika | Leírás | Workflow |
|---------|--------|----------|
| `queries_processed` | Feldolgozott lekérdezések száma | Összes |
| `avg_chunks_retrieved` | Átlagosan visszakeresett chunk-ok száma | Összes |
| `avg_response_length` | Átlagos válasz hossza (karakterek) | Összes |
| `avg_utility_rating` | Átlagos hasznosság értékelés (1-5) | 13, 11 |
| `avg_iterations` | Átlagos iterációk száma | 13, 11 |
| `category_Factual` | Tényszerű kérdések száma | 12 |
| `total_propositions` | Összes generált propozíció | 14 |

## 📊 CSV Táblázat struktúrája

```
workflow_id,workflow_name,timestamp,queries_processed,avg_chunks_retrieved,avg_response_length,...
11,Feedback Loop RAG,2025-12-16T10:25:11.111631,2,5.0,0.0,...
12,Adaptive RAG,2025-12-16T10:25:57.867204,1,3.0,97.0,...
13,Self-RAG,2025-12-16T10:26:46.825335,1,2.0,0.0,...
14,Proposition Chunking RAG,2025-12-16T10:28:53.713104,1,0.0,97.0,...
19,HyDE RAG,2025-12-16T10:29:16.739574,1,2.0,85.0,...
```

## 🔄 Felülírás logikája

- **Első futtatás**: Új CSV fájl létrehozása
- **Újrafuttatás**: Az előző az adott workflow-hoz tartozó sor felülírása (az ID és timestamp alapján)
- **Összes futtatás után**: A teljes táblázat mutatja az 5 workflow-t egy sorban

## 🎨 Vizualizációk

### HTML Dashboard (`workflow_results.html`)
- 📊 Szép grafikus megjelenítés
- 📈 Összehasonlító diagramok
- 🎯 Rendezett táblázatok
- 🌈 Gradiens háttér és modern CSS

Megtekintéshez:
```bash
# Windows
start workflow_results.html

# macOS
open workflow_results.html

# Linux
xdg-open workflow_results.html
```

### Markdown Report (`workflow_results.md`)
- 📝 Strukturált táblázatok
- 📌 Workflow leírások
- 📍 Könnyen olvasható formátum

## 🔍 Eredmények értelmezése

### Feedback Loop RAG [11]
- **Strength**: Magasabb chunk retrieval (5.0 átlag)
- **Use for**: Hosszú, kontextus-gazdag válaszokra

### Adaptive RAG [12]
- **Strength**: Query-type specifikus retrieval
- **Use for**: Vegyes típusú kérdések

### Self-RAG [13]
- **Strength**: Magas utility rating (5.0), iteratív finomítás
- **Use for**: Érdekes válaszokra van szükség

### Proposition Chunking [14]
- **Strength**: Atomi propozíciók alapú retrieval
- **Use for**: Pontosabb chunk-level matching

### HyDE RAG [19]
- **Strength**: Hipotikus dokumentum alapú retrieval
- **Use for**: Szematikus hasonlóság javítása

## 🛠️ Technikai részletek

### ResultsTracker osztály (`workflow_parts/results_tracker.py`)
```python
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results

# Metrikák létrehozása
metrics = create_metrics_from_results(results)

# Tracker inicializálása
tracker = ResultsTracker("workflow_results.csv")

# Eredmény hozzáadása
tracker.add_result(
    workflow_id="11",
    workflow_name="Feedback Loop RAG",
    metrics=metrics
)

# Mentés (felülírás)
tracker.save_results()

# Összefoglalás megjelenítése
print(tracker.get_summary())
```

### Integrálás workflow-kba
Minden workflow-ban az alábbi kód található:
```python
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results

# ... workflow futtatása ...

metrics = create_metrics_from_results(results)
tracker = ResultsTracker()
tracker.add_result(workflow_id="XX", workflow_name="...", metrics=metrics)
tracker.save_results()
print(tracker.get_summary())
```

## 📝 Gyakorlati munkafolyamat

1. **Workflow futtatás**
   ```bash
   python workflows/11_feedback_loop_rag.py --max 10
   ```
   Eredmény: CSV automatikusan frissül, összefoglalást mutat

2. **Másik workflow futtatás**
   ```bash
   python workflows/12_adaptive_rag.py --max 10
   ```
   Eredmény: Új sor/adatok hozzáadódnak

3. **Összefoglalás megtekintése**
   ```bash
   python visualize_results.py
   ```
   Eredmény: HTML és Markdown reportok

4. **Eredmények összehasonlítása**
   Nyisd meg: `workflow_results.md` vagy `workflow_results.html`

## 🐛 Hibaelhárítás

### "workflow_results.csv not found"
- Az első workflow futtatásakor automatikusan létrejön
- Ha nem jön létre: ellenőrizd az írási engedélyeket

### Üres metrikák az CSV-ben
- Bizonyos metrikák csak specifikus workflow-kban érhető el
- Ez normális, az N/A vagy üres cella jelzi

### HTML nem jelenik meg szépre
- Böngészőben nyisd meg közvetlenül (nem file:// protokollon)
- Vagy használd helyette a Markdown verzióját

## 📊 Tanács a metriky értelmezésre

- **Magas chunk retrieval**: Több kontextus, de potenciálisan zaj
- **Alacsony avg_response_length**: Rövid, tömör válaszok
- **Magasabb utility_rating**: Jobban értékelt válaszok
- **Több iteráció**: Iteratív finomítás (Self-RAG)

---

*Létrehozva: 2025-12-16*
*RAG Evaluation Framework v1.0*
