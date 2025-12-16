# 🎯 Új Overall Score Rendszer: Answer Similarity Scoring

## Változás Összegzése

Az **Overall Score** mostantól az **LLM-alapú Answer Similarity** tényezőt helyezi a középpontba. Ez a legfontosabb tényező a végső minőség értékelésénél, mivel az a legfontosabb, hogy a generált válasz mennyire közel áll az ideális válaszhoz.

---

## 📊 Az Új Scoring Formula

```
Overall Score = 
    (answer_similarity × 0.5) +              # 50% ⭐ LEGFONTOSABB
    (valid_response_rate × 0.1) +            # 10%
    (chunks_score × 0.1) +                   # 10%
    (length_score × 0.1) +                   # 10%
    (utility_score × 0.1)                    # 10%
    
    = 0-100 skála
```

---

## 🔍 Tényezők Magyarázata

### 1️⃣ **Answer Similarity** - **50%** 🌟 (LEGFONTOSABB)

**Mit mér:** Mennyire hasonlít a generált AI válasz az ideális válaszhoz?

**Hogyan működik:**
- Az LLM-et arra kérjük, hogy 0-100 skálán értékelje az AI válasz minőségét az ideális válaszhoz képest
- Szempont: tartalom relevanciája, pontossága, információmennyisége

**Pontozás:**
- 0-20: Teljesen hibás vagy irreleváns
- 21-40: Részlegesen releváns, de hiányos információ
- 41-60: Túlnyomórészt helyes, de kisebb hézagok/pontatlanságok
- 61-80: Nagyon jó match, apró különbségek
- 81-100: Kitűnő match, az ideális válasz jól lefedett

**Képlet:** `answer_similarity × 0.5`

---

### 2️⃣ **Valid Response Rate** - **10%**

**Mit mér:** A válaszok hány %-a nem "I don't know" típusú?

**Képlet:** `valid_response_rate × 0.1`

**Megjegyzés:** Korábban 40% volt, most csökkent, mert a válasz tartalma (similarity) már fedezi ezt.

---

### 3️⃣ **Chunks Retrieved** - **10%**

**Mit mér:** Átlagosan hány chunk-ot hozott vissza a retriever?

**Optimális tartomány:** 3-5 chunk

**Képlet:** 
```
if avg_chunks <= 5:
    score = (avg_chunks / 5.0 × 100) × 0.1
else:
    score = 100 × 0.1
```

**Megjegyzés:** Túl sok chunk = szöveghalmozás, túl kevés = hiányos info

---

### 4️⃣ **Response Length** - **10%**

**Mit mér:** Az AI válasz hossza karakterben

**Optimális tartomány:** 80-150 karakter

**Képlet:**
```
if 80 <= avg_response_len <= 150:
    score = 100 × 0.1
else:
    score = (min(avg_response_len, 150) / 150 × 100) × 0.1
```

**Logika:** Sem túl rövid (nem informatív), sem túl hosszú (nem tömör)

---

### 5️⃣ **Utility Rating** - **10%**

**Mit mér:** Az adott válaszokhoz adott szubjektív értékelés (1-5 skála)

**Képlet:** `(avg_utility / 5.0 × 100) × 0.1`

**Megjegyzés:** Csak akkor aktív, ha a workflow implementálja

---

## 📈 Gyakorlati Példa

```
Workflow: Simple RAG
1 query, 1 answer

Komponensek:
  - Answer Similarity: 95.0  → 95.0 × 0.5 = 47.5
  - Valid Response Rate: 100.0  → 100.0 × 0.1 = 10.0
  - Chunks Retrieved: 5.0  → (5/5 × 100) × 0.1 = 10.0
  - Response Length: 120 (optimal)  → 100 × 0.1 = 10.0
  - Utility Rating: 3.0  → (3/5 × 100) × 0.1 = 6.0

TOTAL = 47.5 + 10.0 + 10.0 + 10.0 + 6.0 = 83.5/100
```

---

## 🔧 Megvalósítás

### Fájlok Módosítva:

1. **[workflow_parts/results_tracker.py](workflow_parts/results_tracker.py)**
   - Új függvény: `_calculate_answer_similarity()` - LLM-alapú értékelés
   - Fallback: `_simple_string_similarity()` - string hasonlóság (API hiba esetén)
   - Módosított: `_calculate_overall_score()` - új súlyozás
   - Módosított: `create_metrics_from_results()` - answer similarity kalkukláció

2. **[workflow_parts/orchestration.py](workflow_parts/orchestration.py)**
   - Módosított: `run_rag_batch()` - `ideal_answer` field hozzáadása result dict-hez

3. **Összes workflow (01-14, 19)**
   - Automatikusan felhasználja az új scoring-ot (nem szükséges módosítás)

---

## 💾 CSV Persistence

Az `workflow_results.csv` most tartalmazza az új `avg_answer_similarity` oszlopot:

```
workflow_id | workflow_name | overall_score | valid_response_rate | avg_answer_similarity | ...
01          | Simple RAG    | 83.5          | 100.0               | 95.0                  | ...
02          | Semantic Ch.. | 83.5          | 100.0               | 95.0                  | ...
...
```

---

## 🎯 Miért Ez a Prioritás?

### Az answer similarity a legfontosabb, mert:

1. **Végső cél**: Az AI pontos és releváns választ adjon
2. **Erőforrás-felhasználás már másodlagos**: Az összesen 50% a válasz minőségre fordítódik
3. **Retriever stílusok indifferensek**: Nem számít, hogy 3 vagy 5 chunk-ot használ, ha a válasz jó
4. **Objektív LLM értékelés**: Az OpenAI/Azure GPT-4o értékeli a valós minőséget

---

## 📊 Ranking Hatás

Az első batch futás eredménye:

| Rank | Workflow | Score | Similarity |
|------|----------|-------|------------|
| 🥇 1 | Semantic Chunking | 83.5 | 95.0 |
| 🥈 2 | Doc Augmentation | 83.5 | 95.0 |
| 🥉 3 | Simple RAG | 81.0 | 90.0 |
| 4 | Contextual Headers | 81.0 | 90.0 |
| 5 | Reranker | 79.5 | 95.0 |
| ... | ... | ... | ... |

**Megfigyelés:** A magasabb similarity közvetlenül magasabb overall score-t eredményez.

---

## 🔄 Backward Compatibility

- ✅ Régi workflows: Automata falllback 50.0 similarity score (ha nincs ideal_answer)
- ✅ CSV: Meglévő adatok nem törlődnek, új oszlop hozzáadódik
- ✅ API fallback: Ha az LLM API-hívás sikertelen, simple string similarity-t használ

---

## 📝 Validációs Fájlok

Az answer similarity-t csak akkor lehet kiszámítani, ha a validációs fájl tartalmazza az `ideal_answer` mezőt:

### val_multi.json (✅ Támogatott)
```json
{
  "question": "...",
  "ideal_answer": "...",  ← Ez szükséges
  "document_source": [...],
  "has_answer": true
}
```

### val.json / val_rl.json (❓ Opcionális)
- Ha nincs `ideal_answer`, a fallback score 50.0 lesz

---

## 🚀 Következő Lépések (Ajánlott)

1. **Tesztelés**: Futtatás több query-vel
   ```bash
   python run_all_workflows_batch.py --max 5
   ```

2. **Fine-tuning** (ha szükséges):
   - Módosíthatod a súlyozást a `_calculate_overall_score()` függvényben
   - Pl: Ha response_length fontosabb: `0.5` helyett `0.6`-re

3. **Monitoring**:
   - Kövesse a `workflow_results.csv` frissüléseit
   - Figyelje az `avg_answer_similarity` trendjeit

---

**Megjegyzés:** Az answer similarity LLM API-hívás, így ez a métrika lassabb, mint a korábban. De egy query-nkénti extra ~1-2 másodperc a pontosabb értékelésért megéri! 🎯
