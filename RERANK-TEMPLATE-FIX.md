# Rerank-Qualität: Ursache & Lösungsweg (2026-09-19)

## Symptom
Mit aktivem Reranker wurde die Retrieval-Qualität **schlechter** statt besser.
A/B (6 Ground-Truth-Fragen, RAGFlow-Retrieval-API):

| Config | rec@1 | rec@3 | MRR |
|---|---|---|---|
| ohne Rerank (`vec 0.6`, `keyword`, `thr 0.2`) | **5/6** | **6/6** | **0.92** |
| mit Rerank | 0/6 | 1–3/6 | ≤0.29 |

Die eigentlich richtige Quelle landete mit Rerank auf Rang 2–15 statt 1.

## Ursache (bewiesen)
Das ausgelieferte Modell ist **`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`**
(Architektur `Qwen3ForSequenceClassification`). Es wurde mit dem
**Qwen3-Reranker-Prompt** trainiert und braucht dieses feste Format:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the
Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: <task>
<Query>: <query>
<Document>: <document><|im_end|>
<|im_start|>assistant
<think>

</think>
```

Das `tokenizer_config.json` des Modells hat **kein `chat_template`
(`chat_template: None`)**. vLLM 0.29 implementiert ein Score-Template
(`SupportsScoreTemplate.get_score_template`) **nur für `jina_vl`** — nicht für
Qwen3. vLLM hängt daher `query` + `document` **roh** aneinander.

Messung an identischen Paaren (relevant vs. irrelevant):

| Eingabe | relevant | irrelevant |
|---|---|---|
| roh (was RAGFlow sendet) | **0.3003** | **0.1213** |
| mit Qwen3-Template | **0.9997** | **0.0001** |

RAGFlow ersetzt im Hybrid-Blend den Embedding-Term durch den Rerank-Score
(`final = tkweight·tksim + vtweight·vtsim`, `vtsim` = Rerank). Unkalibrierte
Scores (0.12–0.30 statt 0.0/1.0) ziehen semantisch relevante Chunks nach unten
→ Rerank verschlechtert das Ranking.

## Lösungsweg (empfohlen): Template-Proxy vor vLLM
`scripts/rerank-proxy.py` (stdlib, kein Image-Build) nimmt RAGFlow's
Cohere-Request `POST /v1/rerank {model,query,documents,top_n}` entgegen,
**formatiert Query/Dokumente mit dem Qwen3-Template** und ruft vLLM `/rerank`
mit den vorformatierten Texten auf (vLLM konkateniert dann korrekt). Antwort
bleibt Cohere-konform (`results[].relevance_score`).

Verifiziert über den Proxy mit RAGFlow-Request-Shape:
**0.9999 (relevant) / 0.0000 (irrelevant)**.

### Chart-Integration (skizziert)
1. `python:3.12-slim` als Sidecar `rerank-proxy` (Port z. B. 8000),
   Script via ConfigMap gemountet, `VLLM_URL=http://127.0.0.1:30000/v1`.
2. nginx-Dashboard-Config: `location /v1/ { proxy_pass http://127.0.0.1:8000/v1/; }`
   (bzw. `/rerank`) → RAGFlow `base_url` (`https://<appid>.aimighty.olares.de/v1`)
   bleibt unverändert.
3. Keine RAGFlow-Änderung nötig; keine zusätzliche GPU-Last.

### Alternativen
- **TEI (HuggingFace Text Embeddings Inference)** mit z. B.
  `BAAI/bge-reranker-v2-m3`: wendet das Reranker-Template intern an, liefert
  kalibrierte Scores. Anderes Modell/Image, weniger Eigenbau.
- **vLLM `--chat-template`**: **nicht gangbar** in vLLM 0.29 (Scoring nutzt für
  Cross-Encoder kein generisches Chat-Template; nur `jina_vl` hat eines).

## Interimsverhalten
Bis der Proxy ausgerollt ist, bleibt der Reranker in RAGFlow **besser
deaktiviert** (Skill `olares-ragflow`: `rerank_id` leer, `vec 0.6`, `keyword`,
`thr 0.2` → rec@1 5/6).

## Post-Fix-Verifikation (App 26.9.6, 2026-09-19)
- Über den Olares-Entrance (rohe Cohere-Anfrage): relevanter Doc **0.9999**,
  irrelevanter **0.0000** (vorher 0.30 / 0.12).
- Diskrimination im echten Korpus (Frage „Welche Sozialversicherungsnummer hat
  Marc Bayer?"): Sozialversicherungsausweis **0.9995**, TK-Anmeldebestätigung
  **0.9973** (enthält dieselbe Nummer → beide valide), Reise-Tagebuch
  (irrelevant) **0.0000**.
- RAGFlow-A/B (6 Fragen): mit Rerank rec@1 **5/6**, MRR 0.86 (vorher **0/6**).
  Der verbleibende „Rückstand" gegenüber ohne Rerank (MRR 0.92) ist eine
  **mehrdeutige Ground-Truth** — das TK-Dokument enthält dieselbe
  Sozialversicherungsnummer; der Embedder rangierte diese valide Quelle zufällig
  auf #1. Kein Qualitätsverlust bei der Antwort.
- **Fazit:** Rerank liefert jetzt kalibrierte Scores und ist in den Assistenten
  (Tier 2) aktiv. Tier 1 (MCP/Helper) bleibt bewusst ohne Rerank (~2 s schneller).
