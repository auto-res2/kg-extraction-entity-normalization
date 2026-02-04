# KG Extraction from JacRED: Mention-Span Constrained Entity Normalization

## 1. Background

### JacRED Dataset: Japanese Document-level Relation Extraction Dataset

- **Source**: https://github.com/YoumiMa/JacRED (clone to `/tmp/JacRED`)
- **Splits**: train (1400 docs), dev (300 docs), test (300 docs)
- **Format**: Each document has:
  - `title`: document title
  - `sents`: tokenized sentences (list of list of tokens)
  - `vertexSet`: entities with mentions (list of entity groups, each containing mention dicts with `name`, `type`, `sent_id`, `pos`)
  - `labels`: relations as `{h, t, r, evidence}` where h/t are vertexSet indices and r is a Wikidata P-code
- **9 entity types**: PER, ORG, LOC, ART, DAT, TIM, MON, %, NA
- **35 relation types**: Wikidata P-codes (P131, P27, P569, P570, P19, P20, P40, P3373, P26, P1344, P463, P361, P6, P127, P112, P108, P137, P69, P166, P170, P175, P123, P1441, P400, P36, P1376, P276, P937, P155, P156, P710, P527, P1830, P121, P674)
- **Statistics**: Avg ~17 entities/doc, avg ~20 relations/doc, avg ~253 chars/doc

## 2. Base Implementation (already provided)

The following files implement the baseline extraction:

- **run_experiment.py**: Main orchestrator. Loads data, runs conditions (Baseline, Two-Stage), prints comparison table, saves results.json.
- **data_loader.py**: Data loading from JacRED JSON files, document selection (10 stratified from dev split), few-shot example selection, domain/range constraint table construction from training data.
- **llm_client.py**: Gemini API wrapper using `google-genai` library with Structured Outputs (`response_mime_type="application/json"` + `response_schema`), ThinkingConfig, and retry logic.
- **prompts.py**: All prompt templates including system prompt with 35 relation types defined in Japanese, extraction prompt (baseline and recall-oriented modes), and verification prompt for Stage 2.
- **extraction.py**: Baseline condition:
  - `run_baseline()`: Single LLM call extraction with post-filtering (invalid labels, invalid entity types).
- **evaluation.py**: Entity alignment (3-pass: exact match -> normalized match -> substring match) and micro-averaged P/R/F1 computation.
- **schemas.py**: JSON schemas for Gemini Structured Outputs (extraction schema with entities+relations, verification schema with decisions).

### Key code details

- Entity alignment maps predicted entity IDs to gold vertexSet indices using 3-pass matching.
- Domain/range constraints are built from training data: for each relation P-code, store the set of (head_type, tail_type) pairs observed.
- Verification (Stage 2) processes candidates in batches of 10, asking the LLM to judge each candidate.

## 3. Baseline Results (for comparison)

```
Model: gemini-3-flash-preview (thinking_budget=0)
              Precision   Recall     F1    TP    FP    FN
Baseline           0.26     0.16   0.20    24    70   124
```

**Key issue**: Recall is very low (0.16). One contributing factor is poor entity alignment -- the LLM may extract entities with names that don't exactly match gold vertexSet mentions, causing misalignment and inflating both FP and FN counts.

## 4. Environment Setup

```bash
# Clone JacRED dataset
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# Install dependencies
pip install google-genai openai

# Set API key
export GEMINI_API_KEY="<your-key>"
```

## 5. API Configuration

- **Model**: `gemini-3-flash-preview` (recommended) or `gemini-2.0-flash`
- **Structured Outputs**: `response_mime_type="application/json"` + `response_schema` dict
- **Temperature**: 0.2
- **ThinkingConfig**: `thinking_budget=0` for speed, `2048` for quality
- Configuration is in `llm_client.py` (the `MODEL` constant and `call_gemini()` function)

## 6. Task: Implement Mention-Span Constrained Entity Normalization

### Goal

Improve entity extraction quality by requiring exact text spans from the document, then normalize co-referent mentions before relation extraction. This improves entity alignment accuracy.

### Design

1. **Step 1: Entity extraction with span constraints** (1 LLM call)
   - Modify the extraction schema to require:
     - `mention_text`: exact substring from the document (must be a verbatim span)
     - `sentence_index`: which sentence the mention appears in (0-indexed)
     - `canonical_name`: normalized/canonical entity name (e.g., full name for abbreviations)
   - The prompt instructs the LLM to extract ALL mentions of each entity, with their exact text spans

2. **Step 2: Cluster predicted entities by canonical_name**
   - Merge co-referent mentions (e.g., group mentions like "アメリカ合衆国", "アメリカ", "米国" under one canonical entity)
   - Each cluster gets a single entity ID
   - This mirrors how JacRED's vertexSet groups multiple mentions of the same entity

3. **Step 3: Run relation extraction on clustered entities** (1 LLM call)
   - Provide the LLM with the clustered entity list (showing all mentions/contexts for each entity)
   - Extract relations between clustered entities
   - This gives the LLM richer context about each entity

4. **Step 4: Improved entity alignment using mention_text spans**
   - When aligning predicted entities to gold vertexSet, use the extracted mention_text spans for more precise matching
   - Each predicted entity cluster has multiple mention_text values; match these against the gold vertexSet mentions
   - This should produce better alignment than the current name-only 3-pass matching

### Implementation Details

- **Add `SPAN_ENTITY_SCHEMA` in `schemas.py`**:
  ```python
  SPAN_ENTITY_SCHEMA = {
      "type": "object",
      "properties": {
          "entity_mentions": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "mention_text": {"type": "string"},
                      "sentence_index": {"type": "integer"},
                      "canonical_name": {"type": "string"},
                      "type": {"type": "string", "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]},
                  },
                  "required": ["mention_text", "sentence_index", "canonical_name", "type"],
              },
          },
      },
      "required": ["entity_mentions"],
  }
  ```

- **Create `entity_normalization.py`** with:
  - `cluster_mentions(mentions: list[dict]) -> list[dict]`: Groups mentions by canonical_name (normalized), returns list of entity clusters, each with `id`, `canonical_name`, `type`, `mentions` (list of mention_text values)
  - `build_clustered_entity_prompt(doc_text, clusters)`: Formats clustered entities for the relation extraction prompt

- **Modify entity alignment in `evaluation.py`** (or add a new function):
  - `align_entities_with_spans(clusters, gold_vertex_set)`: Uses mention_text spans from each cluster for matching against gold mentions
  - Match logic: for each cluster, collect all mention_text values; for each gold entity, collect all mention names; compute overlap

- **Add `run_entity_normalized()` in `extraction.py`**:
  - Step 1: Call `call_gemini()` with span entity prompt and `SPAN_ENTITY_SCHEMA`
  - Step 2: Call `cluster_mentions()` to group by canonical_name
  - Step 3: Build relation extraction prompt with clustered entities, call `call_gemini()` with `EXTRACTION_SCHEMA`
  - Step 4: Parse relations, apply constraint filtering
  - Return clusters (as entity list), triples, and stats

- **Add `build_span_entity_prompt()` in `prompts.py`**:
  - Instructs the LLM to extract entity mentions with exact text spans
  - Emphasizes: mention_text must be a verbatim substring from the document
  - Include few-shot example showing span-based extraction

- **Update `run_experiment.py`** to add the third condition

### Expected Improvement

- **Better entity alignment**: Span-based mentions provide more matching surface against gold vertexSet, reducing both FP (from misalignment) and FN (from entity fragmentation).
- **Better relation extraction**: Clustered entities with multiple mention contexts give the LLM richer information.
- **Cost**: ~2x baseline (1 entity extraction call + 1 relation extraction call).

### Evaluation

- Same P/R/F1 computation on the same 10 dev documents
- Report per-document results and aggregate metrics
- **Also report entity alignment accuracy**: number of predicted entities successfully aligned vs total predicted, compared to baseline alignment rate
- Compare: Baseline vs EntityNormalization

### Output Format

The final comparison table should look like:
```
              Precision   Recall     F1    TP    FP    FN
Baseline           ...      ...    ...   ...   ...   ...
EntityNorm         ...      ...    ...   ...   ...   ...
```

Additionally, report entity alignment stats:
```
              Predicted  Aligned  Alignment%
Baseline           ...      ...        ...
EntNorm            ...      ...        ...
```
