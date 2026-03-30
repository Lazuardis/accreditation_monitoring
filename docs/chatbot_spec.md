# Chatbot Specification (Structured Q&A)

## Purpose

Define the contract for adding a chatbot sub-tab to `streamlit.py` that answers structured project-monitoring questions from existing app data, with deterministic calculations and optional Gemini-based natural-language phrasing.

Scope is intentionally limited to information retrieval and KPI summaries. No idea generation, no open-ended planning, and no free-form reasoning beyond formatting.

## Design Principles

1. Deterministic first
- All filtering, aggregation, and formulas must be computed in Python/Pandas from the same dataframe used by analytics (`st.session_state.df`).
- LLM must not be the source of truth for numbers.

2. Bounded capability
- Chatbot only supports predefined intents.
- Unsupported intents return guidance + supported examples.

3. Auditable answers
- Every answer includes compact evidence and formula metadata.
- Each answer includes `as_of` timestamp.

4. Graceful degradation
- If Gemini/API fails, return deterministic plain-text response from the same result payload.

## Target UI Placement

Add a sub-tab under existing `Analytics` tab:
- Sub-tab A: existing charts/dashboard
- Sub-tab B: `Chat Assistant`

Alternative allowed:
- Add a 4th top-level tab named `Chat Assistant`

Preferred: sub-tab inside Analytics for context continuity.

## Data Source Contract

Primary source:
- `st.session_state.df` (already loaded/edited in app)

Required columns for chatbot features:
- `Standar`
- `SubStandar`
- `Item`
- `Uraian`
- `Bobot`
- `PIC`
- `Progress`
- `Item Preseden/Referensi`

Optional:
- `URL Link`

Preprocessing must mirror analytics conventions:
- Trim strings
- Normalize empty-like values to NA
- Coerce `Bobot` and `Progress` to numeric
- Clamp `Progress` to `[0, 100]`
- Build `Net_Bobot`, `PIC_Count`, `Net_Bobot_Per_PIC`, and `leaf_df` using existing analytics logic

## Supported Intents (v1)

All questions should be mapped to one of the intents below.

1. `overall_progress_by_pic`
- User asks: “How much is AB current progress?”
- Output: weighted average progress for PIC AB, plus workload share and task count.

2. `tasks_by_pic`
- User asks: “What tasks are for AB?”
- Output: list of AB tasks with progress, relative weight, URL if available.

3. `stalled_tasks_by_pic`
- User asks: “What tasks AB still hasn’t progressed?”
- Output: AB tasks where progress <= threshold (default 0, optional <=20 mode).

4. `collaboration_partners_by_pic`
- User asks: “With who should AB collaborate?”
- Output: task-level collaborator mapping from multi-PIC assignments.

5. `overall_status_summary`
- User asks: “Overall status now?”
- Output: total weighted progress, active leaf count, weight integrity.

6. `chapter_progress_summary`
- User asks: “Progress per chapter/standar?”
- Output: chapter completion table sorted ascending.

## Intent Router Contract

### Function
`route_intent(user_text: str) -> dict`

### Return shape
- `intent`: one of supported intents or `unsupported`
- `entities`: extracted params
  - `pic` (if present)
  - `threshold` (for stalled tasks; default 0)
- `confidence`: float 0..1
- `notes`: optional diagnostics

### Routing Strategy
v1 recommendation:
- Rule-based keyword patterns + PIC token extraction from known PIC names.
- Optional Gemini classification is allowed only for fallback when rules fail.

## PIC Normalization Contract

### Function
`normalize_pic(raw_pic: str, available_pics: list[str]) -> dict`

### Behavior
- Trim, uppercase, remove extra spaces.
- Exact match first.
- Alias table optional (`PIC_ALIASES` dict).
- Fuzzy fallback (small tolerance) optional; if ambiguous, return explicit disambiguation error.

### Return shape
- `ok`: bool
- `pic`: normalized PIC when ok
- `error`: message when not ok
- `candidates`: optional list

## Deterministic Handler Contracts

Each handler must return a standardized payload.

### Common payload
- `intent`
- `status`: `ok` | `empty` | `error`
- `answer_data`: dict with intent-specific values
- `evidence_rows`: list[dict] (compact records)
- `formula_used`: list[str]
- `as_of`: ISO timestamp
- `warnings`: list[str]

### Handlers

1. `handle_overall_progress_by_pic(pic, prepared_data)`
- Computes:
  - planned responsibility (%): sum(Net_Bobot_Per_PIC)
  - weighted avg progress (%): weighted by Net_Bobot_Per_PIC
  - actual progress share (%): planned * avg / 100
  - task_count

2. `handle_tasks_by_pic(pic, prepared_data)`
- Returns task table for PIC with:
  - hierarchy ids (`Standar`,`SubStandar`,`Item`)
  - `Uraian`, `Progress`, `Net_Bobot_Per_PIC`, `URL Link`

3. `handle_stalled_tasks_by_pic(pic, threshold, prepared_data)`
- Filters PIC tasks where `Progress <= threshold`
- Returns task list + count + percentage of PIC workload impacted

4. `handle_collaboration_partners_by_pic(pic, prepared_data)`
- For each PIC task, parse original multi-PIC field and return other PICs
- Include summary counts: solo vs collaborative tasks

5. `handle_overall_status_summary(prepared_data)`
- Returns global KPIs:
  - overall weighted progress
  - leaf count
  - weight integrity

6. `handle_chapter_progress_summary(prepared_data)`
- Returns completion per `Standar`

## Gemini Integration Contract (Formatting Only)

### Function
`format_answer_with_gemini(result_payload, user_question) -> str`

### Input to Gemini
- User question
- Structured deterministic payload (JSON-like)
- Strict system instruction:
  - Do not alter numeric values
  - Do not invent entities/tasks
  - Use concise Bahasa Indonesia or English matching user language
  - Mention if data is empty or partial

### Output style
- 3 sections max:
  - `Summary`
  - `Key Points`
  - `Evidence`

### Failure fallback
If Gemini fails/timeout:
- Return `format_answer_plain(result_payload)`.

## Plain Formatter Contract

### Function
`format_answer_plain(result_payload) -> str`

Must produce readable text without LLM using template per intent.

## Error / Fallback Policy

1. No data loaded
- Message: ask user to upload/load data in Configuration tab.

2. Unsupported intent
- Return supported intent examples.

3. PIC not found
- Return closest candidates.

4. Empty result
- Explicitly state no matching rows.

5. Gemini unavailable
- Show deterministic answer without wording enhancement.

## Logging And Observability

Keep simple in v1:
- In-memory list in session state (`chat_logs`) with:
  - timestamp
  - user query
  - routed intent
  - normalized entities
  - status
  - latency_ms
- Optional CSV append later.

## Security And Credentials

Use one of:
- `st.secrets["GEMINI_API_KEY"]`
- environment variable `GEMINI_API_KEY`

Never hardcode credentials in code or docs.

## Suggested File/Code Structure (Incremental)

Option A (minimal edits): keep in `streamlit.py`
- add helper block: preprocessing, routing, handlers, formatters
- add chat sub-tab UI in Analytics

Option B (cleaner): split modules
- `chatbot/prepare.py`
- `chatbot/router.py`
- `chatbot/handlers.py`
- `chatbot/formatters.py`
- import into `streamlit.py`

Recommended for maintainability: Option B, but Option A is acceptable for quick v1.

## Example Q&A Mapping

1. “Berapa progress PIC NAR sekarang?”
- intent: `overall_progress_by_pic`
- entity: `pic=NAR`

2. “Task apa saja untuk LMS?”
- intent: `tasks_by_pic`
- entity: `pic=LMS`

3. “Untuk PS, mana yang belum jalan?”
- intent: `stalled_tasks_by_pic`
- entity: `pic=PS`, threshold default `0`

4. “Dengan siapa MRI perlu kolaborasi?”
- intent: `collaboration_partners_by_pic`
- entity: `pic=MRI`

## Acceptance Criteria (v1)

1. Chatbot answers all supported intents with deterministic values matching analytics tables.
2. PIC normalization handles typical input variants (case/spacing).
3. Gemini formatting is optional and does not affect correctness.
4. Unsupported questions return clear guidance.
5. Every response includes concise evidence context.

## Out Of Scope (v1)

- Full RAG over unstructured docs
- Autonomous recommendations/decision making
- Multi-turn tool calling beyond intent + handler execution
