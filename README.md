# Accreditation Monitoring App

Streamlit app for managing accreditation document hierarchy, weight allocation, PIC assignment, dependency mapping, task progress tracking, and weighted analytics.

The app entrypoint is `streamlit.py`. This README is intentionally based on what is directly called by that file.

## What The App Does

The UI has 3 tabs:

1. `Configuration`
- Edit hierarchy weights (`Bobot`) by level.
- Assign PICs to sections/items.
- Map dependencies/references (`Item Preseden/Referensi`) to internal items or external docs.
- Review dependency summary with resolved PIC ownership.

2. `Task Tracker`
- Select a leaf node (Item, standalone SubStandar, or standalone Standar).
- Update task `Progress` (0, 20, 40, 60, 80, 100).
- Save supporting `URL Link` per leaf task.
- Export updated master data to CSV format.

3. `Analytics`
- Compute weighted progress and workload from hierarchy weights.
- Show chapter progress, PIC responsibility share, planned vs actual PIC progress, completion rate, and PIC collaboration table.
- Provide per-PIC drilldown and CSV export for collaboration data.

## Runtime Topology

Single-file app logic lives in `streamlit.py` with these core functions:

- `clean_label(val)`: normalizes tree labels.
- `build_tree_data(df)`: builds hierarchy tree nodes for selection UI.
- `convert_to_original_format(df)`: prepares downloadable CSV output with expected column order.
- `main()`: tab layout, all editors, tracker logic, analytics pipeline.

All stateful interactions use `st.session_state`.

## Files Directly Used By `streamlit.py`

### Required / Default Inputs

- `data_akreditasi_updated(3).csv`
  - Default dataset loaded when no manual upload is provided.
- `list_dosen_tekdik.csv`
  - PIC/member registry used for assignment multiselect.
- `list_external_dokumen.csv`
  - External document registry (`item`, `PIC`) for dependency mapping.

### User Input

- Uploaded CSV via sidebar file uploader (optional override for default dataset).

### Generated / Exported

- `debug_calc_df.csv`
  - Debug snapshot of analytics calculation dataframe.
- Downloaded from UI:
  - `data_akreditasi_updated.csv` (master export)
  - `collaboration_<PIC>.csv` (PIC collaboration export)

## Expected Data Columns

Primary master dataset (uploaded/default) is expected to include, or app will initialize missing fields where applicable:

- `Standar`
- `SubStandar`
- `Item`
- `Uraian`
- `Bobot`
- `PIC`
- `Item Preseden/Referensi`
- `Progress`

Optional runtime-added column:

- `URL Link` (added in Task Tracker if absent)

Notes:
- `Standar` values are forward-filled at load.
- Duplicate keys by (`Standar`, `SubStandar`, `Item`) are dropped (keep first).
- Empty hierarchy rows are removed.

## Session State Keys

The app persists these keys:

- `df`: working master dataframe.
- `current_selection`: selected node from tree.
- `tree_key`: incremented to force tree widget reset.
- `members_df`: PIC registry from `list_dosen_tekdik.csv`.
- `ad_hoc_docs`: list of external documents.
- `ad_hoc_pics`: mapping external document -> PIC.

## Hierarchy And Editing Rules

### Weight Editing (`Configuration`)

- Global view edits `Standar` header rows.
- Selecting a `Standar` edits its `SubStandar` header rows.
- Selecting a `SubStandar` edits its `Item` rows.
- Save is allowed even if totals are not 100%, but UI warns when total differs from 100%.

### PIC Assignment (`Configuration`)

- PIC values are stored as comma-separated names in `PIC`.
- Multi-select options come from `list_dosen_tekdik.csv`.

### Dependency Mapping (`Configuration`)

Reference targets can include:
- Internal `Item` labels.
- Standalone `SubStandar` leaves (substandar without item children).
- External docs from `list_external_dokumen.csv` plus in-session registry additions.

## Progress Update Rules (`Task Tracker`)

Progress can be updated only for leaf nodes:

- `Item` rows.
- Standalone `SubStandar` rows (no `Item` children).
- Standalone `Standar` rows (no `SubStandar` children).

Allowed progress values are fixed discrete options:

- `0, 20, 40, 60, 80, 100`

`URL Link` is stored on the same leaf row.

## Analytics Mechanics (`Analytics`)

### Data Cleanup

- Trims string columns and normalizes blank-like values.
- Coerces `Bobot` and `Progress` to numeric.
- Progress is clamped to `[0, 100]`.
- Validation summary is shown if cleanup was needed.

### Net Weight Formula (`Net_Bobot`)

Given:
- `Bab_Weight`: weight from `Standar` header row.
- `Sub_Weight`: weight from `SubStandar` header row.
- `Item Bobot`: item-level `Bobot`.

Formulas:

- Item row:
  - `Net_Bobot = (Bab_Weight * Sub_Weight * Item_Bobot) / 10000`
- SubStandar row (non-item):
  - `Net_Bobot = (Bab_Weight * Sub_Weight) / 100`
- Standar row (non-sub):
  - `Net_Bobot = Bab_Weight`

### Leaf Definition For KPI

Leaf tasks are:
- Any non-empty `Item` row.
- SubStandar rows with no item children.
- Standar rows with no substandar children.

### KPI Calculations

- Overall weighted progress:
  - `sum(Progress * Net_Bobot) / 100`
- Weight integrity:
  - `sum(Net_Bobot)` across leaf rows (target is near `100%`).
- PIC responsibility split:
  - `Net_Bobot_Per_PIC = Net_Bobot / PIC_Count` when multiple PICs share a task.
- PIC planned vs actual:
  - Planned = sum of `Net_Bobot_Per_PIC`
  - Actual = Planned * weighted avg progress / 100
- PIC completion rate:
  - `Actual / Planned * 100`

## Run Locally

Install deps and run:

```bash
pip install -r requirements.txt
streamlit run streamlit.py
```

## Current Scope (As Implemented)

- App behavior is stateful in-memory during session.
- Edits to member registry (`list_dosen_tekdik.csv`) are persisted to file.
- External registry additions are session-only unless separately saved back to `list_external_dokumen.csv` (current code initializes from file but does not write updates to it).
- Analytics writes `debug_calc_df.csv` on each analytics render.

## Suggested Next Step For Chatbot Tab

If you add a chatbot sub-tab, use deterministic intent handlers over `st.session_state.df` (structured Q&A only), and optionally pass computed outputs to LLM only for natural-language formatting.
