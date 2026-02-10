import pandas as pd

IN = 'bisdig_outline_status_terakreditasi.csv'
OUT = 'bisdig_outline_status_terakreditasi.fixed.csv'

# Read all columns as strings to preserve original layout
df = pd.read_csv(IN, dtype=str)
# Strip column names
df.columns = df.columns.str.strip()

# Strip whitespace in string cells
for c in df.columns:
    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Drop rows that are entirely empty or all NaN/empty strings
non_empty_mask = df.apply(lambda row: any((not pd.isna(v) and str(v) != '') for v in row), axis=1)
df = df[non_empty_mask].reset_index(drop=True)

# Normalize Bobot: remove commas, coerce to numeric with 2 decimals, keep as string formatted
if 'Bobot' in df.columns:
    def clean_bobot(x):
        if pd.isna(x) or str(x).strip() == '':
            return ''
        s = str(x).replace(',', '')
        try:
            v = float(s)
            return f"{v:.2f}"
        except Exception:
            return str(x).strip()
    df['Bobot'] = df['Bobot'].apply(clean_bobot)

# Fill-forward Standar to detect groups
df['Standar_ff'] = df['Standar'].ffill()

# Identify Standar header rows: Standar present & SubStandar empty & Item empty
is_standar_header = df['Standar'].notna() & (df['SubStandar'].isna() | (df['SubStandar'] == '')) & (df['Item'].isna() | (df['Item'] == ''))

# For each header row, if there exists any row with the same Standar_ff and non-empty SubStandar, drop the header row
to_drop = []
for idx, row in df[is_standar_header].iterrows():
    s = row['Standar']
    if s is None or (isinstance(s, float) and pd.isna(s)):
        continue
    has_sub = ((df['Standar_ff'] == s) & df['SubStandar'].notna() & (df['SubStandar'] != '')).any()
    if has_sub:
        to_drop.append(idx)

if to_drop:
    df = df.drop(index=to_drop).reset_index(drop=True)

# Remove helper column
df = df.drop(columns=['Standar_ff'])

# Write fixed CSV
df.to_csv(OUT, index=False)
print(f"Wrote cleaned file: {OUT} (rows: {len(df)})")
