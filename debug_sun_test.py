import pandas as pd

calc_df = pd.read_csv('debug_calc_df.csv')

# Normalization function (same as in streamlit)
def _norm(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s.endswith('.0') and s[:-2].isdigit():
        return s[:-2]
    return s

# Build lookup maps preserving first-found order
item_lookup = {}
sub_lookup = {}
standar_lookup = {}

for idx, crow in calc_df.iterrows():
    uraian = crow.get('Uraian') if 'Uraian' in crow.index else None
    if pd.isna(uraian) or str(uraian).strip() == '':
        continue
    uraian = str(uraian).strip()

    it = _norm(crow.get('Item'))
    ss = _norm(crow.get('SubStandar'))
    st_key = _norm(crow.get('Standar'))

    if it is not None:
        if it not in item_lookup:
            item_lookup[it] = (idx, uraian)
        continue
    if ss is not None and st_key is not None:
        tup = (st_key, ss)
        if tup not in sub_lookup:
            sub_lookup[tup] = (idx, uraian)
        continue
    if st_key is not None:
        if st_key not in standar_lookup:
            standar_lookup[st_key] = (idx, uraian)

print('Built lookups:')
print(' - item keys:', len(item_lookup))
print(' - sub keys :', len(sub_lookup))
print(' - stan keys:', len(standar_lookup))

# Rebuild sun_df rows using same logic as streamlit
sun = calc_df[['Standar', 'SubStandar', 'Item', 'Bab_Weight', 'Sub_Weight', 'Net_Bobot']].copy()
sun['Bab_Weight'] = pd.to_numeric(sun['Bab_Weight'], errors='coerce')
sun['Sub_Weight'] = pd.to_numeric(sun['Sub_Weight'], errors='coerce')
sun['Net_Bobot'] = pd.to_numeric(sun['Net_Bobot'], errors='coerce').fillna(0.0)

def is_blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")

def not_blank(s: pd.Series) -> pd.Series:
    return ~is_blank(s)

standar_row_mask = not_blank(sun['Standar']) & is_blank(sun['SubStandar']) & is_blank(sun['Item'])
substandar_row_mask = not_blank(sun['SubStandar']) & is_blank(sun['Item'])
item_row_mask = not_blank(sun['Item'])

# Expandable logic
sub_header = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Sub_Weight']].copy()
sub_header = sub_header.dropna(subset=['Standar', 'SubStandar'])
sub_sum = sub_header.dropna(subset=['Sub_Weight']).groupby('Standar')['Sub_Weight'].sum()
expandable_standars = set(sub_sum[(sub_sum - 100.0).abs() <= 0.01].index)

rows = []
standar_weights = sun.loc[standar_row_mask, ['Standar', 'Bab_Weight']].copy()
standar_weights = standar_weights.dropna(subset=['Standar', 'Bab_Weight'])
for _, r in standar_weights.iterrows():
    s = str(r['Standar']).strip()
    bw = float(r['Bab_Weight']) if pd.notna(r['Bab_Weight']) else 0.0
    if s not in expandable_standars:
        rows.append({'Standar': s, 'SubStandar': pd.NA, 'Item': pd.NA, 'Value': bw})

items = sun.loc[item_row_mask, ['Standar', 'SubStandar', 'Item', 'Net_Bobot']].copy()
items = items.dropna(subset=['Standar', 'SubStandar', 'Item'])
items = items[items['Standar'].isin(expandable_standars)]
if not items.empty:
    items = items.groupby(['Standar', 'SubStandar', 'Item'], as_index=False)['Net_Bobot'].sum()
    for _, r in items.iterrows():
        rows.append({'Standar': str(r['Standar']).strip(), 'SubStandar': str(r['SubStandar']).strip(), 'Item': str(r['Item']).strip(), 'Value': float(r['Net_Bobot'])})

sub_with_items = set(items['SubStandar'].unique()) if not items.empty else set()
leaf_sub = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Bab_Weight', 'Sub_Weight']].copy()
leaf_sub = leaf_sub.dropna(subset=['Standar', 'SubStandar'])
leaf_sub = leaf_sub[leaf_sub['Standar'].isin(expandable_standars)]
leaf_sub = leaf_sub[~leaf_sub['SubStandar'].isin(sub_with_items)]
leaf_sub = leaf_sub.dropna(subset=['Sub_Weight'])
if not leaf_sub.empty:
    leaf_sub['Value'] = (leaf_sub['Bab_Weight'].fillna(0) * leaf_sub['Sub_Weight'].fillna(0)) / 100.0
    leaf_sub = leaf_sub.groupby(['Standar', 'SubStandar'], as_index=False)['Value'].sum()
    for _, r in leaf_sub.iterrows():
        rows.append({'Standar': str(r['Standar']).strip(), 'SubStandar': str(r['SubStandar']).strip(), 'Item': pd.NA, 'Value': float(r['Value'])})

sun_df = pd.DataFrame(rows)

def clean_val(x):
    if x is None or pd.isna(x) or str(x).strip() == "" or str(x).lower() == "nan":
        return None
    return str(x).strip()

for col in ['Standar', 'SubStandar', 'Item']:
    if col in sun_df.columns:
        sun_df[col] = sun_df[col].apply(clean_val)

# Now compute Uraian using the lookups
def _find_uraian(row):
    it = _norm(row.get('Item'))
    ss = _norm(row.get('SubStandar'))
    st_key = _norm(row.get('Standar'))
    if it is not None and it in item_lookup:
        return item_lookup[it][1]
    if ss is not None and st_key is not None:
        tup = (st_key, ss)
        if tup in sub_lookup:
            return sub_lookup[tup][1]
    if st_key is not None and st_key in standar_lookup:
        return standar_lookup[st_key][1]
    return ''

sun_df['Uraian'] = sun_df.apply(_find_uraian, axis=1)

print('\nRebuilt sun_df rows with Uraian:')
print(sun_df.to_string(index=False))

print('\nRows where Uraian is empty:')
print(sun_df[sun_df['Uraian'] == ''].to_string(index=False))
