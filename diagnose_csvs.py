import pandas as pd

files = [
    ('bisdig_outline_status_terakreditasi.csv', 'status'),
    ('bisdig_outline_v1.csv', 'v1')
]

reports = {}

for fname, key in files:
    df = pd.read_csv(fname, dtype=str)
    df.columns = df.columns.str.strip()
    # keep original copy
    raw = df.copy()
    # strip whitespace in string cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # coerce numeric bobot
    df['Bobot_num'] = pd.to_numeric(df.get('Bobot', pd.Series()), errors='coerce')

    # ffill Standar like streamlit does for working view
    df['Standar_ff'] = df['Standar'].ffill()

    total_rows = len(df)
    unique_standars = df['Standar_ff'].dropna().unique().tolist()

    # detect Standar-level rows (Standar present, SubStandar & Item blank)
    stan_level = df[df['Standar'].notna() & (df['SubStandar'].isna() | (df['SubStandar']=='') ) & (df['Item'].isna() | (df['Item']==''))]
    sub_headers = df[df['SubStandar'].notna() & (df['Item'].isna() | (df['Item']=='') )]
    item_rows = df[df['Item'].notna() & (df['Item']!='')]

    # For each Standar, check counts
    stan_summary = []
    for s in pd.Series(unique_standars):
        sstr = str(s)
        mask = df['Standar_ff'] == s
        has_stan_row = ((df['Standar'].notna()) & mask & (df['SubStandar'].isna() | (df['SubStandar']=='')) & (df['Item'].isna() | (df['Item']==''))).any()
        has_sub = ((df['SubStandar'].notna()) & mask).any()
        has_item = ((df['Item'].notna()) & mask).any()
        # sum sub weights
        sub_mask = (df['Standar_ff'] == s) & (df['SubStandar'].notna()) & (df['Item'].isna() | (df['Item']==''))
        sub_sum = pd.to_numeric(df.loc[sub_mask, 'Bobot'], errors='coerce').sum(min_count=1)
        stan_summary.append((sstr, has_stan_row, bool(has_sub), bool(has_item), float(sub_sum) if pd.notna(sub_sum) else None))

    reports[key] = {
        'rows': total_rows,
        'unique_standars_count': len(unique_standars),
        'stan_level_count': len(stan_level),
        'sub_headers_count': len(sub_headers),
        'item_rows_count': len(item_rows),
        'stan_summary': stan_summary,
        'tail_rows': raw.tail(5).to_dict(orient='records')
    }

# Print concise comparison
print('\n=== Quick Stats ===')
for k in reports:
    r = reports[k]
    print(f"\nFile: {k}")
    print(f" Rows: {r['rows']}, Unique Standars: {r['unique_standars_count']}")
    print(f" Standar-level rows: {r['stan_level_count']}, Sub headers: {r['sub_headers_count']}, Item rows: {r['item_rows_count']}")

# compare per-Standar sub weight sums
print('\n=== Per-Standar SubWeight sums (None=NA) ===')
all_standars = sorted(set([s for r in reports.values() for s,_ in zip([None],[]) ]))
# build union of standars from both reports
union = set()
for k in reports:
    for s, *_ in reports[k]['stan_summary']:
        union.add(s)
union = sorted(union, key=lambda x: (float(x) if x.replace('.','',1).isdigit() else 9999, x))

for s in union:
    line = [s]
    for k in ['status', 'v1']:
        ss = next((t for t in reports[k]['stan_summary'] if t[0]==s), None)
        if ss is None:
            line.append('MISSING')
        else:
            has_stan, has_sub, has_item, sub_sum = ss[1], ss[2], ss[3], ss[4]
            line.append(f"stan_row={has_stan}, sub={has_sub}, item={has_item}, sub_sum={sub_sum}")
    print(' | '.join(line))

# Print notable anomalies for status file
print('\n=== Notable anomalies in status file ===')
status = reports['status']
# standards where sub_sum exists but not approx 100
for s, has_stan, has_sub, has_item, sub_sum in status['stan_summary']:
    if sub_sum is not None and abs(sub_sum - 100.0) > 0.01:
        print(f"Standar {s}: sub_sum={sub_sum}, has_stan={has_stan}, has_item={has_item}")

print('\n=== Tail rows of status file (raw) ===')
for r in reports['status']['tail_rows']:
    print(r)

print('\nDone')
