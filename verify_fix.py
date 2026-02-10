import pandas as pd
import numpy as np

def verify_fix():
    print("Loading data...")
    csv_file = 'bisdig_outline_status_terakreditasi.csv'
    try:
        df = pd.read_csv(csv_file, dtype={'SubStandar': str, 'Item': str, 'Standar': str}, index_col=False)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

    # --- Pre-processing from main() ---
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all')
    hier_cols = ['Standar', 'SubStandar', 'Item', 'Uraian']
    df = df.dropna(subset=hier_cols, how='all')

    df['Standar'] = df['Standar'].ffill()
    df = df.drop_duplicates(subset=["Standar","SubStandar","Item"], keep="first")

    def _blank(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    df = df[~(
        df['Standar'].notna() &
        _blank(df['SubStandar']) &
        _blank(df['Item']) &
        _blank(df['Uraian'])
    )].copy()

    if 'Bobot' not in df.columns: df['Bobot'] = 0.0
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    for c in ['Standar','SubStandar','Item','Uraian','PIC','Item Preseden/Referensi']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["", "nan", "None"]), c] = pd.NA

    df['Bobot'] = pd.to_numeric(df.get('Bobot', 0), errors='coerce')

    # --- Weights Calculation ---
    bab_mask = df['SubStandar'].isna() & df['Item'].isna() & df['Standar'].notna()
    bab_weights = df.loc[bab_mask, ['Standar', 'Bobot']].copy().rename(columns={'Bobot': 'Bab_Weight'})
    bab_weights['Bab_Weight'] = pd.to_numeric(bab_weights['Bab_Weight'], errors='coerce')

    sub_mask = df['SubStandar'].notna() & df['Item'].isna()
    sub_weights = df.loc[sub_mask, ['Standar', 'SubStandar', 'Bobot']].copy().rename(columns={'Bobot': 'Sub_Weight'})
    sub_weights['Sub_Weight'] = pd.to_numeric(sub_weights['Sub_Weight'], errors='coerce')

    calc_df = df.merge(bab_weights, on='Standar', how='left')
    calc_df = calc_df.merge(sub_weights[['Standar','SubStandar','Sub_Weight']], on=['Standar','SubStandar'], how='left')

    def calculate_net(row):
        bw = float(row['Bab_Weight']) if pd.notna(row['Bab_Weight']) else 0.0
        iw = float(row['Bobot']) if pd.notna(row['Bobot']) else 0.0
        sw = float(row['Sub_Weight']) if pd.notna(row['Sub_Weight']) else 0.0

        if pd.notna(row['Item']):
            return (bw * sw * iw) / 10000.0
        elif pd.notna(row['SubStandar']):
            return (bw * sw) / 100.0
        else:
            return bw

    calc_df['Net_Bobot'] = calc_df.apply(calculate_net, axis=1)

    # --- Sunburst Generation Logic ---
    sun = calc_df[['Standar', 'SubStandar', 'Item', 'Bab_Weight', 'Sub_Weight', 'Net_Bobot']].copy()
    sun['Net_Bobot']  = pd.to_numeric(sun['Net_Bobot'], errors='coerce').fillna(0.0)

    def not_blank(s): return ~s.isna()
    def is_blank(s): return s.isna()

    standar_row_mask    = not_blank(sun['Standar']) & is_blank(sun['SubStandar']) & is_blank(sun['Item'])
    substandar_row_mask = not_blank(sun['SubStandar']) & is_blank(sun['Item'])
    item_row_mask       = not_blank(sun['Item'])

    # Determine expandable standars
    sub_header = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Sub_Weight']].copy()
    sub_header = sub_header.dropna(subset=['Standar', 'SubStandar'])
    sub_sum = sub_header.dropna(subset=['Sub_Weight']).groupby('Standar')['Sub_Weight'].sum()
    expandable_standars = set(sub_sum[(sub_sum - 100.0).abs() <= 0.01].index)

    rows = []

    # A) Non-expandable Standar
    standar_weights = sun.loc[standar_row_mask, ['Standar', 'Bab_Weight']].copy().dropna()
    for _, r in standar_weights.iterrows():
        s = str(r['Standar']).strip()
        bw = float(r['Bab_Weight'])
        if s not in expandable_standars:
            rows.append({'Standar': s, 'SubStandar': None, 'Item': None, 'Value': bw})

    # B) Expandable Standar: Item leaves
    items = sun.loc[item_row_mask, ['Standar', 'SubStandar', 'Item', 'Net_Bobot']].copy().dropna()
    items = items[items['Standar'].isin(expandable_standars)]
    if not items.empty:
        items = items.groupby(['Standar', 'SubStandar', 'Item'], as_index=False)['Net_Bobot'].sum()
        for _, r in items.iterrows():
            rows.append({
                'Standar': str(r['Standar']).strip(),
                'SubStandar': str(r['SubStandar']).strip(),
                'Item': str(r['Item']).strip(),
                'Value': float(r['Net_Bobot'])
            })

    # C) Expandable Standar: Leaf SubStandar
    sub_with_items = set(items['SubStandar'].unique()) if not items.empty else set()
    leaf_sub = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Bab_Weight', 'Sub_Weight']].copy().dropna(subset=['Standar','SubStandar'])
    leaf_sub = leaf_sub[leaf_sub['Standar'].isin(expandable_standars)]
    leaf_sub = leaf_sub[~leaf_sub['SubStandar'].isin(sub_with_items)]
    leaf_sub = leaf_sub.dropna(subset=['Sub_Weight'])
    if not leaf_sub.empty:
        leaf_sub['Value'] = (leaf_sub['Bab_Weight'].fillna(0) * leaf_sub['Sub_Weight'].fillna(0)) / 100.0
        leaf_sub = leaf_sub.groupby(['Standar', 'SubStandar'], as_index=False)['Value'].sum()
        for _, r in leaf_sub.iterrows():
            rows.append({
                'Standar': str(r['Standar']).strip(),
                'SubStandar': str(r['SubStandar']).strip(),
                'Item': None,
                'Value': float(r['Value'])
            })

    sun_df = pd.DataFrame(rows)
    
    print("\n--- Applying Fix Logic ---")
    if not sun_df.empty:
        sun_df = sun_df[sun_df['Value'] > 0]
        
        # --- FIX START ---
        # 1. Uniformly treat empty strings as None
        for col in ['Standar', 'SubStandar', 'Item']:
            if col in sun_df.columns:
                 sun_df[col] = sun_df[col].replace({"": None, "nan": None, "None": None})

        # 2. Re-identify parents with robust logic
        # Parent Standar: A Standar that has children (SubStandar is not None)
        parents_standar = set(sun_df.loc[sun_df['SubStandar'].notna(), 'Standar'].unique())
        
        # Parent SubStandar: A (Standar, SubStandar) that has children (Item is not None)
        parents_sub = set(
            sun_df.loc[sun_df['Item'].notna()]
                  .apply(lambda x: (x['Standar'], x['SubStandar']), axis=1)
                  .unique()
        )

        # 3. Drop rows that are parents
        # Drop if it is a Standar-only row AND that Standar is in parents_standar
        drop_standar = sun_df['SubStandar'].isna() & sun_df['Standar'].isin(parents_standar)
        
        # Drop if it is a SubStandar row (Item is None) AND that (Standar, SubStandar) is in parents_sub
        def is_parent_sub(row):
            if pd.isna(row['SubStandar']): return False
            return (row['Standar'], row['SubStandar']) in parents_sub
            
        drop_sub = sun_df['Item'].isna() & sun_df.apply(is_parent_sub, axis=1)
        
        print(f"Dropping {drop_standar.sum()} Standar parents and {drop_sub.sum()} SubStandar parents.")
        
        sun_df = sun_df[~(drop_standar | drop_sub)]
        # --- FIX END ---

    print(f"\nFinal sun_df size: {len(sun_df)}")
    
    # Check for '2' again
    check_2 = sun_df[sun_df['Standar'] == '2']
    print("\nRows for Standar '2':")
    print(check_2)

    # Verification: Ensure no row is a parent of another
    print("\nVerifying hierarchy integrity...")
    is_valid = True
    
    # Tuples of (Standar, SubStandard, Item) - Item can be None, SubStandard can be None
    paths = []
    for _, row in sun_df.iterrows():
        s = row['Standar']
        ss = row['SubStandar'] if pd.notna(row['SubStandar']) else None
        i = row['Item'] if pd.notna(row['Item']) else None
        paths.append((s, ss, i))
        
    for p1 in paths:
        # Check if p1 is a prefix of any other path p2
        # p1 is prefix if:
        # 1. p1=(A, None, None) and p2=(A, B, ...)
        # 2. p1=(A, B, None) and p2=(A, B, C)
        for p2 in paths:
            if p1 == p2: continue
            
            # Case 1: p1 is Standar leaf
            if p1[1] is None and p1[2] is None:
                if p1[0] == p2[0]:
                    print(f"FAIL: {p1} is a parent of {p2}")
                    is_valid = False
            
            # Case 2: p1 is SubStandar leaf
            if p1[1] is not None and p1[2] is None:
                if p1[0] == p2[0] and p1[1] == p2[1]:
                    print(f"FAIL: {p1} is a parent of {p2}")
                    is_valid = False
                    
    if is_valid:
        print("SUCCESS: Hierarchy is valid! No invalid parent-leaf relationships found.")
    else:
        print("FAILURE: Hierarchy still contains invalid relationships.")

if __name__ == "__main__":
    verify_fix()
