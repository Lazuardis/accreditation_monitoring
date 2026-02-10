import pandas as pd
import numpy as np

def debug_sunburst():
    csv_file = 'bisdig_outline_status_terakreditasi.csv'
    try:
        df = pd.read_csv(csv_file, dtype={'SubStandar': str, 'Item': str, 'Standar': str}, index_col=False)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

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
    if 'PIC' not in df.columns: df['PIC'] = ""
    # if 'Item Preseden/Referensi' not in df.columns: df['Item Preseden/Referensi'] = "" # Not needed for sunburst logic check

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # --- Tab 3 Logic imitation ---
    # 0) Normalize blanks
    for c in ['Standar','SubStandar','Item','Uraian','PIC','Item Preseden/Referensi']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["", "nan", "None"]), c] = pd.NA

    df['Bobot'] = pd.to_numeric(df.get('Bobot', 0), errors='coerce')

    # 1) BAB weights
    bab_mask = df['SubStandar'].isna() & df['Item'].isna() & df['Standar'].notna()
    bab_weights = df.loc[bab_mask, ['Standar', 'Bobot']].copy()
    bab_weights = bab_weights.rename(columns={'Bobot': 'Bab_Weight'})
    bab_weights['Bab_Weight'] = pd.to_numeric(bab_weights['Bab_Weight'], errors='coerce')

    # 2) SubStandar weights
    sub_mask = df['SubStandar'].notna() & df['Item'].isna()
    sub_weights = df.loc[sub_mask, ['Standar', 'SubStandar', 'Bobot']].copy()
    sub_weights = sub_weights.rename(columns={'Bobot': 'Sub_Weight'})
    sub_weights['Sub_Weight'] = pd.to_numeric(sub_weights['Sub_Weight'], errors='coerce')

    # 3) Merge
    calc_df = df.merge(bab_weights, on='Standar', how='left')
    calc_df = calc_df.merge(sub_weights[['Standar','SubStandar','Sub_Weight']], on=['Standar','SubStandar'], how='left')

    # 5) Net Bobot
    def calculate_net(row):
        bw = row['Bab_Weight']
        sw = row['Sub_Weight']
        iw = row['Bobot']

        bw = 0.0 if pd.isna(bw) else float(bw)
        iw = 0.0 if pd.isna(iw) else float(iw)

        if pd.notna(row['Item']):
            sw = 0.0 if pd.isna(sw) else float(sw)
            return (bw * sw * iw) / 10000.0
        elif pd.notna(row['SubStandar']):
            sw = 0.0 if pd.isna(sw) else float(sw)
            return (bw * sw) / 100.0
        else:
            return bw

    calc_df['Net_Bobot'] = calc_df.apply(calculate_net, axis=1)

    def is_blank(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    def not_blank(s):
        return ~is_blank(s)

    sun = calc_df[['Standar', 'SubStandar', 'Item', 'Bab_Weight', 'Sub_Weight', 'Net_Bobot']].copy()
    sun['Bab_Weight'] = pd.to_numeric(sun['Bab_Weight'], errors='coerce')
    sun['Sub_Weight'] = pd.to_numeric(sun['Sub_Weight'], errors='coerce')
    sun['Net_Bobot']  = pd.to_numeric(sun['Net_Bobot'], errors='coerce').fillna(0.0)

    standar_row_mask    = not_blank(sun['Standar']) & is_blank(sun['SubStandar']) & is_blank(sun['Item'])
    substandar_row_mask = not_blank(sun['SubStandar']) & is_blank(sun['Item'])
    item_row_mask       = not_blank(sun['Item'])

    # Logic to build rows list
    sub_header = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Sub_Weight']].copy()
    sub_header = sub_header.dropna(subset=['Standar', 'SubStandar'])
    sub_sum = sub_header.dropna(subset=['Sub_Weight']).groupby('Standar')['Sub_Weight'].sum()
    expandable_standars = set(sub_sum[(sub_sum - 100.0).abs() <= 0.01].index)
    
    print(f"Expandable Standars: {expandable_standars}")

    rows = []

    # A) Non-expandable
    standar_weights = sun.loc[standar_row_mask, ['Standar', 'Bab_Weight']].copy()
    standar_weights = standar_weights.dropna(subset=['Standar', 'Bab_Weight'])
    for _, r in standar_weights.iterrows():
        s = str(r['Standar']).strip()
        bw = float(r['Bab_Weight']) if pd.notna(r['Bab_Weight']) else 0.0
        if s not in expandable_standars:
            rows.append({'Standar': s, 'SubStandar': pd.NA, 'Item': pd.NA, 'Value': bw})

    # B) Expandable - Items
    items = sun.loc[item_row_mask, ['Standar', 'SubStandar', 'Item', 'Net_Bobot']].copy()
    items = items.dropna(subset=['Standar', 'SubStandar', 'Item'])
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

    # C) Expandable - Leaf SubStandar
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
            rows.append({
                'Standar': str(r['Standar']).strip(),
                'SubStandar': str(r['SubStandar']).strip(),
                'Item': pd.NA,
                'Value': float(r['Value'])
            })

    sun_df = pd.DataFrame(rows)
    print("Initial sun_df contents (head):")
    print(sun_df.head())
    
    # Cleaning logic
    if not sun_df.empty:
        sun_df = sun_df[sun_df['Value'] > 0]
        
        standars_with_sub = set(sun_df.loc[sun_df['SubStandar'].notna(), 'Standar'].unique())
        
        # (Standar, SubStandar) pairs that have item children
        item_pairs = set()
        if sun_df['Item'].notna().any():
            item_pairs = set(
                sun_df.loc[sun_df['Item'].notna(), ['Standar', 'SubStandar']]
                      .apply(lambda r: (r['Standar'], r['SubStandar']), axis=1)
                      .tolist()
            )

        print(f"Standars with sub: {standars_with_sub}")
        print(f"Item pairs count: {len(item_pairs)}")

        keep_mask = pd.Series(True, index=sun_df.index)

        # Drop Standar-only rows when that Standar has SubStandar rows
        drop_standar_parent = sun_df['SubStandar'].isna() & sun_df['Item'].isna() & sun_df['Standar'].isin(standars_with_sub)
        print(f"Dropping {drop_standar_parent.sum()} Standar parent rows")
        
        if drop_standar_parent.any():
            print("Dropped rows sample:")
            print(sun_df[drop_standar_parent].head())

        keep_mask &= ~drop_standar_parent

        # Drop SubStandar-only rows when that (Standar, SubStandar) has Item children
        def has_item_child(row):
            if pd.isna(row['SubStandar']):
                return False
            return (row['Standar'], row['SubStandar']) in item_pairs

        drop_sub_parent = sun_df['Item'].isna() & sun_df['SubStandar'].notna() & sun_df.apply(has_item_child, axis=1)
        print(f"Dropping {drop_sub_parent.sum()} SubStandar parent rows")
        keep_mask &= ~drop_sub_parent

        sun_df = sun_df[keep_mask]
    
    print("\n--- Final sun_df Analysis for '2' ---")
    problem_rows = sun_df[sun_df['Standar'] == '2']
    print(problem_rows)
    
    # Check for leaf violation
    # Check if ('2', NA, NA) is present
    parent_present = problem_rows[(problem_rows['SubStandar'].isna()) & (problem_rows['Item'].isna())]
    if not parent_present.empty:
        print("ALERT: '2' is present as a standalone parent leaf.")
        # Check if there are any other rows for '2'
        children = problem_rows[problem_rows['SubStandar'].notna()]
        if not children.empty:
            print("ALERT: '2' ALSO has children in the dataframe! This causes the error.")
            print("Children:")
            print(children)
        else:
            print("No children found for '2'. It should be a valid leaf.")
    else:
        print("'2' is NOT present as a standalone parent leaf.")

if __name__ == "__main__":
    debug_sunburst()
