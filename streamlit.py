import streamlit as st
import pandas as pd
import streamlit_antd_components as sac
import plotly.express as px
import io


# Set page config
st.set_page_config(page_title="Accreditation Monitoring", layout="wide")

def clean_label(val):
    if pd.isna(val) or str(val).lower() == 'nan':
        return ""
    return str(val).strip()

def build_tree_data(df):
    """Transforms flat CSV into nested sac.TreeItem objects with Weights."""
    tree_items = []
    sep = " | "
    for standar_name, standar_group in df.groupby('Standar', sort=False):
        sub_items = []
        sub_headers = standar_group[standar_group['SubStandar'].notna() & standar_group['Item'].isna()]
        for _, sub_head_row in sub_headers.iterrows():
            sub_id = clean_label(sub_head_row['SubStandar'])
            sub_uraian = clean_label(sub_head_row['Uraian'])
            sub_bobot = sub_head_row.get('Bobot', 0)
            leaf_items = []
            leaf_df = standar_group[standar_group['SubStandar'].astype(str) == sub_id]
            leaf_df = leaf_df[leaf_df['Item'].notna()]
            for _, item_row in leaf_df.iterrows():
                i_id = clean_label(item_row['Item'])
                i_ur = clean_label(item_row['Uraian'])
                i_bobot = item_row.get('Bobot', 0)
                leaf_items.append(sac.TreeItem(label=f"{i_id}{sep}{i_ur} ({i_bobot}%)"))
            sub_items.append(sac.TreeItem(label=f"{sub_id}{sep}{sub_uraian} ({sub_bobot}%)", children=leaf_items))
        
        standar_label_row = standar_group[standar_group['SubStandar'].isna() & standar_group['Item'].isna()]
        if not standar_label_row.empty:
            s_id = clean_label(standar_name)
            s_ur = clean_label(standar_label_row.iloc[0]['Uraian'])
            s_bobot = standar_label_row.iloc[0].get('Bobot', 0)
            label = f"{s_id}{sep}{s_ur} ({s_bobot}%)" if s_ur else f"{s_id} ({s_bobot}%)"
            tree_items.append(sac.TreeItem(label=label, children=sub_items))
    return tree_items



def convert_to_original_format(df):
    """
    Mengonversi DataFrame session_state kembali ke format CSV asli.
    Menghilangkan duplikasi Standar (ffill kebalikan) dan menangani NaN.
    """
    # 1. Buat salinan agar tidak merusak data di session_state
    export_df = df.copy()

    # 2. Kembalikan kolom 'Standar' ke format aslinya
    # Kita hanya menyisakan nilai Standar jika baris tersebut adalah baris pertama 
    # atau jika nilainya berbeda dari baris sebelumnya.
    export_df['Standar'] = export_df['Standar'].where(export_df['Standar'] != export_df['Standar'].shift(), None)

    # 3. Pastikan kolom numerik (Bobot) bersih dari format string yang aneh
    export_df['Bobot'] = export_df['Bobot'].fillna(0)
    export_df['Item Preseden/Referensi'] = export_df.get('Item Preseden/Referensi', "").fillna("")
    export_df['PIC'] = export_df.get('PIC', "").fillna("")
    export_df['Progress'] = export_df.get('Progress', 0).fillna(0)
    
    # 4. Susun kolom sesuai urutan asli
    cols = ['Standar', 'SubStandar', 'Item', 'Uraian', 'Bobot', 'PIC', 'Item Preseden/Referensi','Progress']
    export_df = export_df[cols]

    # 5. Konversi NaN menjadi string kosong agar di CSV terlihat sebagai ,,
    export_df = export_df.fillna("")

    return export_df

def main():
    st.title("Accreditation Monitoring")

    tab1, tab2, tab3 = st.tabs(["Configuration", "Task Tracker", "Analytics"])

    with tab1:

        # 1. Sidebar Controls
        st.sidebar.header("Controls")
        if st.sidebar.button("🔄 Reset App & Data", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

        # uploaded file automatically by accessing bisdig_outline_v1.csv
        if uploaded_file is None:
            try:
                with open('bisdig_outline_v1.csv', 'rb') as f:
                    uploaded_file = io.BytesIO(f.read())
            except FileNotFoundError:
                pass

        if uploaded_file is not None:
            # --- DATA INITIALIZATION ---
            if 'df' not in st.session_state:
                df = pd.read_csv(uploaded_file, dtype={'SubStandar': str, 'Item': str, 'Standar': str}, index_col=False)
                df.columns = df.columns.str.strip()
                df['Standar'] = df['Standar'].ffill()
                if 'Bobot' not in df.columns: df['Bobot'] = 0.0
                if 'PIC' not in df.columns: df['PIC'] = ""
                if 'Item Preseden/Referensi' not in df.columns: df['Item Preseden/Referensi'] = ""
                # Strip whitespace from string columns
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                st.session_state.df = df
            
            # --- SELECTION STATE INITIALIZATION ---
            if 'current_selection' not in st.session_state:
                st.session_state.current_selection = None
            if 'tree_key' not in st.session_state:
                st.session_state.tree_key = 0

            df = st.session_state.df

            # --- 1. DOCUMENT HIERARCHY ---
            st.subheader("1. Document Hierarchy")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1: show_line = st.toggle("Show Hierarchy Lines", value=True)
            with col_t2: expand_all = st.toggle("Expand All", value=True)

            # RECTIFIED RESET LOGIC: Changing the key forces the tree to reset
            if st.button("Back to Global View", key="reset_tree_btn"):
                st.session_state.current_selection = None
                st.session_state.tree_key += 1 # Nuclear reset of the tree widget
                st.rerun()



            tree_data = build_tree_data(df)
            
            # Draw the tree with a dynamic key
            tree_selection = sac.tree(
                items=tree_data,
                label='',
                index=None,
                show_line=show_line,
                open_all=expand_all,
                key=f"tree_widget_{st.session_state.tree_key}"
            )

            # Capture user selection from the tree
            if tree_selection is not None:
                st.session_state.current_selection = tree_selection

            st.divider()

    # --- 2. WEIGHT ASSIGNMENT EDITOR ---
            st.subheader("2. Weight Assignment Editor")
            
            with st.expander("Weight Settings", expanded=True):
                display_df = pd.DataFrame()
                level_description = ""
                selected_node = st.session_state.current_selection

                if not selected_node:
                    # LEVEL 1: GLOBAL
                    mask = df['SubStandar'].isna() & df['Item'].isna()
                    global_df = df[mask].copy()
                    display_df = global_df.drop_duplicates(subset=['Standar'], keep='first')
                    display_df['DisplayLabel'] = display_df['Standar'] + " - " + display_df['Uraian'].fillna("")
                    level_description = "Global View: Assigning Weights to Main Chapters (BAB)"
                else:
                    sep = " | "
                    # Extract the ID and ignore the suffix
                    node_id = selected_node.split(sep)[0].strip()
                    
                    if node_id in df['Standar'].astype(str).values:
                        mask = (df['Standar'] == node_id) & (df['SubStandar'].notna()) & (df['Item'].isna())
                        display_df = df[mask].copy()
                        display_df['DisplayLabel'] = display_df['SubStandar'].astype(str) + " " + display_df['Uraian'].fillna("")
                        level_description = f"Sub-Sections under {node_id}"
                    
                    elif node_id in df['SubStandar'].astype(str).values:
                        mask = (df['SubStandar'].astype(str) == node_id) & (df['Item'].notna())
                        display_df = df[mask].copy()
                        display_df['DisplayLabel'] = display_df['Item'].astype(str) + ": " + display_df['Uraian'].fillna("")
                        level_description = f"Specific Items under Section {node_id}"

                # --- RESTORED STATUS AND WARNING LOGIC ---
                if not display_df.empty:
                    st.info(f"💡 {level_description}")
                    
                    edit_view = display_df[['DisplayLabel', 'Bobot']].copy()
                    editor_key = f"weight_editor_{str(selected_node).split(' | ')[0] if selected_node else 'global'}"

                    updated_table = st.data_editor(
                        edit_view,
                        column_config={
                            "DisplayLabel": st.column_config.TextColumn("Section / Item Name", disabled=True),
                            "Bobot": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, format="%.2f%%" )
                        },
                        hide_index=True,
                        use_container_width=True,
                        key=editor_key
                    )

                    # Restoration of Total Weight Validation
                    total_bobot = float(updated_table['Bobot'].sum())
                    
                    if abs(total_bobot - 100.0) > 0.01:
                        st.warning(f"⚠️ Total weight sum is {total_bobot:.2f}% — it should equal 100%. Please adjust before saving.")
                    else:
                        st.success("✅ Total weights sum to 100%. Ready to save.")

                    if st.button("💾 Save Weights", key="save_weights_btn"):
                        # Positional mapping to avoid IndexError
                        for i in range(len(updated_table)):
                            actual_index_in_master = display_df.index[i]
                            new_bobot_value = updated_table.iloc[i]['Bobot']
                            st.session_state.df.at[actual_index_in_master, 'Bobot'] = new_bobot_value
                        
                        st.toast("Progress saved to master data!", icon='💾')
                        st.success(f"Weights for '{level_description}' successfully committed to memory.")
                        st.rerun() 
                else:
                    # Restoration of No Selection / Leaf Warning
                    if selected_node:
                        st.warning("📍 You have selected a leaf item. Weights are assigned to groups; please select a parent (Chapter or Sub-Section) in the tree above.")
                    else:
                        st.info("👈 Use the hierarchy tree to select a specific section, or edit Global Chapter weights below.")

            st.write(selected_node)

            # --- 3. MEMBER TASK ASSIGNMENT ---
            st.divider()
            st.subheader("3. Member Task Assignment")

            if 'members_df' not in st.session_state:
                st.session_state.members_df = pd.read_csv('list_dosen_tekdik.csv', dtype=str)

            with st.expander("➕ Add New Member to Database"):
                new_mem = st.text_input("New Member Name")
                if st.button("Add Member"):
                    if new_mem.strip():
                        st.session_state.members_df = pd.concat([st.session_state.members_df, pd.DataFrame({'Name': [new_mem.strip()]})], ignore_index=True)
                        st.session_state.members_df.to_csv('list_dosen_tekdik.csv', index=False)
                        st.rerun()

            if not display_df.empty:
                assignment_view = display_df[['DisplayLabel', 'PIC']].copy()
                # Convert string to list for multiselect
                assignment_view['PIC'] = assignment_view['PIC'].apply(lambda x: [i.strip() for i in str(x).split(',')] if (pd.notna(x) and x != "") else [])

                pic_editor_key = f"pic_editor_{str(selected_node).split(' | ')[0] if selected_node else 'global'}"
                
                updated_pic_table = st.data_editor(
                    assignment_view,
                    column_config={
                        "DisplayLabel": st.column_config.TextColumn("Section / Item Name", disabled=True),
                        "PIC": st.column_config.MultiselectColumn("Assigned PICs", options=st.session_state.members_df['Name'].tolist())
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=pic_editor_key
                )

                if st.button("💾 Save Assignments"):
                    for i in range(len(updated_pic_table)):
                        st.session_state.df.at[display_df.index[i], 'PIC'] = ", ".join(updated_pic_table.iloc[i]['PIC'])
                    
                    # Save updated members to CSV
                    
                    
                    st.success("Assignments saved!")
                    st.rerun()

            # st.dataframe(st.session_state.df, use_container_width=True)

            

    # --- 4. DEPENDENCY/REFERENCE MAPPING ---
            st.divider()
            st.subheader("4. Dependency/Reference Mapping")



            # 1. Robust Initialization for External Registry
            # We check for both keys to ensure they exist as a pair
            if 'ad_hoc_docs' not in st.session_state or 'ad_hoc_pics' not in st.session_state:
                try:
                    # Attempt to load the existing CSV
                    ext_df = pd.read_csv('list_external_dokumen.csv', dtype=str).dropna()
                    st.session_state.ad_hoc_docs = ext_df['item'].tolist()
                    st.session_state.ad_hoc_pics = dict(zip(ext_df['item'], ext_df['PIC']))
                except Exception:
                    # If file doesn't exist or error occurs, initialize empty
                    st.session_state.ad_hoc_docs = []
                    st.session_state.ad_hoc_pics = {}
             
            # Ensure the column exists in the master dataframe
            if 'Item Preseden/Referensi' not in st.session_state.df.columns:
                st.session_state.df['Item Preseden/Referensi'] = ""

            # 2. External/Ad-hoc Document Registry
            with st.expander("📂 External Document Registry", expanded=False):
                st.write("Add documents and their PICs that are NOT listed in the master CSV.")
                
                reg_col1, reg_col2, reg_col3 = st.columns([2, 2, 1])
                with reg_col1:
                    new_doc_input = st.text_input("Document Name", key="ext_doc_name")
                with reg_col2:
                    new_pic_input = st.text_input("Assign PIC", key="ext_pic_name")
                with reg_col3:
                    st.write("##") # Alignment
                    if st.button("Add to Registry"):
                        if new_doc_input.strip() and new_doc_input not in st.session_state.ad_hoc_docs:
                            st.session_state.ad_hoc_docs.append(new_doc_input.strip())
                            st.session_state.ad_hoc_pics[new_doc_input.strip()] = new_pic_input.strip()
                            st.success("Registered!")
                            st.rerun()
                
                # Display current registry
                if st.session_state.ad_hoc_docs:
                    reg_display = pd.DataFrame({
                        "Document": st.session_state.ad_hoc_docs,
                        # Safe retrieval using .get() to prevent AttributeErrors
                        "PIC": [st.session_state.ad_hoc_pics.get(d, "N/A") for d in st.session_state.ad_hoc_docs]
                    })
                    # st.dataframe(reg_display, use_container_width=True, hide_index=True)

            # 3. Linking Interface
            if not display_df.empty:
                st.info(f"🔗 Mapping dependencies for: **{level_description}**")
                
                mapping_view = display_df[['DisplayLabel', 'Item Preseden/Referensi']].copy()
                
                # Pre-processing string to list for MultiselectColumn
                mapping_view['Item Preseden/Referensi'] = mapping_view['Item Preseden/Referensi'].apply(
                    lambda x: [i.strip() for i in str(x).split(',')] if (pd.notna(x) and str(x) != "" and str(x) != "nan") else []
                )

                # --- DYNAMIC OPTION LOGIC ---
                # 1. Get all actual Items
                items_list = df[df['Item'].notna()].apply(lambda row: f"{row['Item']}: {row['Uraian']}", axis=1).tolist()
                
                # 2. Get SubStandars that act as "Leaves" (They have no Items under them)
                sub_with_items = df[df['Item'].notna()]['SubStandar'].unique()
                leaf_substandars = df[
                    df['SubStandar'].notna() & 
                    df['Item'].isna() & 
                    ~df['SubStandar'].isin(sub_with_items)
                ].apply(lambda row: f"{row['SubStandar']}: {row['Uraian']}", axis=1).tolist()

                selectable_options = items_list + leaf_substandars + st.session_state.ad_hoc_docs

                dep_key = f"dep_editor_{str(selected_node).split(' | ')[0] if selected_node else 'global'}"

                updated_dep_table = st.data_editor(
                    mapping_view,
                    column_config={
                        "DisplayLabel": st.column_config.TextColumn("Target Document/Item", disabled=True),
                        "Item Preseden/Referensi": st.column_config.MultiselectColumn(
                            "Precedent/Reference Items",
                            help="Select items, standalone sub-standards, or external documents",
                            options=selectable_options,
                            width="large"
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=dep_key
                )

                if st.button("💾 Save Dependency Mapping"):
                    for i in range(len(updated_dep_table)):
                        actual_index_in_master = display_df.index[i]
                        selected_list = updated_dep_table.iloc[i]['Item Preseden/Referensi']
                        st.session_state.df.at[actual_index_in_master, 'Item Preseden/Referensi'] = ", ".join(selected_list)
                    
                    st.success("Mapping successfully updated.")
                    st.rerun()
            else:
                st.info("👈 Select a section in the hierarchy to begin mapping document precedents.")

            # 4. Summary View with PIC Tracking
            with st.expander("🔍 Dependency Summary Table (with PICs)"):
                summary_df = st.session_state.df[st.session_state.df['Item Preseden/Referensi'] != ""].copy()
                if not summary_df.empty:
                    def get_precedent_pics(prec_str):
                        # --- 1) Guard: NaN / float / None / empty ---
                        if prec_str is None:
                            return ""
                        # pandas NaN is float; also handle other non-strings
                        if not isinstance(prec_str, str):
                            # if it's NaN (float), return empty; otherwise cast safely
                            try:
                                if pd.isna(prec_str):
                                    return ""
                            except Exception:
                                pass
                            prec_str = str(prec_str)

                        prec_str = prec_str.strip()
                        if prec_str == "":
                            return ""

                        # --- 2) Split safely (ignore empty segments) ---
                        precs = [p.strip() for p in prec_str.split(',') if p and p.strip()]
                        if not precs:
                            return ""

                        # --- 3) Prepare lookups safely ---
                        ad_hoc = st.session_state.get("ad_hoc_pics", {})  # avoid KeyError
                        has_item = "Item" in df.columns
                        has_sub = "SubStandar" in df.columns
                        has_pic = "PIC" in df.columns

                        pic_list = []
                        for p in precs:
                            # Check External Registry
                            if p in ad_hoc:
                                pic_list.append(f"{p} ({ad_hoc[p]})")
                                continue

                            # Check internal DF (matching by ID or Label)
                            # Extract ID from labels like "3.1.1: Jaminan..."
                            clean_p = p.split(':', 1)[0].strip()

                            if has_pic and (has_item or has_sub):
                                cond = False
                                if has_item:
                                    cond = (df["Item"] == clean_p)
                                if has_sub:
                                    cond = cond | (df["SubStandar"] == clean_p) if isinstance(cond, pd.Series) else (df["SubStandar"] == clean_p)

                                match = df[cond] if isinstance(cond, pd.Series) else df.iloc[0:0]

                                if not match.empty:
                                    pic_val = match.iloc[0]["PIC"]
                                    # avoid "nan" showing up
                                    if pd.isna(pic_val):
                                        pic_list.append(p)
                                    else:
                                        pic_list.append(f"{p} ({pic_val})")
                                else:
                                    pic_list.append(p)
                            else:
                                # If expected columns aren't there, just return raw precedent
                                pic_list.append(p)

                        return " | ".join(pic_list)


                    summary_df['PIC Status'] = summary_df['Item Preseden/Referensi'].apply(get_precedent_pics)
                    st.dataframe(summary_df[['Uraian', 'Item Preseden/Referensi', 'PIC Status']], use_container_width=True, hide_index=True)
                else:
                    st.write("No dependencies mapped yet.")

            # st.dataframe(st.session_state.df if 'df' in st.session_state else pd.DataFrame(), use_container_width=True)

        else:
            st.info("Please upload the accreditation CSV file to begin.")

        with tab2:
        
            # st.dataframe(st.session_state.df if 'df' in st.session_state else pd.DataFrame(), use_container_width=True)


            col1, col2 = st.columns([3, 2])

            with col1:

                # --- 1. DOCUMENT HIERARCHY ---
                st.subheader("1. Document Hierarchy")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1: show_line = st.toggle("Show Hierarchy Lines", value=True, key="tracker_show_line")
                with col_t2: expand_all = st.toggle("Expand All", value=True, key="tracker_expand_all")

                # RECTIFIED RESET LOGIC: Changing the key forces the tree to reset
                if st.button("Back to Global View", key="tracker_reset_tree_btn",):
                    st.session_state.current_selection = None
                    st.session_state.tree_key += 1 # Nuclear reset of the tree widget
                    st.rerun()


                # if df exist as local variable then proceed
                if 'df' in st.session_state:
                    df = st.session_state.df
                    tree_data = build_tree_data(df)
                    
                    # Draw the tree with a dynamic key
                    tree_selection = sac.tree(
                        items=tree_data,
                        label='',
                        index=None,
                        show_line=show_line,
                        open_all=expand_all,
                        key=f"tracker_tree_widget_{st.session_state.tree_key}"
                    )

                    # Capture user selection from the tree
                    if tree_selection is not None:
                        st.session_state.current_selection = tree_selection
                # else:
                #     st.info("Please upload the accreditation CSV file in the Main Editor tab to populate the hierarchy.")


            with col2:
                # --- PROGRESS TRACKING SECTION ---
                st.subheader("Progress Tracker")

                if 'df' in st.session_state:
                    df = st.session_state.df

                    # 1. Initialize Progress Column in Master DF
                    if 'Progress' not in st.session_state.df.columns:
                        st.session_state.df['Progress'] = 0  # Default to 0

                    # 1b. Initialize URL Link Column in Master DF
                    if 'URL Link' not in st.session_state.df.columns:
                        st.session_state.df['URL Link'] = ""


                    df = st.session_state.df
                    selected_node = st.session_state.current_selection

               
                    # 2. Logic to check for "Leaf" eligibility
                    is_leaf = False
                    target_index = None

                    def is_blank(series):
                        return series.isna() | (series.astype(str).str.strip() == "")

                    def not_blank(series):
                        return ~is_blank(series)

                    if selected_node:
                        sep = " | "
                        node_id = selected_node.split(sep)[0].strip()

                        # --- Case 1: Item leaf ---
                        item_mask = (df['Item'].astype(str).str.strip() == node_id) & not_blank(df['Item'])
                        item_match = df[item_mask]
                        if not item_match.empty:
                            is_leaf = True
                            target_index = item_match.index[0]
                        else:
                            # --- Case 2: SubStandar leaf (no Item children) ---
                            sub_mask = (df['SubStandar'].astype(str).str.strip() == node_id) & not_blank(df['SubStandar'])
                            sub_match = df[sub_mask]

                            if not sub_match.empty:
                                # children exist if ANY row under this SubStandar has a non-blank Item
                                child_items = df[sub_mask & not_blank(df['Item'])]
                                has_children = not child_items.empty
                                if not has_children:
                                    is_leaf = True
                                    target_index = sub_match.index[0]

                            else:
                                # --- Case 3: Standar leaf (no SubStandar children) ---
                                std_mask = (df['Standar'].astype(str).str.strip() == node_id) & not_blank(df['Standar'])
                                std_match = df[std_mask]

                                if not std_match.empty:
                                    # children exist if ANY row under this Standar has a non-blank SubStandar
                                    child_subs = df[std_mask & not_blank(df['SubStandar'])]
                                    has_children = not child_subs.empty
                                    if not has_children:
                                        is_leaf = True
                                        # Prefer the true Standar header row (SubStandar blank & Item blank)
                                        header = df[std_mask & is_blank(df['SubStandar']) & is_blank(df['Item'])]
                                        target_index = (header.index[0] if not header.empty else std_match.index[0])


                    # 3. Dynamic UI Based on Eligibility
                    if selected_node:
                        if is_leaf:
                            st.success(f"📝 Updating Progress for: **{selected_node}**")

                            # Current progress
                            try:
                                current_val = int(df.at[target_index, 'Progress'])
                            except Exception:
                                current_val = 0

                            options = [0, 25, 50, 75, 100]
                            default_ix = options.index(current_val) if current_val in options else 0

                            new_progress = st.selectbox(
                                "Select Completion Percentage",
                                options=options,
                                index=default_ix,
                                format_func=lambda x: f"{x}%",
                                key=f"progress_sel_{node_id}"
                            )

                            # --- NEW: URL input (stored per task row) ---
                            current_url = df.at[target_index, 'URL Link'] if 'URL Link' in df.columns else ""
                            current_url = "" if pd.isna(current_url) else str(current_url).strip()

                            new_url = st.text_input(
                                "Associated Work URL (Drive/Notion/GitHub/etc.)",
                                value=current_url,
                                placeholder="Paste a link here (e.g., https://...)",
                                key=f"url_input_{node_id}"
                            )

                            col_btn1, col_btn2 = st.columns([1, 1])
                            with col_btn1:
                                if st.button("Update Progress", key=f"btn_progress_{node_id}"):
                                    st.session_state.df.at[target_index, 'Progress'] = new_progress
                                    st.toast(f"Progress set to {new_progress}%", icon="✅")
                                    st.rerun()

                            with col_btn2:
                                if st.button("Save URL", key=f"btn_url_{node_id}"):
                                    st.session_state.df.at[target_index, 'URL Link'] = new_url.strip()
                                    st.toast("URL saved!", icon="🔗")
                                    st.rerun()
                                
                        else:
                            st.warning("⚠️ **Selection Ineligible**")
                            st.info("Progress can only be updated for the lowest-level items. Please select a specific Item or a standalone Sub-Standard from the hierarchy tree.")
                else:
                    st.info("👈 Please select a document from the Hierarchy Tree to update its progress.")

                # 4. Progress Summary (Optional)
                # with st.expander("📈 Overall Completion Overview"):
                #     # Calculate average progress of leaf items
                #     leaf_mask = (df['Item'].notna()) | (df['SubStandar'].notna() & df['Item'].isna() & ~df['SubStandar'].isin(df['SubStandar'][df['Item'].notna()]))
                #     avg_progress = df[leaf_mask]['Progress'].mean()
                    
                #     st.metric("Total Accreditation Completion", f"{avg_progress:.1f}%")
                #     st.progress(avg_progress / 100)





            # Contoh penggunaan dalam Streamlit untuk tombol Download:
            if st.button("💾 Ekspor ke CSV"):
                final_df = convert_to_original_format(st.session_state.df)
                
                # Konversi ke CSV string
                csv_buffer = io.StringIO()
                final_df.to_csv(csv_buffer, index=False)
                csv_output = csv_buffer.getvalue()

                st.download_button(
                    label="Download Updated CSV",
                    data=csv_output,
                    file_name="data_akreditasi_updated.csv",
                    mime="text/csv",
                )

            else:
                st.info("Upload a CSV in the Main Editor tab to populate analytics.")

        with tab3:
            if 'df' in st.session_state:
                df = st.session_state.df.copy()
                st.header("Accreditation Analytics Dashboard")

                # st.dataframe(df)

                # --- 0) Normalize blanks properly (critical) ---
                for c in ['Standar','SubStandar','Item','Uraian','PIC','Item Preseden/Referensi']:
                    if c in df.columns:
                        df[c] = df[c].astype(str).str.strip()
                        df.loc[df[c].isin(["", "nan", "None"]), c] = pd.NA

                df['Bobot'] = pd.to_numeric(df.get('Bobot', 0), errors='coerce')

                # --- 1) BAB weights (Standar header rows only) ---
                bab_mask = df['SubStandar'].isna() & df['Item'].isna() & df['Standar'].notna()
                bab_weights = df.loc[bab_mask, ['Standar', 'Bobot']].copy()
                bab_weights = bab_weights.rename(columns={'Bobot': 'Bab_Weight'})

                # If a BAB has missing weight, keep it NA (don’t force 100)
                bab_weights['Bab_Weight'] = pd.to_numeric(bab_weights['Bab_Weight'], errors='coerce')

                # --- 2) SubStandar weights (Sub header rows only) ---
                sub_mask = df['SubStandar'].notna() & df['Item'].isna()
                sub_weights = df.loc[sub_mask, ['Standar', 'SubStandar', 'Bobot']].copy()
                sub_weights = sub_weights.rename(columns={'Bobot': 'Sub_Weight'})
                sub_weights['Sub_Weight'] = pd.to_numeric(sub_weights['Sub_Weight'], errors='coerce')

                # --- 3) Merge weights into working DF ---
                calc_df = df.merge(bab_weights, on='Standar', how='left')
                calc_df = calc_df.merge(sub_weights[['Standar','SubStandar','Sub_Weight']], on=['Standar','SubStandar'], how='left')

                # --- 4) Validate Sub_Weight integrity within each Standar (only if user provided any) ---
                # If a Standar has at least one non-NA Sub_Weight, then require sum≈100
                sub_sum = sub_weights.groupby('Standar', as_index=False)['Sub_Weight'].sum(min_count=1)  # min_count keeps NA if all NA
                sub_sum = sub_sum.dropna(subset=['Sub_Weight'])

                for _, r in sub_sum.iterrows():
                    s = r['Standar']
                    tot = float(r['Sub_Weight'])
                    # if abs(tot - 100.0) > 0.01:
                    #     st.warning(f"⚠️ Sub_Weight under Standar '{s}' totals {tot:.2f}% (should be 100%).")

                # --- 5) Compute Net_Bobot (strict: missing weights -> 0 contribution, not 100) ---
                # Definitions:
                # - BAB weight applies to Standar leaf and its descendants
                # - Sub_Weight applies to substandar (and items under it)
                # - Item uses its own Bobot column as item-weight under its SubStandar

                def calculate_net(row):
                    bw = row['Bab_Weight']
                    sw = row['Sub_Weight']
                    iw = row['Bobot']

                    bw = 0.0 if pd.isna(bw) else float(bw)
                    iw = 0.0 if pd.isna(iw) else float(iw)

                    if pd.notna(row['Item']):
                        # Item row: requires both bw and sw and iw
                        sw = 0.0 if pd.isna(sw) else float(sw)
                        return (bw * sw * iw) / 10000.0

                    elif pd.notna(row['SubStandar']):
                        # SubStandar header row (or leaf substandar): requires bw and its own Sub_Weight
                        sw = 0.0 if pd.isna(sw) else float(sw)
                        return (bw * sw) / 100.0

                    else:
                        # Standar header row (or leaf standar): just its BAB weight
                        return bw

                calc_df['Net_Bobot'] = calc_df.apply(calculate_net, axis=1)

                # calc_df to csv
                calc_df.to_csv("debug_calc_df.csv", index=False)

                def is_blank(s: pd.Series) -> pd.Series:
                    return s.isna() | (s.astype(str).str.strip() == "")

                def not_blank(s: pd.Series) -> pd.Series:
                    return ~is_blank(s)

                sun = calc_df[['Standar', 'SubStandar', 'Item', 'Bab_Weight', 'Sub_Weight', 'Net_Bobot']].copy()

                # Numeric safety
                sun['Bab_Weight'] = pd.to_numeric(sun['Bab_Weight'], errors='coerce')
                sun['Sub_Weight'] = pd.to_numeric(sun['Sub_Weight'], errors='coerce')
                sun['Net_Bobot']  = pd.to_numeric(sun['Net_Bobot'], errors='coerce').fillna(0.0)

                # Row types (blank-aware)
                standar_row_mask    = not_blank(sun['Standar']) & is_blank(sun['SubStandar']) & is_blank(sun['Item'])
                substandar_row_mask = not_blank(sun['SubStandar']) & is_blank(sun['Item'])
                item_row_mask       = not_blank(sun['Item'])

                # Expandable Standar: only when Sub_Weight sum ≈ 100 (counting only provided sub weights)
                sub_header = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Sub_Weight']].copy()
                sub_header = sub_header.dropna(subset=['Standar', 'SubStandar'])
                sub_sum = sub_header.dropna(subset=['Sub_Weight']).groupby('Standar')['Sub_Weight'].sum()

                expandable_standars = set(sub_sum[(sub_sum - 100.0).abs() <= 0.01].index)

                rows = []

                # A) Non-expandable Standar: add ONLY Standar node (no dummy children)
                standar_weights = sun.loc[standar_row_mask, ['Standar', 'Bab_Weight']].copy()
                standar_weights = standar_weights.dropna(subset=['Standar', 'Bab_Weight'])

                for _, r in standar_weights.iterrows():
                    s = str(r['Standar']).strip()
                    bw = float(r['Bab_Weight']) if pd.notna(r['Bab_Weight']) else 0.0
                    if s not in expandable_standars:
                        rows.append({'Standar': s, 'SubStandar': pd.NA, 'Item': pd.NA, 'Value': bw})

                # B) Expandable Standar: add item leaves (if they exist)
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

                # C) Expandable Standar: leaf SubStandar (no items underneath) — only if Sub_Weight exists
                sub_with_items = set(items['SubStandar'].unique()) if not items.empty else set()

                leaf_sub = sun.loc[substandar_row_mask, ['Standar', 'SubStandar', 'Bab_Weight', 'Sub_Weight']].copy()
                leaf_sub = leaf_sub.dropna(subset=['Standar', 'SubStandar'])
                leaf_sub = leaf_sub[leaf_sub['Standar'].isin(expandable_standars)]
                leaf_sub = leaf_sub[~leaf_sub['SubStandar'].isin(sub_with_items)]
                leaf_sub = leaf_sub.dropna(subset=['Sub_Weight'])  # if sub weight missing, skip it (no placeholder)

                if not leaf_sub.empty:
                    leaf_sub['Value'] = (leaf_sub['Bab_Weight'].fillna(0) * leaf_sub['Sub_Weight'].fillna(0)) / 100.0
                    leaf_sub = leaf_sub.groupby(['Standar', 'SubStandar'], as_index=False)['Value'].sum()
                    for _, r in leaf_sub.iterrows():
                        rows.append({
                            'Standar': str(r['Standar']).strip(),
                            'SubStandar': str(r['SubStandar']).strip(),
                            'Item': pd.NA,  # stop at substandar level
                            'Value': float(r['Value'])
                        })

                sun_df = pd.DataFrame(rows)

                pos_total = sun_df.loc[sun_df['Value'] > 0, 'Value'].sum() if not sun_df.empty else 0.0

                col1, col2 = st.columns(2)

                with col1:


                    if sun_df.empty or pos_total <= 0:
                        st.info("No positive BAB weights found to visualize yet. (Sunburst needs >0 weight to draw visible slices.)")
                        st.dataframe(standar_weights.sort_values('Standar'), use_container_width=True, hide_index=True)
                    else:
                        sun_df = sun_df[sun_df['Value'] > 0]

                        fig = px.sunburst(
                            sun_df,
                            path=['Standar', 'SubStandar', 'Item'],
                            values='Value'
                        )
                        fig.update_traces(hovertemplate="<b>%{label}</b><br>Weight: %{value:.4f}%<extra></extra>")
                        st.plotly_chart(fig, use_container_width=True)


                # --- E. PIC COUNT + DISTRIBUTE NET BOBOT ---
                # Count PICs per row (handles blanks safely)
                def count_pics(x):
                    if pd.isna(x):
                        return 0
                    parts = [p.strip() for p in str(x).split(',') if p and p.strip() and str(p).strip().lower() != "nan"]
                    return len(parts)

                calc_df["PIC_Count"] = calc_df["PIC"].apply(count_pics)

                # Divide net bobot equally among PICs (if no PIC assigned, keep as original net bobot)
                calc_df["Net_Bobot_Per_PIC"] = calc_df.apply(
                    lambda r: (r["Net_Bobot"] / r["PIC_Count"]) if r["PIC_Count"] > 0 else r["Net_Bobot"],
                    axis=1
                )



                
 

                # --- 2. DEFINE LEAF NODES ---
                sub_with_items = calc_df[calc_df['Item'].notna()]['SubStandar'].unique()
                standar_with_subs = calc_df[calc_df['SubStandar'].notna()]['Standar'].unique()
                
                leaf_mask = (calc_df['Item'].notna()) | \
                            (calc_df['SubStandar'].notna() & calc_df['Item'].isna() & ~calc_df['SubStandar'].isin(sub_with_items)) | \
                            (calc_df['Standar'].notna() & calc_df['SubStandar'].isna() & calc_df['Item'].isna() & ~calc_df['Standar'].isin(standar_with_subs))
                
                leaf_df = calc_df[leaf_mask].copy()

                # st.dataframe(leaf_df)



                with col2:

                    # --- 3. ANALYTICS RENDERING ---
                    # Total Progress = Sum(Progress * Net_Bobot) / 100 (Since Net_Bobot sum should be 100)
                    total_progress = (leaf_df['Progress'] * leaf_df['Net_Bobot']).sum() / 100
                    
                    # col_m1, col_m2, col_m3 = st.columns(3)
                 
                    st.metric("Overall Weighted Progress", f"{total_progress:.2f}%")
        
                    st.metric("Total Active Items", len(leaf_df))
        
                    # Check if the weight hierarchy is balanced (should sum to 100%)
                    total_net = leaf_df['Net_Bobot'].sum()
                    st.metric("Weight Integrity", f"{total_net:.1f}%", help="Should be 100%")
                    
                    st.progress(min(total_progress / 100, 1.0))
                st.divider()

                # --- 4. PIC WORKLOAD (Using Net Bobot for Responsibility) ---
                st.subheader("👨‍💻 PIC Responsibility (Weighted)")
                
                # Explode PICs for accurate individual attribution
                pic_master = leaf_df[leaf_df['PIC'].fillna("").str.strip() != ""].copy()

    

                # st.dataframe(pic_master)

                if not pic_master.empty:

                    pic_master['PIC'] = pic_master['PIC'].str.split(',')
                    pic_analysis = pic_master.explode('PIC')
                    pic_analysis['PIC'] = pic_analysis['PIC'].str.strip()

                    # st.dataframe(pic_analysis)


                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        # Pie chart based on Net_Bobot (Real impact on the 100% total)
                        fig_pie = px.pie(pic_analysis, values='Net_Bobot_Per_PIC', names='PIC', 
                                       title="Global Responsibility (%)", hole=0.4)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_p2:
                        # Bar chart for task counts
                        task_counts = pic_analysis['PIC'].value_counts().reset_index()
                        fig_bar = px.bar(task_counts, x='PIC', y='count', title="Task Count per PIC", text_auto=True)
                        st.plotly_chart(fig_bar, use_container_width=True)

     
                # --- 5. STANDAR PERFORMANCE ---
                st.subheader("📂 Chapter Progress Summary")
                standar_stats = leaf_df.groupby('Standar').apply(
                    lambda x: (x['Progress'] * x['Net_Bobot']).sum() / x['Net_Bobot'].sum() if x['Net_Bobot'].sum() > 0 else 0
                ).reset_index()
                standar_stats.columns = ['Standar', 'Completion (%)']
                
                fig_standar = px.bar(standar_stats, y='Standar', x='Completion (%)', orientation='h',
                                   range_x=[0, 100], color='Completion (%)', color_continuous_scale='RdYlGn', text_auto='.2f')
                st.plotly_chart(fig_standar, use_container_width=True)


                # --- 6. PIC PLANNED vs ACTUAL PROGRESS ---
                st.subheader("📊 PIC Performance: Planned Responsibility vs Actual Progress")

                if not pic_master.empty:
                    # Ensure clean numeric types
                    pic_analysis['Net_Bobot_Per_PIC'] = pd.to_numeric(pic_analysis['Net_Bobot_Per_PIC'], errors='coerce').fillna(0.0)
                    pic_analysis['Progress'] = pd.to_numeric(pic_analysis['Progress'], errors='coerce').fillna(0.0)

                    # Planned responsibility per PIC (sum of distributed weights)
                    planned = pic_analysis.groupby('PIC', as_index=False)['Net_Bobot_Per_PIC'].sum()
                    planned = planned.rename(columns={'Net_Bobot_Per_PIC': 'Planned Responsibility (%)'})

                    # Weighted average progress per PIC using distributed weights
                    def weighted_avg_progress(g):
                        w = g['Net_Bobot_Per_PIC']
                        if w.sum() <= 0:
                            return 0.0
                        return (g['Progress'] * w).sum() / w.sum()

                    avg_prog = pic_analysis.groupby('PIC').apply(weighted_avg_progress).reset_index(name='Avg Progress')

                    # Merge
                    pic_comparison = planned.merge(avg_prog, on='PIC', how='left')

                    # Actual progress share (planned responsibility * avg progress)
                    pic_comparison['Actual Progress (%)'] = (
                        pic_comparison['Planned Responsibility (%)'] * pic_comparison['Avg Progress'] / 100.0
                    )

                    # Reshape for grouped bar chart
                    plot_data = pic_comparison.melt(
                        id_vars='PIC',
                        value_vars=['Planned Responsibility (%)', 'Actual Progress (%)'],
                        var_name='Metric',
                        value_name='Percentage'
                    )

                    fig_comparison = px.bar(
                        plot_data,
                        x='PIC',
                        y='Percentage',
                        color='Metric',
                        barmode='group',
                        title="Planned Responsibility vs Actual Progress by PIC",
                        text_auto='.2f'
                    )

                    fig_comparison.update_layout(yaxis_title="Percentage (%)", xaxis_title="PIC")
                    st.plotly_chart(fig_comparison, use_container_width=True)

                    
                    # Optional: Show the data table
                    with st.expander("📋 View Detailed Numbers"):
                        st.dataframe(pic_comparison, use_container_width=True, hide_index=True)
                else:
                    st.info("No PIC assignments found. Assign PICs in the Configuration tab to see performance metrics.")
            
# ==========================================
                # PERSONAL PIC DASHBOARD
                # ==========================================
                st.divider()
                st.header("👤 PIC Analytics")
                st.markdown("*Drill down into individual PIC performance with renormalized task weights*")
                
                # Check if we have PIC data to work with
                if not pic_master.empty and 'PIC' in pic_analysis.columns:
                    
                    # Get unique PICs from the exploded data
                    available_pics = sorted(pic_analysis['PIC'].unique())
                    
                    if len(available_pics) > 0:
                        
                        # === TWO-COLUMN LAYOUT: 0.3 : 0.7 ===
                        col_left, col_right = st.columns([0.3, 0.7])
                        
                        with col_left:
                            # PIC Selection Dropdown
                            selected_pic = st.selectbox(
                                "Select PIC to analyze:",
                                options=available_pics,
                                help="Choose a PIC to view their personal task breakdown"
                            )
                            
                            # Display selected PIC in larger text
                            st.markdown(f"## {selected_pic}")
                            st.markdown("---")
                            
                            # Filter data for selected PIC
                            pic_tasks = pic_analysis[pic_analysis['PIC'] == selected_pic].copy()
                            
                            if not pic_tasks.empty:
                                # Calculate total weight for this PIC (for renormalization)
                                total_pic_weight = pic_tasks['Net_Bobot_Per_PIC'].sum()
                                
                                # Renormalize weights to 100% relative to this PIC's workload
                                if total_pic_weight > 0:
                                    pic_tasks['Relative_Weight'] = (pic_tasks['Net_Bobot_Per_PIC'] / total_pic_weight) * 100
                                else:
                                    pic_tasks['Relative_Weight'] = 0
                                
                                # Calculate actual progress contribution (relative)
                                pic_tasks['Relative_Progress'] = (pic_tasks['Relative_Weight'] * pic_tasks['Progress']) / 100
                                
                                # Display summary metrics VERTICALLY
                                st.metric(
                                    "Total Tasks Assigned",
                                    len(pic_tasks),
                                    help=f"Number of tasks assigned to {selected_pic}"
                                )
                                
                                avg_progress = (pic_tasks['Progress'] * pic_tasks['Net_Bobot_Per_PIC']).sum() / total_pic_weight if total_pic_weight > 0 else 0
                                st.metric(
                                    "Weighted Avg Progress",
                                    f"{avg_progress:.2f}%",
                                    help="Average progress weighted by task importance"
                                )
                                
                                st.metric(
                                    "Global Responsibility",
                                    f"{total_pic_weight:.2f}%",
                                    help="Percentage of total accreditation weight"
                                )
                        
                        with col_right:
                            if not pic_tasks.empty:
                                # === TASK-LEVEL PLANNED vs ACTUAL CHART ===
                                st.subheader(f"📊 Task Performance")
                                
                                # Prepare data for plotting with text truncation
                                chart_data = pic_tasks[['Uraian', 'Relative_Weight', 'Relative_Progress']].copy()
                                
                                # Function to truncate long text
                                def truncate_text(text, max_length=60):
                                    """Truncate text to max_length and add ellipsis if needed"""
                                    text_str = str(text)
                                    if len(text_str) <= max_length:
                                        return text_str
                                    else:
                                        return text_str[:max_length-3] + "..."
                                
                                # Create truncated labels for display
                                chart_data['Uraian_Display'] = chart_data['Uraian'].apply(lambda x: truncate_text(x, 60))
                                
                                # Sort by relative weight for better visualization
                                chart_data = chart_data.sort_values('Relative_Weight', ascending=True)
                                
                                # Reshape for grouped bar chart
                                plot_data = chart_data.melt(
                                    id_vars='Uraian_Display',
                                    value_vars=['Relative_Weight', 'Relative_Progress'],
                                    var_name='Metric',
                                    value_name='Percentage'
                                )
                                
                                # Rename for clarity
                                plot_data['Metric'] = plot_data['Metric'].map({
                                    'Relative_Weight': 'Planned Responsibility (%)',
                                    'Relative_Progress': 'Actual Progress (%)'
                                })
                                
                                # Create grouped bar chart
                                fig_tasks = px.bar(
                                    plot_data,
                                    y='Uraian_Display',
                                    x='Percentage',
                                    color='Metric',
                                    barmode='group',
                                    orientation='h',
                                    text_auto='.1f',
                                    color_discrete_map={
                                        'Planned Responsibility (%)': '#636EFA',
                                        'Actual Progress (%)': '#00CC96'
                                    }
                                )
                                
                                fig_tasks.update_layout(
                                    xaxis_title="Percentage (%)",
                                    yaxis_title="Task Description",
                                    height=max(400, len(chart_data) * 35),  # Dynamic height based on task count
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    # Wrap text on y-axis
                                    yaxis=dict(
                                        tickmode='linear',
                                        automargin=True
                                    ),
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                
                                st.plotly_chart(fig_tasks, use_container_width=True)
                                st.caption(f"Note: Weight percentage is relative to {selected_pic}'s total assigned workload.", text_alignment='center')
                            else:
                                st.warning(f"No tasks found for {selected_pic}")          
                                st.markdown("---")
                        
                        # === COLLABORATION TABLE ===
                        st.subheader(f"🤝 Collaboration Partners")
                        
                        # Build collaboration table by working with calc_df directly
                        collab_data = []
                        
                        for idx, row in pic_tasks.iterrows():
                            uraian = row['Uraian']
                            
                            # Create matching key
                            match_key = (
                                row['Standar'], 
                                row['SubStandar'] if pd.notna(row['SubStandar']) else '',
                                row['Item'] if pd.notna(row['Item']) else '',
                                uraian
                            )
                            
                            # Find matching row in calc_df (which has the original PIC column)
                            calc_df_match = calc_df[
                                (calc_df['Standar'] == match_key[0]) & 
                                (calc_df['SubStandar'].fillna('') == match_key[1]) & 
                                (calc_df['Item'].fillna('') == match_key[2]) &
                                (calc_df['Uraian'] == match_key[3])
                            ]
                            
                            if not calc_df_match.empty:
                                # Get the original PIC string
                                original_pic_string = calc_df_match.iloc[0]['PIC']
                                
                                # Parse all PICs from the original string
                                if pd.notna(original_pic_string) and str(original_pic_string).strip():
                                    all_pics = [p.strip() for p in str(original_pic_string).split(',') if p.strip()]
                                    
                                    # Remove the selected PIC to show only collaborators
                                    other_pics = [p for p in all_pics if p != selected_pic]
                                    
                                    if other_pics:
                                        collab_data.append({
                                            'Uraian': uraian,
                                            'Other PICs': ', '.join(other_pics)
                                        })
                                    else:
                                        # Solo task
                                        collab_data.append({
                                            'Uraian': uraian,
                                            'Other PICs': '—'
                                        })
                                else:
                                    collab_data.append({
                                        'Uraian': uraian,
                                        'Other PICs': '—'
                                    })
                            else:
                                # Fallback if no match found
                                collab_data.append({
                                    'Uraian': uraian,
                                    'Other PICs': 'N/A'
                                })
                        
                        if collab_data:
                            collab_df = pd.DataFrame(collab_data)
                            collab_df.insert(0, 'No.', range(1, len(collab_df) + 1))
                            
                            # Show table with custom column widths
                            st.dataframe(
                                collab_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'No.': st.column_config.NumberColumn(
                                        'No.',
                                        width= "1"

                                    ),
                                    'Uraian': st.column_config.TextColumn(
                                        'Task Description',
                                        width='8'
                                    ),
                                    'Other PICs': st.column_config.TextColumn(
                                        'Collaboration Partners',
                                        width='4'
                                    )
                                }
                            )
                            
                            # Summary stats
                            solo_tasks = len([d for d in collab_data if d['Other PICs'] == '—'])
                            collab_tasks = len(collab_data) - solo_tasks
                            
                            col_c1, col_c2, col_c3 = st.columns([0.4, 0.3, 0.3])
                            with col_c1:
                                st.download_button(
                                    label="📥 Download Collaboration Data",
                                    data=collab_df.to_csv(index=False),
                                    file_name=f"collaboration_{selected_pic}.csv",
                                    mime="text/csv",
                                    help="Download this PIC's collaboration details as CSV"
                                )
                            with col_c2:
                                st.metric("Solo Tasks", solo_tasks, help="Tasks assigned only to this PIC")
                            with col_c3:
                                st.metric("Collaborative Tasks", collab_tasks, help="Tasks requiring coordination with others")
                        else:
                            st.info("No collaboration data available for this PIC.")

                    
                    else:
                        st.info("No PICs available to analyze.")
                
                else:
                    st.info("No PIC assignments found. Assign PICs in the Configuration tab to see personal dashboards.")    
    
            else:
                st.info("Please upload data to generate weighted analytics.")

if __name__ == "__main__":
    main()