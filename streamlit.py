import streamlit as st
import pandas as pd
import streamlit_antd_components as sac
import plotly.express as px
import plotly.io as pio
import io
import difflib
import re
import os
import json
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _result_payload(intent, status, answer_data=None, evidence_rows=None, formula_used=None, warnings=None):
    return {
        "intent": intent,
        "status": status,
        "answer_data": answer_data or {},
        "evidence_rows": evidence_rows or [],
        "formula_used": formula_used or [],
        "as_of": _now_iso(),
        "warnings": warnings or [],
    }


def _is_blank_series(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _split_pics(pic_value) -> list:
    if pd.isna(pic_value):
        return []
    return [p.strip() for p in str(pic_value).split(",") if p and p.strip() and p.strip().lower() != "nan"]


def style_analytics_figure(
    fig,
    *,
    x_title=None,
    y_title=None,
    show_legend=True,
    height=None,
    font_size=None,
    margin_left=20,
    margin_right=20,
    margin_top=20,
    margin_bottom=20,
):
    fig.update_layout(
        title=None,
        xaxis_title=x_title,
        yaxis_title=y_title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if show_legend else None,
    )
    if height is not None:
        fig.update_layout(height=height)
    if font_size is not None:
        fig.update_layout(font=dict(size=font_size))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(142,160,184,0.18)', zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def truncate_text(text, max_length=60):
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    return text_str[:max_length - 3] + "..."


def natural_sort_key(value):
    parts = re.split(r'(\d+(?:\.\d+)*)', str(value))
    key = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r'\d+(?:\.\d+)*', part):
            key.extend(int(piece) for piece in part.split('.'))
        else:
            key.append(part.lower())
    return tuple(key)


def build_chapter_progress_figure(chart_stats, progress_scale, *, for_pdf=False, y_title="Chapter"):
    chart_df = chart_stats.copy()
    if for_pdf:
        chart_df['Display_Label'] = chart_df['Display_Label'].apply(lambda x: truncate_text(x, 52))

    fig = px.bar(
        chart_df,
        y='Display_Label',
        x='Completion (%)',
        orientation='h',
        range_x=[0, 100],
        color='Completion (%)',
        color_continuous_scale=progress_scale,
        text_auto='.1f'
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Completion: %{x:.1f}%<extra></extra>"
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=chart_df['Display_Label'].tolist(),
        autorange='reversed',
    )
    style_analytics_figure(
        fig,
        x_title="Completion (%)",
        y_title=y_title,
        show_legend=False,
        height=360 if for_pdf else None,
        font_size=10 if for_pdf else None,
        margin_left=180 if for_pdf else 20,
        margin_right=28 if for_pdf else 20,
    )
    return fig


def build_pic_planned_vs_actual_figure(plot_data, planned_color, actual_color, *, for_pdf=False):
    fig = px.bar(
        plot_data,
        x='PIC',
        y='Percentage',
        color='Metric',
        barmode='group',
        text_auto='.1f',
        color_discrete_map={
            'Planned Responsibility (%)': planned_color,
            'Actual Progress (%)': actual_color
        }
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<extra></extra>"
    )
    style_analytics_figure(
        fig,
        x_title="PIC",
        y_title="Percentage (%)",
        height=230 if for_pdf else None,
        font_size=9 if for_pdf else None,
        margin_left=42 if for_pdf else 20,
        margin_right=28 if for_pdf else 20,
    )
    return fig


def build_substandar_progress_stats(leaf_df, raw_df, target_standar):
    target_standar = str(target_standar).strip()
    target_leaf_df = leaf_df[leaf_df['Standar'].astype(str).str.strip() == target_standar].copy()
    if target_leaf_df.empty:
        return pd.DataFrame(columns=['SubStandar', 'Completion (%)', 'SubStandar_Name', 'Display_Label'])

    substandar_names = raw_df[
        raw_df['Standar'].astype(str).str.strip().eq(target_standar)
        & raw_df['SubStandar'].notna()
        & raw_df['Item'].isna()
    ][['SubStandar', 'Uraian']].copy()
    substandar_names = substandar_names.rename(columns={'Uraian': 'SubStandar_Name'})

    substandar_stats = target_leaf_df.groupby('SubStandar').apply(
        lambda x: (x['Progress'] * x['Net_Bobot']).sum() / x['Net_Bobot'].sum() if x['Net_Bobot'].sum() > 0 else 0.0
    ).reset_index(name='Completion (%)')
    substandar_stats = substandar_stats.merge(substandar_names, on='SubStandar', how='left')
    substandar_stats['Display_Label'] = (
        substandar_stats['SubStandar'].astype(str) + ": " + substandar_stats['SubStandar_Name'].fillna("")
    )
    return substandar_stats


def build_mixed_chapter_progress_stats(leaf_df, raw_df, expanded_standar):
    chapter_names = raw_df[
        raw_df["SubStandar"].isna() & raw_df["Item"].isna() & raw_df["Standar"].notna()
    ][["Standar", "Uraian"]].copy()
    chapter_names = chapter_names.rename(columns={"Uraian": "Standar_Name"})

    standar_stats = leaf_df.groupby("Standar").apply(
        lambda x: (x["Progress"] * x["Net_Bobot"]).sum() / x["Net_Bobot"].sum() if x["Net_Bobot"].sum() > 0 else 0.0
    ).reset_index(name="Completion (%)")
    standar_stats = standar_stats.merge(chapter_names, on="Standar", how="left")
    standar_stats["Display_Label"] = standar_stats["Standar"].astype(str) + ": " + standar_stats["Standar_Name"].fillna("")

    expanded_standar = str(expanded_standar).strip()
    non_expanded_stats = standar_stats[standar_stats["Standar"].astype(str).str.strip() != expanded_standar].copy()
    non_expanded_stats["Sort_Key"] = non_expanded_stats["Standar"].astype(str).str.strip()

    expanded_substandar_stats = build_substandar_progress_stats(leaf_df, raw_df, expanded_standar).copy()
    if not expanded_substandar_stats.empty:
        expanded_substandar_stats["Sort_Key"] = expanded_substandar_stats["SubStandar"].astype(str).str.strip()
        expanded_substandar_stats = expanded_substandar_stats[["Completion (%)", "Display_Label", "Sort_Key"]]

    combined_stats = pd.concat(
        [
            non_expanded_stats[["Completion (%)", "Display_Label", "Sort_Key"]],
            expanded_substandar_stats,
        ],
        ignore_index=True,
    )
    combined_stats = combined_stats.sort_values("Sort_Key", ascending=True, key=lambda s: s.map(natural_sort_key))
    return combined_stats


def build_pic_completion_rate_figure(pic_completion, progress_scale, *, for_pdf=False):
    fig = px.bar(
        pic_completion,
        y='PIC',
        x='Completion Rate (%)',
        orientation='h',
        range_x=[0, 100],
        color='Completion Rate (%)',
        color_continuous_scale=progress_scale,
        text_auto='.1f'
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Completion Rate: %{x:.1f}%<extra></extra>"
    )
    style_analytics_figure(
        fig,
        x_title="Completion Rate (%)",
        y_title="PIC",
        show_legend=False,
        height=230 if for_pdf else None,
        font_size=9 if for_pdf else None,
        margin_left=70 if for_pdf else 20,
        margin_right=28 if for_pdf else 20,
    )
    return fig


def figure_to_png_bytes(fig, width, height, scale=2):
    return pio.to_image(fig, format="png", width=width, height=height, scale=scale)


def draw_wrapped_text(pdf, text, x, y_top, max_width, line_height=11):
    words = str(text).split()
    lines = []
    current_line = ""

    for word in words:
        trial = word if not current_line else f"{current_line} {word}"
        if pdf.stringWidth(trial, "Helvetica", 8.5) <= max_width:
            current_line = trial
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    current_y = y_top
    for line in lines:
        pdf.drawString(x, current_y, line)
        current_y -= line_height
    return current_y


def draw_image_centered(pdf, image_reader, box_x, box_y, box_width, box_height, image_width, image_height):
    scale = min(box_width / image_width, box_height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    draw_x = box_x + (box_width - draw_width) / 2
    draw_y = box_y + (box_height - draw_height) / 2
    pdf.drawImage(
        image_reader,
        draw_x,
        draw_y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask='auto'
    )


def build_summary_pdf_bytes(total_progress, fig_standar, fig_comparison, fig_completion, generated_at=None):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed.")

    generated_at = generated_at or datetime.now()
    buffer = io.BytesIO()
    page_width, page_height = landscape(A4)
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))

    margin = 28
    gutter = 16
    header_h = 74
    section_gap = 16
    caption_h = 28
    content_top = page_height - margin - header_h
    content_height = content_top - margin
    left_w = (page_width - (margin * 2) - gutter) * 0.56
    right_w = (page_width - (margin * 2) - gutter) - left_w
    right_chart_h = (content_height - section_gap - (caption_h * 2)) / 2
    left_chart_h = content_height - caption_h

    pdf.setTitle("Accreditation Analytics Summary")
    pdf.setFillColor(colors.HexColor("#122033"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, page_height - margin - 8, "Accreditation Analytics Summary")

    pdf.setFillColor(colors.HexColor("#617187"))
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        margin,
        page_height - margin - 24,
        f"Generated at {generated_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    kpi_x = page_width - margin - 185
    kpi_y = page_height - margin - 52
    pdf.setStrokeColor(colors.HexColor("#D7DEE8"))
    pdf.setFillColor(colors.HexColor("#F7FAFC"))
    pdf.roundRect(kpi_x, kpi_y, 185, 46, 10, stroke=1, fill=1)
    pdf.setFillColor(colors.HexColor("#617187"))
    pdf.setFont("Helvetica", 9)
    pdf.drawString(kpi_x + 12, kpi_y + 29, "Overall Weighted Progress")
    pdf.setFillColor(colors.HexColor("#1FA971"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(kpi_x + 12, kpi_y + 10, f"{total_progress:.2f}%")

    chapter_img_width = 920
    chapter_img_height = 980
    bottom_img_width = 600
    bottom_img_height = 420

    chapter_img = ImageReader(io.BytesIO(figure_to_png_bytes(fig_standar, width=chapter_img_width, height=chapter_img_height)))
    comparison_img = ImageReader(io.BytesIO(figure_to_png_bytes(fig_comparison, width=bottom_img_width, height=bottom_img_height)))
    completion_img = ImageReader(io.BytesIO(figure_to_png_bytes(fig_completion, width=bottom_img_width, height=bottom_img_height)))

    left_x = margin
    right_x = margin + left_w + gutter
    chart_y = margin + caption_h

    draw_image_centered(
        pdf,
        chapter_img,
        left_x,
        chart_y,
        left_w,
        left_chart_h,
        chapter_img_width,
        chapter_img_height,
    )

    top_right_y = margin + caption_h + right_chart_h + section_gap + caption_h
    draw_image_centered(
        pdf,
        comparison_img,
        right_x,
        top_right_y,
        right_w,
        right_chart_h,
        bottom_img_width,
        bottom_img_height,
    )
    draw_image_centered(
        pdf,
        completion_img,
        right_x,
        margin + caption_h,
        right_w,
        right_chart_h,
        bottom_img_width,
        bottom_img_height,
    )

    pdf.setFillColor(colors.HexColor("#243447"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(left_x, margin + 16, "Ringkasan Progres per Bab")
    pdf.setFont("Helvetica", 8.5)
    draw_wrapped_text(
        pdf,
        "Grafik ini menunjukkan tingkat penyelesaian tertimbang pada setiap bab akreditasi sehingga pembaca dapat melihat bab mana yang sudah maju dan bab mana yang masih tertinggal.",
        left_x,
        margin + 5,
        left_w - 8,
    )

    upper_caption_y = top_right_y - 14
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(right_x, upper_caption_y, "Perbandingan Beban dan Realisasi PIC")
    pdf.setFont("Helvetica", 8.5)
    draw_wrapped_text(
        pdf,
        "Grafik ini membandingkan porsi tanggung jawab yang direncanakan untuk setiap PIC dengan progres aktual yang sudah dihasilkan dari tugas yang mereka pegang.",
        right_x,
        upper_caption_y - 11,
        right_w - 8,
    )

    lower_caption_y = margin + 16
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(right_x, lower_caption_y, "Tingkat Penyelesaian Individual PIC")
    pdf.setFont("Helvetica", 8.5)
    draw_wrapped_text(
        pdf,
        "Grafik ini menunjukkan persentase penyelesaian kerja masing-masing PIC berdasarkan bobot tugas yang menjadi tanggung jawabnya.",
        right_x,
        lower_caption_y - 11,
        right_w - 8,
    )

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def prepare_chatbot_data(df: pd.DataFrame) -> dict:
    """
    Build normalized and derived dataframes used by chatbot handlers.
    Mirrors analytics formulas used in the dashboard.
    """
    if df is None or df.empty:
        return {
            "status": "empty",
            "error": "No data available.",
            "raw_df": pd.DataFrame(),
            "calc_df": pd.DataFrame(),
            "leaf_df": pd.DataFrame(),
            "pic_analysis": pd.DataFrame(),
            "validation_issues": {},
            "available_pics": [],
        }

    working_df = df.copy()

    for col in ["Standar", "SubStandar", "Item", "Uraian", "PIC", "Item Preseden/Referensi", "URL Link"]:
        if col not in working_df.columns:
            working_df[col] = ""
        working_df[col] = working_df[col].astype(str).str.strip()
        working_df.loc[working_df[col].isin(["", "nan", "None"]), col] = pd.NA

    if "Bobot" not in working_df.columns:
        working_df["Bobot"] = 0
    if "Progress" not in working_df.columns:
        working_df["Progress"] = 0

    raw_bobot = working_df["Bobot"].copy()
    raw_progress = working_df["Progress"].copy()

    working_df["Bobot"] = pd.to_numeric(working_df["Bobot"], errors="coerce")
    working_df["Progress"] = pd.to_numeric(working_df["Progress"], errors="coerce")

    invalid_bobot_mask = raw_bobot.notna() & working_df["Bobot"].isna()
    invalid_progress_mask = raw_progress.notna() & working_df["Progress"].isna()
    negative_bobot_mask = working_df["Bobot"].fillna(0) < 0
    negative_progress_mask = working_df["Progress"].fillna(0) < 0
    over_progress_mask = working_df["Progress"].fillna(0) > 100

    working_df["Bobot"] = working_df["Bobot"].fillna(0.0)
    working_df["Progress"] = working_df["Progress"].fillna(0.0).clip(lower=0.0, upper=100.0)

    validation_issues = {
        "Invalid weight values converted to 0": int(invalid_bobot_mask.sum()),
        "Negative weight values found": int(negative_bobot_mask.sum()),
        "Invalid progress values converted to 0": int(invalid_progress_mask.sum()),
        "Negative progress values clamped to 0": int(negative_progress_mask.sum()),
        "Progress values above 100 clamped to 100": int(over_progress_mask.sum()),
    }

    bab_mask = working_df["SubStandar"].isna() & working_df["Item"].isna() & working_df["Standar"].notna()
    bab_weights = working_df.loc[bab_mask, ["Standar", "Bobot"]].copy().rename(columns={"Bobot": "Bab_Weight"})
    bab_weights["Bab_Weight"] = pd.to_numeric(bab_weights["Bab_Weight"], errors="coerce")

    sub_mask = working_df["SubStandar"].notna() & working_df["Item"].isna()
    sub_weights = working_df.loc[sub_mask, ["Standar", "SubStandar", "Bobot"]].copy().rename(
        columns={"Bobot": "Sub_Weight"}
    )
    sub_weights["Sub_Weight"] = pd.to_numeric(sub_weights["Sub_Weight"], errors="coerce")

    calc_df = working_df.merge(bab_weights, on="Standar", how="left")
    calc_df = calc_df.merge(sub_weights[["Standar", "SubStandar", "Sub_Weight"]], on=["Standar", "SubStandar"], how="left")

    def calculate_net(row):
        bw = 0.0 if pd.isna(row["Bab_Weight"]) else float(row["Bab_Weight"])
        sw = 0.0 if pd.isna(row["Sub_Weight"]) else float(row["Sub_Weight"])
        iw = 0.0 if pd.isna(row["Bobot"]) else float(row["Bobot"])

        if pd.notna(row["Item"]):
            return (bw * sw * iw) / 10000.0
        if pd.notna(row["SubStandar"]):
            return (bw * sw) / 100.0
        return bw

    calc_df["Net_Bobot"] = calc_df.apply(calculate_net, axis=1)

    calc_df["PIC_Count"] = calc_df["PIC"].apply(lambda x: len(_split_pics(x)))
    calc_df["Net_Bobot_Per_PIC"] = calc_df.apply(
        lambda r: (r["Net_Bobot"] / r["PIC_Count"]) if r["PIC_Count"] > 0 else r["Net_Bobot"],
        axis=1,
    )

    sub_with_items = calc_df[calc_df["Item"].notna()]["SubStandar"].unique()
    standar_with_subs = calc_df[calc_df["SubStandar"].notna()]["Standar"].unique()

    leaf_mask = (
        (calc_df["Item"].notna())
        | (calc_df["SubStandar"].notna() & calc_df["Item"].isna() & ~calc_df["SubStandar"].isin(sub_with_items))
        | (
            calc_df["Standar"].notna()
            & calc_df["SubStandar"].isna()
            & calc_df["Item"].isna()
            & ~calc_df["Standar"].isin(standar_with_subs)
        )
    )
    leaf_df = calc_df[leaf_mask].copy()

    pic_master = leaf_df[leaf_df["PIC"].fillna("").astype(str).str.strip() != ""].copy()
    if not pic_master.empty:
        pic_master["PIC"] = pic_master["PIC"].str.split(",")
        pic_analysis = pic_master.explode("PIC")
        pic_analysis["PIC"] = pic_analysis["PIC"].astype(str).str.strip()
        pic_analysis = pic_analysis[pic_analysis["PIC"] != ""].copy()
    else:
        pic_analysis = pd.DataFrame(columns=list(leaf_df.columns))

    available_pics = sorted(pic_analysis["PIC"].dropna().unique().tolist()) if not pic_analysis.empty else []

    return {
        "status": "ok",
        "raw_df": working_df,
        "calc_df": calc_df,
        "leaf_df": leaf_df,
        "pic_analysis": pic_analysis,
        "validation_issues": validation_issues,
        "available_pics": available_pics,
    }


def normalize_pic(raw_pic: str, available_pics: list, pic_aliases: dict | None = None) -> dict:
    if not raw_pic or not str(raw_pic).strip():
        return {"ok": False, "error": "PIC is required.", "candidates": available_pics[:5]}

    pic_aliases = pic_aliases or {}
    normalized_input = str(raw_pic).strip().upper()

    if normalized_input in pic_aliases:
        normalized_input = str(pic_aliases[normalized_input]).strip().upper()

    canonical = {str(p).strip().upper(): str(p).strip() for p in available_pics}
    if normalized_input in canonical:
        return {"ok": True, "pic": canonical[normalized_input], "candidates": []}

    close_keys = difflib.get_close_matches(normalized_input, list(canonical.keys()), n=3, cutoff=0.7)
    candidates = [canonical[k] for k in close_keys]
    return {
        "ok": False,
        "error": f"PIC '{raw_pic}' not found.",
        "candidates": candidates,
    }


def handle_overall_progress_by_pic(pic: str, prepared_data: dict) -> dict:
    intent = "overall_progress_by_pic"
    pic_analysis = prepared_data.get("pic_analysis", pd.DataFrame())
    if pic_analysis.empty:
        return _result_payload(intent, "empty", warnings=["No PIC assignments found in current data."])

    pic_rows = pic_analysis[pic_analysis["PIC"] == pic].copy()
    if pic_rows.empty:
        return _result_payload(intent, "empty", warnings=[f"No task rows found for PIC '{pic}'."])

    planned = float(pic_rows["Net_Bobot_Per_PIC"].sum())
    if planned > 0:
        avg_progress = float((pic_rows["Progress"] * pic_rows["Net_Bobot_Per_PIC"]).sum() / planned)
    else:
        avg_progress = 0.0
    actual = planned * avg_progress / 100.0

    evidence_cols = ["Standar", "SubStandar", "Item", "Uraian", "Progress", "Net_Bobot_Per_PIC"]
    if "URL Link" in pic_rows.columns:
        evidence_cols.append("URL Link")
    evidence_rows = pic_rows[evidence_cols].head(20).to_dict("records")

    return _result_payload(
        intent,
        "ok",
        answer_data={
            "pic": pic,
            "planned_responsibility_pct": round(planned, 4),
            "weighted_avg_progress_pct": round(avg_progress, 4),
            "actual_progress_share_pct": round(actual, 4),
            "task_count": int(len(pic_rows)),
        },
        evidence_rows=evidence_rows,
        formula_used=[
            "planned_responsibility_pct = sum(Net_Bobot_Per_PIC)",
            "weighted_avg_progress_pct = sum(Progress * Net_Bobot_Per_PIC) / sum(Net_Bobot_Per_PIC)",
            "actual_progress_share_pct = planned_responsibility_pct * weighted_avg_progress_pct / 100",
        ],
    )


def handle_tasks_by_pic(pic: str, prepared_data: dict) -> dict:
    intent = "tasks_by_pic"
    pic_analysis = prepared_data.get("pic_analysis", pd.DataFrame())
    if pic_analysis.empty:
        return _result_payload(intent, "empty", warnings=["No PIC assignments found in current data."])

    pic_rows = pic_analysis[pic_analysis["PIC"] == pic].copy()
    if pic_rows.empty:
        return _result_payload(intent, "empty", warnings=[f"No tasks found for PIC '{pic}'."])

    total_weight = float(pic_rows["Net_Bobot_Per_PIC"].sum())
    if total_weight > 0:
        pic_rows["Relative_Weight"] = (pic_rows["Net_Bobot_Per_PIC"] / total_weight) * 100.0
    else:
        pic_rows["Relative_Weight"] = 0.0

    base_cols = ["Standar", "SubStandar", "Item", "Uraian", "Progress", "Net_Bobot_Per_PIC", "Relative_Weight"]
    if "URL Link" in pic_rows.columns:
        base_cols.append("URL Link")

    tasks_df = pic_rows[base_cols].sort_values(["Progress", "Relative_Weight"], ascending=[True, False])
    evidence_rows = tasks_df.head(50).to_dict("records")

    return _result_payload(
        intent,
        "ok",
        answer_data={
            "pic": pic,
            "task_count": int(len(tasks_df)),
            "total_responsibility_pct": round(total_weight, 4),
        },
        evidence_rows=evidence_rows,
        formula_used=[
            "Relative_Weight = Net_Bobot_Per_PIC / sum(Net_Bobot_Per_PIC for selected PIC) * 100",
        ],
    )


def handle_stalled_tasks_by_pic(pic: str, prepared_data: dict, threshold: float = 0.0) -> dict:
    intent = "stalled_tasks_by_pic"
    pic_analysis = prepared_data.get("pic_analysis", pd.DataFrame())
    if pic_analysis.empty:
        return _result_payload(intent, "empty", warnings=["No PIC assignments found in current data."])

    pic_rows = pic_analysis[pic_analysis["PIC"] == pic].copy()
    if pic_rows.empty:
        return _result_payload(intent, "empty", warnings=[f"No tasks found for PIC '{pic}'."])

    stalled_df = pic_rows[pic_rows["Progress"] <= threshold].copy()
    total_weight = float(pic_rows["Net_Bobot_Per_PIC"].sum())
    stalled_weight = float(stalled_df["Net_Bobot_Per_PIC"].sum()) if not stalled_df.empty else 0.0
    impacted_workload = (stalled_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    evidence_cols = ["Standar", "SubStandar", "Item", "Uraian", "Progress", "Net_Bobot_Per_PIC"]
    if "URL Link" in stalled_df.columns:
        evidence_cols.append("URL Link")
    evidence_rows = stalled_df[evidence_cols].sort_values("Progress", ascending=True).head(50).to_dict("records")

    status = "ok" if not stalled_df.empty else "empty"
    return _result_payload(
        intent,
        status,
        answer_data={
            "pic": pic,
            "threshold_pct": float(threshold),
            "stalled_task_count": int(len(stalled_df)),
            "total_task_count": int(len(pic_rows)),
            "stalled_workload_pct": round(stalled_weight, 4),
            "impacted_workload_within_pic_pct": round(impacted_workload, 4),
        },
        evidence_rows=evidence_rows,
        formula_used=[
            "stalled_task = Progress <= threshold_pct",
            "impacted_workload_within_pic_pct = sum(Net_Bobot_Per_PIC stalled) / sum(Net_Bobot_Per_PIC all PIC tasks) * 100",
        ],
        warnings=[] if not stalled_df.empty else [f"No stalled tasks found for PIC '{pic}' at threshold <= {threshold}%."],
    )


def handle_collaboration_partners_by_pic(pic: str, prepared_data: dict) -> dict:
    intent = "collaboration_partners_by_pic"
    pic_analysis = prepared_data.get("pic_analysis", pd.DataFrame())
    calc_df = prepared_data.get("calc_df", pd.DataFrame())
    if pic_analysis.empty:
        return _result_payload(intent, "empty", warnings=["No PIC assignments found in current data."])

    pic_rows = pic_analysis[pic_analysis["PIC"] == pic].copy()
    if pic_rows.empty:
        return _result_payload(intent, "empty", warnings=[f"No tasks found for PIC '{pic}'."])

    collab_records = []
    for _, row in pic_rows.iterrows():
        matched = calc_df[
            (calc_df["Standar"] == row.get("Standar"))
            & (calc_df["SubStandar"].fillna("") == (row.get("SubStandar") if pd.notna(row.get("SubStandar")) else ""))
            & (calc_df["Item"].fillna("") == (row.get("Item") if pd.notna(row.get("Item")) else ""))
            & (calc_df["Uraian"] == row.get("Uraian"))
        ]
        original_pic_string = matched.iloc[0]["PIC"] if not matched.empty else ""
        all_pics = _split_pics(original_pic_string)
        others = [p for p in all_pics if p != pic]
        collab_records.append(
            {
                "Standar": row.get("Standar"),
                "SubStandar": row.get("SubStandar"),
                "Item": row.get("Item"),
                "Uraian": row.get("Uraian"),
                "Other PICs": ", ".join(others) if others else "-",
            }
        )

    collab_df = pd.DataFrame(collab_records)
    solo_tasks = int((collab_df["Other PICs"] == "-").sum()) if not collab_df.empty else 0
    collaborative_tasks = int(len(collab_df) - solo_tasks)

    return _result_payload(
        intent,
        "ok",
        answer_data={
            "pic": pic,
            "task_count": int(len(collab_df)),
            "solo_tasks": solo_tasks,
            "collaborative_tasks": collaborative_tasks,
        },
        evidence_rows=collab_df.head(50).to_dict("records"),
        formula_used=[
            "Other PICs = assigned PIC list minus selected PIC",
            "solo_tasks = count(Other PICs == '-')",
        ],
    )

def handle_overall_status_summary(prepared_data: dict) -> dict:
    intent = "overall_status_summary"
    leaf_df = prepared_data.get("leaf_df", pd.DataFrame())
    if leaf_df.empty:
        return _result_payload(intent, "empty", warnings=["No leaf tasks found to summarize."])

    total_progress = float((leaf_df["Progress"] * leaf_df["Net_Bobot"]).sum() / 100.0)
    total_active_items = int(len(leaf_df))
    weight_integrity = float(leaf_df["Net_Bobot"].sum())

    evidence_cols = ["Standar", "SubStandar", "Item", "Uraian", "Progress", "Net_Bobot"]
    evidence_rows = leaf_df[evidence_cols].head(20).to_dict("records")

    return _result_payload(
        intent,
        "ok",
        answer_data={
            "overall_weighted_progress_pct": round(total_progress, 4),
            "total_active_items": total_active_items,
            "weight_integrity_pct": round(weight_integrity, 4),
        },
        evidence_rows=evidence_rows,
        formula_used=[
            "overall_weighted_progress_pct = sum(Progress * Net_Bobot) / 100",
            "weight_integrity_pct = sum(Net_Bobot across leaf tasks)",
        ],
    )


def handle_chapter_progress_summary(prepared_data: dict) -> dict:
    intent = "chapter_progress_summary"
    leaf_df = prepared_data.get("leaf_df", pd.DataFrame())
    raw_df = prepared_data.get("raw_df", pd.DataFrame())
    if leaf_df.empty:
        return _result_payload(intent, "empty", warnings=["No leaf tasks found to summarize by chapter."])

    chapter_names = raw_df[
        raw_df["SubStandar"].isna() & raw_df["Item"].isna() & raw_df["Standar"].notna()
    ][["Standar", "Uraian"]].copy()
    chapter_names = chapter_names.rename(columns={"Uraian": "Standar_Name"})

    chapter_stats = leaf_df.groupby("Standar").apply(
        lambda x: (x["Progress"] * x["Net_Bobot"]).sum() / x["Net_Bobot"].sum() if x["Net_Bobot"].sum() > 0 else 0.0
    ).reset_index(name="Completion (%)")
    chapter_stats = chapter_stats.merge(chapter_names, on="Standar", how="left")
    chapter_stats["Display_Label"] = chapter_stats["Standar"].astype(str) + ": " + chapter_stats["Standar_Name"].fillna("")
    chapter_stats = chapter_stats.sort_values("Completion (%)", ascending=True)

    return _result_payload(
        intent,
        "ok",
        answer_data={
            "chapter_count": int(len(chapter_stats)),
        },
        evidence_rows=chapter_stats.to_dict("records"),
        formula_used=[
            "chapter_completion_pct = sum(Progress * Net_Bobot) / sum(Net_Bobot) within each Standar",
        ],
    )


CHAT_INTENTS = {
    "overall_progress_by_pic",
    "tasks_by_pic",
    "stalled_tasks_by_pic",
    "collaboration_partners_by_pic",
    "overall_status_summary",
    "chapter_progress_summary",
}


def _extract_pic_from_text(user_text: str, available_pics: list) -> str | None:
    if not user_text or not available_pics:
        return None
    tokens = re.findall(r"[A-Za-z0-9_]+", user_text.upper())
    token_set = set(tokens)
    for pic in available_pics:
        p = str(pic).strip()
        if p and p.upper() in token_set:
            return p
    return None


def route_intent(user_text: str, available_pics: list) -> dict:
    text = (user_text or "").strip()
    lower = text.lower()
    pic_guess = _extract_pic_from_text(text, available_pics)

    threshold = 0.0
    if "20" in lower:
        threshold = 20.0

    stalled_keys = ["belum", "not progress", "stalled", "belum jalan", "belum progress", "belum progressed", "blocked"]
    collab_keys = ["collab", "kolabor", "collabor", "dengan siapa", "with who", "partner"]
    task_keys = ["task", "tugas", "responsible", "apa saja", "list"]
    chapter_keys = ["chapter", "standar", "per chapter", "per standar", "bab"]
    overall_keys = ["overall", "ringkasan", "summary", "status sekarang", "overall status"]
    progress_keys = ["progress", "kemajuan", "progres"]

    intent = "unsupported"
    confidence = 0.45

    if any(k in lower for k in collab_keys):
        intent = "collaboration_partners_by_pic"
        confidence = 0.9 if pic_guess else 0.75
    elif any(k in lower for k in stalled_keys):
        intent = "stalled_tasks_by_pic"
        confidence = 0.9 if pic_guess else 0.75
    elif any(k in lower for k in task_keys):
        intent = "tasks_by_pic"
        confidence = 0.9 if pic_guess else 0.75
    elif any(k in lower for k in chapter_keys) and not pic_guess:
        intent = "chapter_progress_summary"
        confidence = 0.8
    elif any(k in lower for k in overall_keys) and not pic_guess:
        intent = "overall_status_summary"
        confidence = 0.8
    elif any(k in lower for k in progress_keys) and pic_guess:
        intent = "overall_progress_by_pic"
        confidence = 0.9
    elif pic_guess:
        intent = "tasks_by_pic"
        confidence = 0.6

    return {
        "intent": intent,
        "entities": {
            "pic": pic_guess,
            "threshold": threshold,
        },
        "confidence": confidence,
        "notes": "",
    }


def run_chat_intent(user_text: str, prepared_data: dict, pic_aliases: dict | None = None) -> tuple[dict, dict]:
    route = route_intent(user_text, prepared_data.get("available_pics", []))
    intent = route["intent"]
    entities = route["entities"]

    if intent not in CHAT_INTENTS:
        payload = _result_payload(
            intent="unsupported",
            status="error",
            warnings=[
                "Unsupported question type. Try one of: overall PIC progress, PIC tasks, stalled tasks, collaboration partners, overall status, chapter summary."
            ],
        )
        return route, payload

    if intent in {"overall_progress_by_pic", "tasks_by_pic", "stalled_tasks_by_pic", "collaboration_partners_by_pic"}:
        norm = normalize_pic(entities.get("pic"), prepared_data.get("available_pics", []), pic_aliases=pic_aliases)
        if not norm.get("ok"):
            payload = _result_payload(
                intent=intent,
                status="error",
                warnings=[norm.get("error", "PIC normalization failed.")],
                answer_data={"candidates": norm.get("candidates", [])},
            )
            return route, payload
        entities["pic"] = norm["pic"]

    if intent == "overall_progress_by_pic":
        payload = handle_overall_progress_by_pic(entities["pic"], prepared_data)
    elif intent == "tasks_by_pic":
        payload = handle_tasks_by_pic(entities["pic"], prepared_data)
    elif intent == "stalled_tasks_by_pic":
        payload = handle_stalled_tasks_by_pic(entities["pic"], prepared_data, threshold=float(entities.get("threshold", 0.0)))
    elif intent == "collaboration_partners_by_pic":
        payload = handle_collaboration_partners_by_pic(entities["pic"], prepared_data)
    elif intent == "overall_status_summary":
        payload = handle_overall_status_summary(prepared_data)
    elif intent == "chapter_progress_summary":
        payload = handle_chapter_progress_summary(prepared_data)
    else:
        payload = _result_payload(intent="unsupported", status="error", warnings=["No handler found."])

    return route, payload


def format_answer_plain(route: dict, payload: dict) -> str:
    intent = payload.get("intent")
    status = payload.get("status")
    data = payload.get("answer_data", {})
    warnings = payload.get("warnings", [])
    as_of = payload.get("as_of", "")

    if status in {"error", "empty"}:
        lines = []
        if warnings:
            lines.extend([f"- {w}" for w in warnings])
        if data.get("candidates"):
            lines.append(f"- Did you mean: {', '.join(map(str, data['candidates']))}")
        lines.append("- Supported examples: 'progress PIC NAR', 'tasks for LMS', 'stalled tasks for PS', 'collaboration for MRI', 'overall status', 'chapter summary'")
        return "\n".join(lines)

    if intent == "overall_progress_by_pic":
        return (
            f"PIC {data['pic']} has weighted average progress {data['weighted_avg_progress_pct']:.2f}% "
            f"across {data['task_count']} tasks. Planned responsibility is {data['planned_responsibility_pct']:.2f}% "
            f"and actual progress share is {data['actual_progress_share_pct']:.2f}% (as of {as_of})."
        )

    if intent == "tasks_by_pic":
        return (
            f"Found {data['task_count']} tasks for PIC {data['pic']} with total responsibility "
            f"{data['total_responsibility_pct']:.2f}% (as of {as_of}). Check evidence table for task-level details."
        )

    if intent == "stalled_tasks_by_pic":
        return (
            f"PIC {data['pic']} has {data['stalled_task_count']} stalled tasks (threshold <= {data['threshold_pct']:.0f}%) "
            f"out of {data['total_task_count']} tasks. Impacted workload within PIC scope is "
            f"{data['impacted_workload_within_pic_pct']:.2f}% (as of {as_of})."
        )

    if intent == "collaboration_partners_by_pic":
        return (
            f"PIC {data['pic']} has {data['task_count']} tasks: {data['collaborative_tasks']} collaborative "
            f"and {data['solo_tasks']} solo (as of {as_of}). Check evidence table for partner details."
        )

    if intent == "overall_status_summary":
        return (
            f"Overall weighted progress is {data['overall_weighted_progress_pct']:.2f}% across "
            f"{data['total_active_items']} active items, with weight integrity {data['weight_integrity_pct']:.2f}% "
            f"(as of {as_of})."
        )

    if intent == "chapter_progress_summary":
        return (
            f"Chapter summary generated for {data['chapter_count']} chapters (as of {as_of}). "
            f"Check evidence table for completion percentages per chapter."
        )

    return "The request was processed, but no formatter matched this intent."


def _get_gemini_api_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    env_key = os.getenv("GEMINI_API_KEY", "")
    return str(secret_key or env_key or "").strip()


def format_answer_with_gemini(payload: dict, user_question: str, model_name: str = "gemini-2.0-flash") -> tuple[bool, str, str]:
    """
    Returns: (ok, text, message)
    - ok=True: text is Gemini-formatted answer
    - ok=False: text should be ignored, message explains fallback reason
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        return False, "", "Gemini API key not found. Set GEMINI_API_KEY in Streamlit secrets or environment."

    # Compact deterministic payload to constrain model output.
    llm_payload = {
        "intent": payload.get("intent"),
        "status": payload.get("status"),
        "answer_data": payload.get("answer_data", {}),
        "warnings": payload.get("warnings", []),
        "formula_used": payload.get("formula_used", []),
        "as_of": payload.get("as_of"),
        "evidence_rows": payload.get("evidence_rows", [])[:12],
    }

    system_instruction = (
        "You are a structured data narrator. Use only the provided payload.\n"
        "Rules:\n"
        "1) Never change numeric values, names, or entities.\n"
        "2) Never invent tasks, metrics, or collaborators.\n"
        "3) If status is error/empty, explain clearly and briefly.\n"
        "4) Keep response concise and practical for busy users.\n"
        "5) Preserve user's language style when possible.\n"
        "Output sections:\n"
        "Summary\n"
        "Key Points\n"
        "Evidence\n"
    )
    user_prompt = (
        f"User question:\n{user_question}\n\n"
        f"Deterministic payload:\n{json.dumps(llm_payload, ensure_ascii=True)}\n\n"
        "Write the final answer now."
    )

    # Preferred SDK: google-genai
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config={"system_instruction": system_instruction, "temperature": 0.2},
        )
        text = getattr(resp, "text", "") or ""
        if text.strip():
            return True, text.strip(), "Gemini formatting applied."
    except Exception:
        pass

    # Fallback SDK: google-generativeai
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
        resp = model.generate_content(user_prompt)
        text = getattr(resp, "text", "") or ""
        if text.strip():
            return True, text.strip(), "Gemini formatting applied."
    except Exception as e:
        return False, "", f"Gemini formatting unavailable: {str(e)}"

    return False, "", "Gemini returned empty output."

def main():
    st.title("Accreditation Monitoring")

    tab1, tab2, tab3, tab4 = st.tabs(["Configuration", "Task Tracker", "Analytics", "Chat Assistant"])

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
                with open('data_akreditasi_updated(10).csv', 'rb') as f:
                    uploaded_file = io.BytesIO(f.read())
            except FileNotFoundError:
                pass

        if uploaded_file is not None:
            # --- DATA INITIALIZATION ---
            if 'df' not in st.session_state:
                df = pd.read_csv(uploaded_file, dtype={'SubStandar': str, 'Item': str, 'Standar': str}, index_col=False)
                df.columns = df.columns.str.strip()


                # drop fully empty rows (important!)
                df = df.dropna(how='all')

                # also drop rows where all hierarchy columns are blank/NaN
                hier_cols = ['Standar', 'SubStandar', 'Item', 'Uraian']
                df = df.dropna(subset=hier_cols, how='all')




                df['Standar'] = df['Standar'].ffill()
                df = df.drop_duplicates(subset=["Standar","SubStandar","Item"], keep="first")

                def _blank(s):
                    return s.isna() | (s.astype(str).str.strip() == "")

                # Drop rows where hierarchy is "Standar only" but the label is empty
                # (these are usually the stray empty lines that got filled by ffill)
                df = df[~(
                    df['Standar'].notna() &
                    _blank(df['SubStandar']) &
                    _blank(df['Item']) &
                    _blank(df['Uraian'])
                )].copy()



                if 'Bobot' not in df.columns: df['Bobot'] = 0.0
                if 'PIC' not in df.columns: df['PIC'] = ""
                if 'Item Preseden/Referensi' not in df.columns: df['Item Preseden/Referensi'] = ""
                # Strip whitespace from string columns
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                st.session_state.df = df
            
            # --- SELECTION STATE INITIALIZATION ---
            if 'current_selection' not in st.session_state:
                st.session_state.current_selection = None
            if 'tree_key' not in st.session_state:
                st.session_state.tree_key = 0

            df = st.session_state.df

            col1, col2 = st.columns([5,4])

            with col1:

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

            with col2:

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

                # st.write(selected_node)

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

                            options = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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

                planned_color = '#355CDE'
                actual_color = '#1FA971'
                neutral_color = '#8EA0B8'
                progress_scale = ['#C0392B', '#F39C12', '#27AE60']

                def style_figure(fig, *, x_title=None, y_title=None, show_legend=True):
                    return style_analytics_figure(
                        fig,
                        x_title=x_title,
                        y_title=y_title,
                        show_legend=show_legend,
                    )

                # --- NEW: Chapter Reference Table ---
                with st.expander("📖 Chapter Reference Lookup (Standar ID to Description)", expanded=False):
                    # Get Standar names (Uraian from header rows)
                    standar_lookup = df[df['SubStandar'].isna() & df['Item'].isna() & df['Standar'].notna()][['Standar', 'Uraian']]
                    standar_lookup = standar_lookup.rename(columns={'Uraian': 'Chapter Description'}).sort_values('Standar')
                    st.dataframe(standar_lookup, use_container_width=True, hide_index=True)

                # --- 0) Normalize blanks properly (critical) ---
                for c in ['Standar','SubStandar','Item','Uraian','PIC','Item Preseden/Referensi']:
                    if c in df.columns:
                        df[c] = df[c].astype(str).str.strip()
                        df.loc[df[c].isin(["", "nan", "None"]), c] = pd.NA

                if 'Bobot' not in df.columns:
                    df['Bobot'] = 0
                if 'Progress' not in df.columns:
                    df['Progress'] = 0

                raw_bobot = df['Bobot'].copy()
                raw_progress = df['Progress'].copy()

                df['Bobot'] = pd.to_numeric(df['Bobot'], errors='coerce')
                df['Progress'] = pd.to_numeric(df['Progress'], errors='coerce')

                invalid_bobot_mask = raw_bobot.notna() & df['Bobot'].isna()
                invalid_progress_mask = raw_progress.notna() & df['Progress'].isna()
                negative_bobot_mask = df['Bobot'].fillna(0) < 0
                negative_progress_mask = df['Progress'].fillna(0) < 0
                over_progress_mask = df['Progress'].fillna(0) > 100

                df['Bobot'] = df['Bobot'].fillna(0.0)
                df['Progress'] = df['Progress'].fillna(0.0).clip(lower=0.0, upper=100.0)

                validation_issues = {
                    "Invalid weight values converted to 0": int(invalid_bobot_mask.sum()),
                    "Negative weight values found": int(negative_bobot_mask.sum()),
                    "Invalid progress values converted to 0": int(invalid_progress_mask.sum()),
                    "Negative progress values clamped to 0": int(negative_progress_mask.sum()),
                    "Progress values above 100 clamped to 100": int(over_progress_mask.sum()),
                }

                issue_total = sum(validation_issues.values())
                if issue_total > 0:
                    with st.expander("Data Validation Summary", expanded=False):
                        st.warning("Some numeric fields needed cleanup before analytics were calculated.")
                        validation_df = pd.DataFrame(
                            [{"Issue": label, "Rows": count} for label, count in validation_issues.items() if count > 0]
                        )
                        st.dataframe(validation_df, use_container_width=True, hide_index=True)

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

                        # Remove parent rows that are not true leaves. Plotly's sunburst
                        # requires that any parent row must not also appear alongside
                        # its children as a separate row. If a Standar row exists but
                        # there are SubStandar/Item rows for the same Standar, drop
                        # the Standar-only row. Likewise drop SubStandar-only rows
                        # when Item rows exist for that SubStandar.
                        print("### Debugging Executing Pre-Logic", flush=True)      
                        if not sun_df.empty:
                            sun_df = sun_df[sun_df['Value'] > 0].copy()

                            # --- ROBUST HIERARCHY CLEANING (Fix for 'Non-leaves rows' error) ---
                            # 1. Uniformly treat all empty representations as None
                            def clean_val(x):
                                if x is None or pd.isna(x) or str(x).strip() == "" or str(x).lower() == "nan":
                                    return None
                                return str(x).strip()

                            for col in ['Standar', 'SubStandar', 'Item']:
                                if col in sun_df.columns:
                                    sun_df[col] = sun_df[col].apply(clean_val)

                            # 2. Re-identify parents with robust logic
                            # Parent Standar: A Standar that has children (SubStandar is not None)
                            parents_standar = set(sun_df.loc[sun_df['SubStandar'].notna(), 'Standar'].unique())
                            
                            # Parent SubStandar: A (Standar, SubStandar) that has children (Item is not None)
                            # We create a set of tuples for fast lookup
                            parents_sub = set(
                                sun_df.loc[sun_df['Item'].notna()]
                                      .apply(lambda x: (x['Standar'], x['SubStandar']), axis=1)
                                      .dropna() # safety
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

                            sun_df = sun_df[~(drop_standar | drop_sub)]

                        # --- DEBUG INJECTION START ---
                        print("### Debugging Sunburst Data", flush=True)
                        print(f"Sunburst DF Shape: {sun_df.shape}", flush=True)
                        debug_2 = sun_df[sun_df['Standar'] == '2']
                        if not debug_2.empty:
                            print("Rows for Standar '2':", flush=True)
                            print(debug_2.to_string(), flush=True)
                        else:
                            print("No rows found for Standar '2'", flush=True)
                        
                        print("Unique SubStandar values:", sun_df['SubStandar'].unique(), flush=True)
                        print("Unique Item values:", sun_df['Item'].unique(), flush=True)
                        # --- DEBUG INJECTION END ---

                        # st.dataframe(df)

                        fig = px.sunburst(
                            sun_df,
                            path=['Standar'],
                            values='Value',
                            
                        
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
                summary_report_pdf = None
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
                        fig_pie = px.pie(
                            pic_analysis,
                            values='Net_Bobot_Per_PIC',
                            names='PIC',
                            hole=0.5,
                            color_discrete_sequence=[planned_color, actual_color, '#F4B942', '#DD6B66', '#6C8EAD', '#7B6FD6']
                        )
                        fig_pie.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate="<b>%{label}</b><br>Responsibility: %{value:.2f}%<br>Share: %{percent}<extra></extra>"
                        )
                        style_figure(fig_pie, show_legend=False)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_p2:
                        # Bar chart for task counts
                        task_counts = pic_analysis['PIC'].value_counts().reset_index()
                        task_counts.columns = ['PIC', 'Task Count']
                        task_counts = task_counts.sort_values('Task Count', ascending=False)
                        fig_bar = px.bar(
                            task_counts,
                            x='PIC',
                            y='Task Count',
                            text_auto=True,
                            color_discrete_sequence=[neutral_color]
                        )
                        fig_bar.update_traces(
                            hovertemplate="<b>%{x}</b><br>Task Count: %{y}<extra></extra>"
                        )
                        style_figure(fig_bar, x_title="PIC", y_title="Task Count", show_legend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)

     
                # --- 5. STANDAR PERFORMANCE ---
                st.subheader("📂 Chapter Progress Summary")
                
                mixed_chapter_stats = build_mixed_chapter_progress_stats(leaf_df, df, expanded_standar='6')
                fig_standar = build_chapter_progress_figure(
                    mixed_chapter_stats,
                    progress_scale,
                    y_title="Chapter / Sub-Standard",
                )
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

                    fig_comparison = build_pic_planned_vs_actual_figure(
                        plot_data,
                        planned_color,
                        actual_color,
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)

                    # --- 7. PIC INDIVIDUAL COMPLETION RATE (New Chart) ---
                    st.subheader("🎯 PIC Performance: Individual Completion Rate (%)")
                    
                    # Calculate the completion rate: (Actual Progress / Planned Responsibility) * 100
                    # This is essentially the weighted average progress of tasks assigned to that PIC.
                    pic_completion = pic_comparison.copy()
                    pic_completion['Completion Rate (%)'] = (
                        (pic_completion['Actual Progress (%)'] / pic_completion['Planned Responsibility (%)'] * 100)
                        if not pic_completion.empty else 0
                    ).fillna(0)

                    # Sort for better visualization
                    pic_completion = pic_completion.sort_values('Completion Rate (%)', ascending=True)

                    fig_completion = build_pic_completion_rate_figure(pic_completion, progress_scale)
                    st.plotly_chart(fig_completion, use_container_width=True)

                    export_col1, export_col2 = st.columns([0.6, 0.4])
                    with export_col1:
                        st.caption("Download a one-page PDF executive summary with the overall KPI and these three charts.")
                    with export_col2:
                        if REPORTLAB_AVAILABLE:
                            try:
                                pdf_fig_standar = build_chapter_progress_figure(
                                    mixed_chapter_stats,
                                    progress_scale,
                                    for_pdf=True,
                                    y_title="Chapter / Sub-Standard",
                                )
                                pdf_fig_comparison = build_pic_planned_vs_actual_figure(
                                    plot_data,
                                    planned_color,
                                    actual_color,
                                    for_pdf=True,
                                )
                                pdf_fig_completion = build_pic_completion_rate_figure(
                                    pic_completion,
                                    progress_scale,
                                    for_pdf=True,
                                )
                                summary_report_pdf = build_summary_pdf_bytes(
                                    total_progress=total_progress,
                                    fig_standar=pdf_fig_standar,
                                    fig_comparison=pdf_fig_comparison,
                                    fig_completion=pdf_fig_completion,
                                    generated_at=datetime.now(),
                                )
                            except Exception as export_error:
                                st.warning(
                                    "PDF export is unavailable until image export dependencies are installed correctly "
                                    f"(`{export_error}`)."
                                )
                            else:
                                st.download_button(
                                    label="Download PDF Summary",
                                    data=summary_report_pdf,
                                    file_name="accreditation_analytics_summary.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                        else:
                            st.info("Install `reportlab` to enable PDF export.")

                    # Optional: Show the data table
                    with st.expander("📋 View Detailed Numbers"):
                        st.dataframe(pic_completion, use_container_width=True, hide_index=True)
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
                                        'Planned Responsibility (%)': planned_color,
                                        'Actual Progress (%)': actual_color
                                    }
                                )
                                
                                fig_tasks.update_traces(
                                    hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.1f}%<extra></extra>"
                                )
                                style_figure(fig_tasks, x_title="Percentage (%)", y_title="Task Description")
                                fig_tasks.update_layout(
                                    height=max(400, len(chart_data) * 35)
                                )
                                fig_tasks.update_xaxes(range=[0, 100])
                                fig_tasks.update_yaxes(
                                    tickmode='linear',
                                    automargin=True
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

        with tab4:
            st.header("Chat Assistant")
            st.caption("Deterministic answers from current app data. Use questions about PIC progress, tasks, stalled work, collaboration, overall status, or chapter summary.")

            if "df" not in st.session_state:
                st.info("Please upload/load data first in Configuration tab.")
            else:
                prepared_data = prepare_chatbot_data(st.session_state.df)
                if prepared_data.get("status") != "ok":
                    st.warning(prepared_data.get("error", "Unable to prepare chatbot data."))
                else:
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []
                    if "chat_logs" not in st.session_state:
                        st.session_state.chat_logs = []
                    if "chat_last_payload" not in st.session_state:
                        st.session_state.chat_last_payload = None
                    if "chat_last_route" not in st.session_state:
                        st.session_state.chat_last_route = None

                    top_c1, top_c2 = st.columns([0.65, 0.35])
                    with top_c1:
                        st.markdown(
                            f"Available PICs: `{', '.join(prepared_data.get('available_pics', [])[:20])}`"
                            if prepared_data.get("available_pics")
                            else "No PIC currently available."
                        )
                    with top_c2:
                        use_gemini = True
                        # use_gemini = st.toggle("Use Gemini Wording", value=False, key="chat_use_gemini")
                        gemini_model = "gemini-2.0-flash"
                        # gemini_model = st.text_input("Gemini Model", value="gemini-2.0-flash", key="chat_gemini_model")
                        if st.button("Clear Chat", key="clear_chat_btn", use_container_width=True):
                            st.session_state.chat_history = []
                            st.session_state.chat_logs = []
                            st.session_state.chat_last_payload = None
                            st.session_state.chat_last_route = None
                            st.rerun()

                    for msg in st.session_state.chat_history:
                        with st.chat_message(msg.get("role", "assistant")):
                            st.markdown(msg.get("content", ""))

                    user_text = st.chat_input("Ask: 'progress PIC NAR', 'tasks for LMS', 'stalled tasks for PS', 'collaboration for MRI', 'overall status'")
                    if user_text:
                        st.session_state.chat_history.append({"role": "user", "content": user_text})

                        route, payload = run_chat_intent(user_text, prepared_data)
                        answer_text = format_answer_plain(route, payload)
                        formatting_mode = "plain"
                        formatting_note = ""
                        if use_gemini:
                            ok, gemini_text, gemini_msg = format_answer_with_gemini(
                                payload=payload,
                                user_question=user_text,
                                model_name=(gemini_model or "gemini-2.0-flash").strip(),
                            )
                            if ok and gemini_text:
                                answer_text = gemini_text
                                formatting_mode = "gemini"
                                formatting_note = gemini_msg
                            else:
                                formatting_note = f"Fallback to plain formatting: {gemini_msg}"

                        st.session_state.chat_last_route = route
                        st.session_state.chat_last_payload = payload
                        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
                        st.session_state.chat_logs.append(
                            {
                                "timestamp": _now_iso(),
                                "query": user_text,
                                "intent": route.get("intent"),
                                "entities": str(route.get("entities", {})),
                                "status": payload.get("status"),
                                "formatting_mode": formatting_mode,
                                "formatting_note": formatting_note,
                            }
                        )
                        st.rerun()

                    if st.session_state.chat_last_payload:
                        payload = st.session_state.chat_last_payload
                        route = st.session_state.chat_last_route or {}
                        with st.expander("Latest Execution Details", expanded=True):
                            # st.write(
                            #     {
                            #         "intent": route.get("intent"),
                            #         "confidence": route.get("confidence"),
                            #         "entities": route.get("entities"),
                            #         "status": payload.get("status"),
                            #         "as_of": payload.get("as_of"),
                            #         "warnings": payload.get("warnings"),
                            #         "formatting_mode": (
                            #             st.session_state.chat_logs[-1].get("formatting_mode")
                            #             if st.session_state.chat_logs
                            #             else "plain"
                            #         ),
                            #         "formatting_note": (
                            #             st.session_state.chat_logs[-1].get("formatting_note")
                            #             if st.session_state.chat_logs
                            #             else ""
                            #         ),
                            #     }
                            # )
                            evidence_rows = payload.get("evidence_rows", [])
                            if evidence_rows:
                                st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
                            else:
                                st.caption("No evidence rows for this response.")

if __name__ == "__main__":
    main()

