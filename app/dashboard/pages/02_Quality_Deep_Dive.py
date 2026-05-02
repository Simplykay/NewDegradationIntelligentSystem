import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd
from components.theme import inject_css, COLORS
from components.cards import metric_card
from components.charts import ct_histogram
from api_client import client

st.set_page_config(page_title="Quality Deep Dive", layout="wide")
inject_css()

st.markdown('<div class="section-header">02 — Quality Deep Dive</div>', unsafe_allow_html=True)
st.markdown("CT distribution, false-pass analysis, physical quality metrics.")

try:
    ct_data  = client.eda_ct_distribution()
    gap      = client.eda_wg_ct_gap()
    physical = client.eda_physical_quality()
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    metric_card("Degraded (CT<60)", f"{ct_data.get('degraded_count', 0):,}", color=COLORS["degraded"])
with col2:
    metric_card("At Risk (60-80)", f"{ct_data.get('at_risk_count', 0):,}", color=COLORS["at_risk"])
with col3:
    metric_card("False-Pass Lots", f"{gap.get('false_pass_count', 0):,}",
                delta=f"{gap.get('false_pass_pct', 0):.1f}% of tested lots", color=COLORS["accent_amber"])

st.plotly_chart(ct_histogram(ct_data), use_container_width=True)

st.markdown("---")
st.subheader("Vigor Gap: WG vs CT")
cols = st.columns(3)
with cols[0]:
    metric_card("Mean WG", f"{gap.get('wg_mean', 0):.1f}%", color=COLORS["accent_green"])
with cols[1]:
    metric_card("Mean CT", f"{gap.get('ct_mean', 0):.1f}%", color=COLORS["accent_blue"])
with cols[2]:
    metric_card("Mean Vigor Gap", f"{gap.get('mean_vigor_gap', 0):.1f} pts", color=COLORS["accent_amber"])

st.markdown("---")
st.subheader("Physical Quality Stats")
if physical:
    rows = []
    for col, stats in physical.items():
        rows.append({"Feature": col, **{k: round(v, 3) for k, v in stats.items()}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
