import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from components.theme import inject_css, COLORS
from components.cards import metric_card
from components.charts import regional_bar, stage_line, seasonal_trend_chart
from api_client import client

st.set_page_config(page_title="Portfolio Overview", layout="wide")
inject_css()

st.markdown(f'<div class="section-header">01 — Portfolio Overview</div>', unsafe_allow_html=True)
st.markdown("What is portfolio-wide quality status? Which regions are at risk?")

try:
    overview   = client.eda_overview()
    regional   = client.eda_regional()
    stage_data = client.eda_stage_analysis()
    seasonal   = client.eda_seasonal_trends()
except Exception as e:
    st.error(f"Cannot reach API: {e}. Is uvicorn running on localhost:8000?")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Lots", f"{overview.get('total_lots', 0):,}", color=COLORS["accent_green"])
with c2:
    metric_card("Degraded", f"{overview.get('degraded_pct', 0):.1f}%",
                delta=f"{overview.get('degraded_count', 0):,} lots", color=COLORS["degraded"])
with c3:
    metric_card("At Risk", f"{overview.get('at_risk_pct', 0):.1f}%", color=COLORS["at_risk"])
with c4:
    metric_card("False-Pass Lots", f"{overview.get('false_pass_count', 0):,}",
                delta="WG pass, CT fail", color=COLORS["accent_amber"])

st.markdown("---")
col_l, col_r = st.columns([1, 1])
with col_l:
    st.plotly_chart(regional_bar(regional), width='stretch')
with col_r:
    st.plotly_chart(stage_line(stage_data), width='stretch')

st.plotly_chart(seasonal_trend_chart(seasonal), width='stretch')

with st.expander("Regional Detail Table"):
    import pandas as pd
    if regional:
        st.dataframe(pd.DataFrame(regional).sort_values("degraded_pct", ascending=False),
                     width='stretch')
