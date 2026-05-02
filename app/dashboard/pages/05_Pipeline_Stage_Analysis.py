import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd
from components.theme import inject_css, COLORS
from components.charts import stage_line
from api_client import client

st.set_page_config(page_title="Pipeline Stage Analysis", layout="wide")
inject_css()

st.markdown('<div class="section-header">05 — Pipeline Stage Analysis</div>', unsafe_allow_html=True)
st.markdown("At which stage is seed quality gained or lost most?")

try:
    stage_data = client.eda_stage_analysis()
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

if not stage_data:
    st.warning("No stage data available.")
    st.stop()

df = pd.DataFrame(stage_data)
st.plotly_chart(stage_line(stage_data), use_container_width=True)

st.markdown("---")
st.subheader("Stage-by-Stage Summary")
st.dataframe(df.sort_values("Stage").style.background_gradient(
    subset=["degraded_pct"], cmap="RdYlGn_r"), use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if len(df) >= 2:
        best_stage = df.loc[df["mean_ct"].idxmax()]
        st.success(f"Highest quality: Stage {int(best_stage['Stage'])} "
                   f"(mean CT = {best_stage['mean_ct']:.1f})")
with col2:
    if len(df) >= 2:
        worst_stage = df.loc[df["degraded_pct"].idxmax()]
        st.error(f"Most degraded: Stage {int(worst_stage['Stage'])} "
                 f"({worst_stage['degraded_pct']:.1f}% degraded)")
