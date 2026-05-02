import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd, plotly.graph_objects as go
from components.theme import inject_css, apply_theme, COLORS
from components.charts import ct_histogram, seasonal_trend_chart, regional_bar
from api_client import client

st.set_page_config(page_title="Full EDA", layout="wide")
inject_css()

st.markdown('<div class="section-header">07 — Full EDA</div>', unsafe_allow_html=True)
st.markdown("Complete data exploration — every distribution, every chart.")

try:
    ct_data  = client.eda_ct_distribution()
    seasonal = client.eda_seasonal_trends()
    regional = client.eda_regional()
    physical = client.eda_physical_quality()
    corr     = client.eda_correlation_matrix()
    overview = client.eda_overview()
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "CT Distribution", "Seasonal Trends", "Regional",
    "Physical Quality", "Correlation Matrix", "Summary Stats"
])

with tab1:
    st.plotly_chart(ct_histogram(ct_data), use_container_width=True)
    cols = st.columns(3)
    cols[0].metric("Degraded (<60)", ct_data.get("degraded_count", 0))
    cols[1].metric("At Risk (60-80)", ct_data.get("at_risk_count", 0))
    cols[2].metric("High Quality (>80)", ct_data.get("high_quality_count", 0))

with tab2:
    st.plotly_chart(seasonal_trend_chart(seasonal), use_container_width=True)
    if seasonal:
        st.dataframe(pd.DataFrame(seasonal), use_container_width=True)

with tab3:
    st.plotly_chart(regional_bar(regional), use_container_width=True)
    if regional:
        st.dataframe(pd.DataFrame(regional).sort_values("degraded_pct", ascending=False),
                     use_container_width=True)

with tab4:
    if physical:
        rows = [{"Feature": col, **{k: round(v, 3) for k, v in stats.items()}}
                for col, stats in physical.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Physical quality stats not available.")

with tab5:
    if corr and corr.get("matrix"):
        cols_list = corr["columns"]
        matrix    = corr["matrix"]
        fig = go.Figure(go.Heatmap(
            z=matrix, x=cols_list, y=cols_list,
            colorscale=[[0, COLORS["degraded"]], [0.5, COLORS["bg_card"]], [1, COLORS["accent_green"]]],
            zmin=-1, zmax=1, text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix],
            texttemplate="%{text}",
        ))
        fig.update_layout(title="Feature Correlation Matrix")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    else:
        st.info("Correlation matrix not available.")

with tab6:
    if overview:
        st.json(overview)
