import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd
from components.theme import inject_css, COLORS
from components.cards import metric_card
from components.charts import rm_radar_chart
from components.filters import rm_band_radio
from api_client import client

st.set_page_config(page_title="rm-Band Profiler", layout="wide")
inject_css()

st.markdown('<div class="section-header">06 — rm-Band Profiler</div>', unsafe_allow_html=True)
st.markdown("Which Relative Maturity bands perform best? GDD curves and cross-band comparison.")

try:
    rm_data = client.eda_rm_rankings()
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

with st.sidebar:
    band_choice = rm_band_radio("rm_band_06")
    band_label  = band_choice.split(" (")[0]

col1, col2 = st.columns([1, 2])

with col1:
    if rm_data:
        df = pd.DataFrame(rm_data)
        band_row = df[df["rm_band"] == band_label]
        if not band_row.empty:
            row = band_row.iloc[0]
            metric_card("Mean CT", f"{row['mean_ct']:.1f}",
                        delta=f"{row['lot_count']:,} lots", color=COLORS["accent_green"])
            metric_card("Degraded %", f"{row['degraded_pct']:.1f}%", color=COLORS["degraded"])
        else:
            st.info(f"No data for {band_label}")
        st.dataframe(df[["rm_band", "mean_ct", "degraded_pct", "lot_count"]]
                     .sort_values("degraded_pct"), use_container_width=True)

with col2:
    if rm_data:
        for d in rm_data:
            d["rm_band"] = d.get("rm_band", "")
        st.plotly_chart(rm_radar_chart(rm_data), use_container_width=True)

st.markdown("---")
st.subheader("Cross-Band Comparison")
if rm_data:
    df = pd.DataFrame(rm_data)
    import plotly.express as px
    from components.theme import apply_theme
    fig = px.bar(df, x="rm_band", y="degraded_pct", color="mean_ct",
                 color_continuous_scale=["#EF4444", "#F59E0B", "#22C55E"],
                 text=df["degraded_pct"].apply(lambda x: f"{x:.1f}%"))
    fig.update_layout(title="Degradation % by rm Band", xaxis_title="rm Band",
                      yaxis_title="Degraded %")
    st.plotly_chart(apply_theme(fig), use_container_width=True)
