import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd, plotly.express as px
from components.theme import inject_css, COLORS, apply_theme
from api_client import client

st.set_page_config(page_title="Field Intelligence", layout="wide")
inject_css()

st.markdown('<div class="section-header">03 — Field Intelligence</div>', unsafe_allow_html=True)
st.markdown("How does heat accumulation and irrigation affect CT? (2026 weather data)")

try:
    weather  = client.eda_weather_summary()
    irr      = client.eda_irrigation_breakdown()
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
if weather:
    if "cumulated_dd60" in weather:
        with col1:
            v = weather["cumulated_dd60"]
            st.metric("Avg Season DD60", f"{v.get('mean', 0):,.0f}",
                      delta=f"max {v.get('max', 0):,.0f}")
    if "avg_soil_moisture" in weather:
        with col2:
            v = weather["avg_soil_moisture"]
            st.metric("Avg Soil Moisture", f"{v.get('mean', 0):.3f}",
                      delta=f"range {v.get('min', 0):.3f}–{v.get('max', 0):.3f}")

if irr:
    df_irr = pd.DataFrame(irr).head(10)
    import plotly.graph_objects as go
    fig_irr = go.Figure(go.Bar(
        x=df_irr["irrigation_type"], y=df_irr["pct"],
        marker_color=COLORS["accent_teal"],
        text=[f"{p:.1f}%" for p in df_irr["pct"]], textposition="outside",
    ))
    fig_irr.update_layout(title="Irrigation Type Breakdown (%)", xaxis_title="Type", yaxis_title="% Fields")
    st.plotly_chart(apply_theme(fig_irr), use_container_width=True)

