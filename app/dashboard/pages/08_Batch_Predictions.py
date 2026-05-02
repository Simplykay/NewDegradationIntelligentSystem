import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd
from components.theme import inject_css, COLORS
from api_client import client

st.set_page_config(page_title="Batch Predictions", layout="wide")
inject_css()

st.markdown('<div class="section-header">08 — Batch Predictions</div>', unsafe_allow_html=True)
st.markdown("Upload a CSV of lots → receive CT score, quality class, and shelf-life predictions.")

st.markdown("""
**CSV format:** include columns like `rm`, `origin_region`, `season_yr`, `stage`,
`wg_current`, `moisture`, `mechanical_damage`, `actual_seed_per_lb`.
Optional: `cumulated_dd60`, `avg_soil_moisture`, `irrigation_type`.
""")

template_data = {
    "lot_id": ["LOT001", "LOT002"],
    "rm":     [110.0, 125.0],
    "origin_region": ["AZ", "TX"],
    "season_yr":     [2023, 2022],
    "stage":         [2, 3],
    "wg_current":    [82.0, 76.0],
    "moisture":      [10.2, 11.5],
    "mechanical_damage": [4.5, 7.2],
    "actual_seed_per_lb": [4800, 5200],
}
template_csv = pd.DataFrame(template_data).to_csv(index=False)
st.download_button("Download Template CSV", template_csv, "batch_template.csv", "text/csv")

uploaded = st.file_uploader("Upload lots CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.markdown(f"Uploaded: **{len(df):,} lots**")
    st.dataframe(df.head(5), width='stretch')

    if st.button("Run Batch Predictions", type="primary"):
        with st.spinner("Scoring all lots..."):
            try:
                results = client.predict_batch_df(df)

                st.success(f"Predictions complete for {len(results):,} lots")

                def risk_color(val):
                    if val == "High":   return f"background-color: {COLORS['degraded']}40"
                    if val == "Medium": return f"background-color: {COLORS['at_risk']}40"
                    return f"background-color: {COLORS['high_quality']}40"

                if "risk_level" in results.columns:
                    styled = results.style.applymap(risk_color, subset=["risk_level"])
                    st.dataframe(styled, width='stretch')
                else:
                    st.dataframe(results, width='stretch')

                csv_out = results.to_csv(index=False).encode()
                st.download_button("Download Predictions CSV", csv_out,
                                   "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
