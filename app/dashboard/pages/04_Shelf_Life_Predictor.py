"""Page 04 — ShelfSight — Shelf-Life Predictor (Primary Deliverable)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, pandas as pd
from components.theme import inject_css, COLORS
from components.cards import metric_card, shelf_life_box
from components.filters import rm_slider
from components.charts import survival_curve, multi_survival_curves, forest_plot
from api_client import client

st.set_page_config(page_title="ShelfSight — Shelf-Life Predictor", layout="wide")
inject_css()

st.markdown('<div class="section-header">04 — ShelfSight — Shelf-Life Predictor</div>',
            unsafe_allow_html=True)
st.markdown("How many seasons until this lot's CT drops below 60%? &nbsp;·&nbsp; Powered by **LotGuard · QualityScope · GradeView · ShelfSight**")

# ── Inputs ────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 2])

with col_input:
    st.markdown("**Lot Inputs**")
    rm_val   = rm_slider("rm_04")
    region   = st.selectbox("Origin Region", ["AZ", "TX", "CA", "MS", "AR", "NM"], key="region_04")
    stage    = st.selectbox("Pipeline Stage", [1, 2, 3, 4, 5], key="stage_04")
    season_yr = st.number_input("Season Year", 2017, 2026, 2022, key="yr_04")
    wg       = st.number_input("WG Current (%)", 0.0, 100.0, 80.0, key="wg_04")
    moisture = st.number_input("Moisture (%)", 0.0, 20.0, 10.0, key="moist_04")
    mech_dmg = st.number_input("Mechanical Damage (%)", 0.0, 60.0, 5.0, key="mech_04")
    irr_type = st.selectbox("Irrigation Type",
                             ["FurrowSurge", "CenterPivotNBC", "Dryland", "Flood",
                              "CenterPivotLEPA", "Sprinkler"], key="irr_04")
    dd60     = st.number_input("Cumulated DD60 (heat units)", 0.0, 6000.0, 1500.0, key="dd60_04")
    sm       = st.number_input("Avg Soil Moisture (0-1)", 0.0, 1.0, 0.4, step=0.01, key="sm_04")

    predict_btn = st.button("PREDICT SHELF-LIFE", type="primary", width='stretch')

with col_output:
    if predict_btn:
        payload = {
            "rm": rm_val, "origin_region": region, "season_yr": int(season_yr),
            "stage": stage, "wg_current": wg, "moisture": moisture,
            "mechanical_damage": mech_dmg, "irrigation_type": irr_type,
            "cumulated_dd60": dd60, "avg_soil_moisture": sm,
        }
        try:
            with st.spinner("Running LotGuard · QualityScope · GradeView · ShelfSight..."):
                result = client.predict_single(payload)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            metric_card("Degradation Probability", f"{result['degradation_prob']*100:.1f}%",
                        color=COLORS["degraded"] if result["degradation_prob"] > 0.5 else COLORS["high_quality"])
        with c2:
            metric_card("Predicted CT Score", f"{result['predicted_ct_score']:.1f}",
                        delta=result["quality_class"],
                        color=COLORS["degraded"] if result["predicted_ct_score"] < 60 else COLORS["high_quality"])

        shelf_life = result.get("predicted_shelf_life_seasons")
        if shelf_life:
            shelf_life_box(shelf_life)
        else:
            st.info("Train ShelfSight (run train_models.py) to get shelf-life prediction.")

        st.markdown(f"**Recommendation:** {result['recommendation']}")
        st.markdown(f"**rm Band:** {result.get('rm_band', 'Unknown')} | "
                    f"**Risk Level:** {result['risk_level']}")

        if result.get("shap_top_features"):
            st.markdown("**Top Factors (SHAP):**")
            for f in result["shap_top_features"]:
                direction = "▲ increases risk" if f["shap_value"] > 0 else "▼ reduces risk"
                st.markdown(f"- `{f['feature']}`: {f['shap_value']:+.3f} ({direction})")
    else:
        st.info("Fill in lot inputs and click PREDICT SHELF-LIFE.")

        # Show KM overall while waiting
        try:
            km = client.survival_km_overall()
            if km:
                st.plotly_chart(survival_curve(km, "Portfolio Survival Curve (all lots)"),
                                width='stretch')
        except Exception:
            pass

st.markdown("---")
# ── Multi-group KM curves ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["KM by Region", "KM by Stage", "KM by rm Band", "Cox Forest Plot"])

with tab1:
    try:
        km_r = client.survival_km_by_region()
        if km_r:
            st.plotly_chart(multi_survival_curves(km_r, "Survival by Origin Region"),
                            width='stretch')
    except Exception as e:
        st.warning(f"KM by region unavailable: {e}")

with tab2:
    try:
        km_s = client.survival_km_by_stage()
        if km_s:
            st.plotly_chart(multi_survival_curves(km_s, "Survival by Pipeline Stage"),
                            width='stretch')
    except Exception as e:
        st.warning(f"KM by stage unavailable: {e}")

with tab3:
    try:
        km_b = client.survival_km_by_rm_band()
        if km_b:
            st.plotly_chart(multi_survival_curves(km_b, "Survival by rm Band"),
                            width='stretch')
    except Exception as e:
        st.warning(f"KM by rm band unavailable: {e}")

with tab4:
    try:
        hr = client.survival_cox_hazard_ratios()
        if hr:
            st.plotly_chart(forest_plot(hr), width='stretch')
            st.dataframe(pd.DataFrame(hr), width='stretch')
    except Exception as e:
        st.warning(f"Cox hazard ratios unavailable: {e}")
