"""Cotton Seed Quality Intelligence System — Streamlit dashboard main entry."""
import streamlit as st
from components.theme import inject_css, COLORS

st.set_page_config(
    page_title="Cotton Seed Quality Intelligence System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── API health check ──────────────────────────────────────────────
from api_client import client

try:
    h = client.health()
    api_ok = h.get("status") == "healthy"
    models_ok = h.get("models_loaded", False)
    data_ok   = h.get("data_loaded",   False)
except Exception:
    api_ok = models_ok = data_ok = False

# ── Sidebar status ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0">
        <div style="font-family:'Space Mono',monospace;color:{COLORS['accent_green']};
                    font-size:1.1rem;font-weight:700">COTTON SEED QI</div>
        <div style="color:{COLORS['text_secondary']};font-size:0.75rem;margin-top:2px">
            v5.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    if api_ok:
        st.success("API Connected")
    else:
        st.error("API Offline — start uvicorn")
    if models_ok:
        st.success("Models Loaded")
    elif api_ok:
        st.warning("Models not loaded — run train_models.py")

    st.markdown("---")
    st.markdown(f"""
    <div style="color:{COLORS['text_secondary']};font-size:0.75rem">
        <b>Navigation:</b> Use the pages menu above.<br><br>
        <b>Page 04 — ShelfSight</b> is the primary business deliverable.
    </div>
    """, unsafe_allow_html=True)

# ── Home page ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:40px 0 20px">
    <div style="font-family:'Space Mono',monospace;color:{COLORS['accent_green']};
                font-size:2.5rem;font-weight:700;line-height:1">
        Cotton Seed Quality<br>Intelligence System
    </div>
    <div style="color:{COLORS['text_secondary']};font-size:1.1rem;margin-top:12px;max-width:600px">
        Four AI models — <em>LotGuard · QualityScope · GradeView · ShelfSight</em> — answering:
        how long before this lot degrades below 60% CT?
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div style="background:{COLORS['bg_card']};border:2px solid {COLORS['accent_blue']};
                border-radius:10px;padding:18px;box-shadow:0 0 18px {COLORS['accent_blue']}30">
        <div style="color:{COLORS['accent_green']};font-size:0.8rem;text-transform:uppercase;
                    letter-spacing:0.1em">Primary Deliverable</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.3rem;margin-top:6px">
            Page 04 — ShelfSight
        </div>
        <div style="color:{COLORS['text_secondary']};font-size:0.85rem;margin-top:6px">
            Enter lot characteristics → seasons-until-degradation with survival curve.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border_subtle']};
                border-radius:10px;padding:18px">
        <div style="color:{COLORS['accent_amber']};font-size:0.8rem;text-transform:uppercase;
                    letter-spacing:0.1em">Explore</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.3rem;margin-top:6px">
            Pages 01–03 — Portfolio & Field Intel
        </div>
        <div style="color:{COLORS['text_secondary']};font-size:0.85rem;margin-top:6px">
            Regional risk, quality distribution, weather-driven field intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border_subtle']};
                border-radius:10px;padding:18px">
        <div style="color:{COLORS['accent_teal']};font-size:0.8rem;text-transform:uppercase;
                    letter-spacing:0.1em">Operations</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.3rem;margin-top:6px">
            Pages 05–08 — Stage, Bands, Batch
        </div>
        <div style="color:{COLORS['text_secondary']};font-size:0.85rem;margin-top:6px">
            Pipeline stage analysis, maturity-band profiling, batch scoring with CSV download.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
<div style="color:{COLORS['text_secondary']};font-size:0.8rem">
    Navigate using the <b>pages menu</b> in the sidebar.
    Start with <b>01 Portfolio Overview</b> for a high-level view, or jump to
    <b>04 ShelfSight</b> to score a specific lot.
</div>
""", unsafe_allow_html=True)
