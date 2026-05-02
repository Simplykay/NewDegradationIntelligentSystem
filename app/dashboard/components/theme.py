"""Design system — colors, fonts, Plotly theme, CSS injection."""
import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "bg_primary":     "#0A0F0D",
    "bg_card":        "#111A14",
    "accent_orange":  "#F97316",
    "accent_green":   "#22C55E",
    "accent_amber":   "#F59E0B",
    "accent_teal":    "#14B8A6",
    "accent_blue":    "#3B82F6",
    "text_primary":   "#F0FDF4",
    "text_secondary": "#86EFAC",
    "border_subtle":  "#1E3A2A",
    "degraded":       "#EF4444",
    "at_risk":        "#F59E0B",
    "high_quality":   "#22C55E",
}

FONTS = {
    "display": "'Space Mono', monospace",
    "body":    "'DM Sans', sans-serif",
    "mono":    "'JetBrains Mono', monospace",
}

RISK_COLORS = {
    "High":    COLORS["degraded"],
    "Medium":  COLORS["at_risk"],
    "Low":     COLORS["high_quality"],
    "Degraded":     COLORS["degraded"],
    "At Risk":      COLORS["at_risk"],
    "High Quality": COLORS["high_quality"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg_primary"],
    plot_bgcolor=COLORS["bg_card"],
    font=dict(family=FONTS["mono"], color=COLORS["text_primary"], size=12),
    xaxis=dict(gridcolor=COLORS["border_subtle"], zerolinecolor=COLORS["border_subtle"],
               tickfont=dict(family=FONTS["mono"])),
    yaxis=dict(gridcolor=COLORS["border_subtle"], zerolinecolor=COLORS["border_subtle"],
               tickfont=dict(family=FONTS["mono"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(family=FONTS["mono"])),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(font=dict(family=FONTS["mono"])),
)


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
        font-family: {FONTS['body']};
    }}
    .metric-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border_subtle']};
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }}
    .metric-value {{
        font-family: {FONTS['display']};
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
    }}
    .metric-label {{
        font-family: {FONTS['body']};
        font-size: 0.85rem;
        color: {COLORS['text_secondary']};
        margin-top: 4px;
    }}
    .hero-card {{
        background: {COLORS['bg_card']};
        border: 2px solid {COLORS['accent_blue']};
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 0 20px rgba(59,130,246,0.15);
    }}
    .section-header {{
        font-family: {FONTS['display']};
        color: {COLORS['accent_green']};
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
        border-bottom: 1px solid {COLORS['border_subtle']};
        padding-bottom: 6px;
    }}
    .stDataFrame {{ background: {COLORS['bg_card']}; }}
    .stSelectbox > div, .stSlider > div {{ color: {COLORS['text_primary']}; }}
    .stButton > button {{
        background: {COLORS['accent_green']};
        color: {COLORS['bg_primary']};
        font-family: {FONTS['display']};
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
    }}
    .stButton > button:hover {{
        background: {COLORS['accent_teal']};
        color: {COLORS['bg_primary']};
    }}
    </style>
    """, unsafe_allow_html=True)
