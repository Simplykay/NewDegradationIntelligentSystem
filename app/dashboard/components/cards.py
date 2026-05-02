"""Reusable metric card components."""
import streamlit as st
from .theme import COLORS, FONTS


def metric_card(title: str, value: str, delta: str = "", color: str = None, width: str = "100%"):
    color = color or COLORS["accent_green"]
    delta_html = f'<div style="font-size:0.8rem;color:{COLORS["text_secondary"]};margin-top:4px">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border_subtle']};
                border-radius:10px;padding:18px 22px;margin-bottom:8px;width:{width}">
        <div style="font-family:{FONTS['display']};font-size:2rem;font-weight:700;color:{color};line-height:1">
            {value}
        </div>
        <div style="font-family:{FONTS['body']};font-size:0.85rem;color:{COLORS['text_secondary']};margin-top:4px">
            {title}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(risk_level: str) -> str:
    colors = {"High": COLORS["degraded"], "Medium": COLORS["at_risk"],
              "Low": COLORS["high_quality"]}
    color = colors.get(risk_level, COLORS["text_secondary"])
    return (f'<span style="background:{color}20;color:{color};padding:2px 10px;'
            f'border-radius:12px;font-family:{FONTS["mono"]};font-size:0.8rem">{risk_level}</span>')


def shelf_life_box(seasons: float, color: str = None):
    color = color or COLORS["accent_blue"]
    bars = "█" * min(int(seasons * 2), 10)
    st.markdown(f"""
    <div style="background:{COLORS['bg_card']};border:2px solid {color};border-radius:12px;
                padding:24px;box-shadow:0 0 20px {color}30;margin:12px 0">
        <div style="font-family:{FONTS['body']};color:{COLORS['text_secondary']};font-size:0.85rem;
                    text-transform:uppercase;letter-spacing:0.1em">Predicted Shelf-Life</div>
        <div style="font-family:{FONTS['display']};color:{color};font-size:2.5rem;margin:8px 0">
            {seasons:.1f} SEASONS
        </div>
        <div style="font-family:{FONTS['mono']};color:{color};font-size:1.2rem;letter-spacing:4px">
            {bars}
        </div>
        <div style="font-family:{FONTS['body']};color:{COLORS['text_secondary']};font-size:0.8rem;margin-top:8px">
            until CT drops below 60%
        </div>
    </div>
    """, unsafe_allow_html=True)
