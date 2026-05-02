"""Common sidebar filter widgets."""
import streamlit as st
from .theme import COLORS


def rm_slider(key: str = "rm_slider") -> float:
    val = st.slider("Relative Maturity (rm)", min_value=80.0, max_value=145.0,
                    value=110.0, step=1.0, key=key)
    st.caption("Bands: Ultra-Early <100 | Early 100-110 | Mid-Early 110-120 | "
               "Mid-Full 120-130 | Full >130")
    return val


def rm_band_radio(key: str = "rm_band_radio") -> str:
    return st.radio("Select rm Band", [
        "Ultra-Early (rm < 100)",
        "Early (100-110)",
        "Mid-Early (110-120)",
        "Mid-Full (120-130)",
        "Full (rm > 130)",
    ], key=key)


def region_filter(key: str = "region_filter") -> list[str]:
    regions = ["AZ", "TX", "CA", "MS", "AR", "NM"]
    return st.multiselect("Origin Region", regions, default=regions, key=key)


def stage_filter(key: str = "stage_filter") -> list[int]:
    return st.multiselect("Pipeline Stage", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key=key)


def season_filter(seasons: list[int], key: str = "season_filter") -> list[int]:
    if not seasons:
        return []
    return st.multiselect("Season Year", sorted(seasons),
                           default=sorted(seasons)[-5:], key=key)
