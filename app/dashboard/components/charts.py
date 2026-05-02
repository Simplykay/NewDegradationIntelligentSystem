"""Reusable Plotly chart builders — all use apply_theme()."""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
from .theme import apply_theme, COLORS, RISK_COLORS


def ct_histogram(data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.get("bins", []),
        y=data.get("counts", []),
        marker_color=COLORS["accent_green"],
        name="CT Distribution",
    ))
    threshold = data.get("threshold", 60)
    fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["degraded"],
                  annotation_text=f"CT={threshold} threshold",
                  annotation_font_color=COLORS["degraded"])
    fig.update_layout(title="CT Score Distribution", xaxis_title="CT Score",
                      yaxis_title="Lot Count")
    return apply_theme(fig)


def regional_bar(data: list[dict]) -> go.Figure:
    if not data:
        return go.Figure()
    regions = [d.get("region", "") for d in data]
    values  = [d.get("degraded_pct", 0) for d in data]
    colors  = [COLORS["degraded"] if v > 25 else COLORS["at_risk"] if v > 15 else COLORS["high_quality"]
               for v in values]
    fig = go.Figure(go.Bar(x=regions, y=values, marker_color=colors,
                           text=[f"{v:.1f}%" for v in values], textposition="outside"))
    fig.update_layout(title="Degradation Rate by Region (%)", xaxis_title="Region",
                      yaxis_title="Degraded %")
    return apply_theme(fig)


def stage_line(data: list[dict]) -> go.Figure:
    if not data:
        return go.Figure()
    stages = [d["Stage"] for d in data]
    mean_ct = [d.get("mean_ct", 0) for d in data]
    deg_pct = [d.get("degraded_pct", 0) for d in data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stages, y=mean_ct, mode="lines+markers",
                              name="Mean CT", marker_color=COLORS["accent_green"],
                              line=dict(width=3)))
    fig.add_trace(go.Scatter(x=stages, y=deg_pct, mode="lines+markers",
                              name="Degraded %", marker_color=COLORS["degraded"],
                              line=dict(width=3), yaxis="y2"))
    fig.update_layout(title="Quality by Pipeline Stage",
                      xaxis_title="Stage",
                      yaxis=dict(title="Mean CT Score"),
                      yaxis2=dict(title="Degraded %", overlaying="y", side="right"))
    return apply_theme(fig)


def scatter_wg_ct(df_records: list[dict]) -> go.Figure:
    if not df_records:
        return go.Figure()
    import pandas as pd
    df = pd.DataFrame(df_records)
    if "WG_Current" not in df or "CT_Current" not in df:
        return go.Figure()
    df["color"] = df["CT_Current"].apply(
        lambda x: COLORS["degraded"] if x < 60 else COLORS["at_risk"] if x < 80 else COLORS["high_quality"])
    fig = go.Figure(go.Scatter(
        x=df["WG_Current"], y=df["CT_Current"],
        mode="markers", marker=dict(color=df["color"].tolist(), size=4, opacity=0.6),
        text=df.get("Origin_Region", ""),
    ))
    fig.add_hline(y=60, line_dash="dash", line_color=COLORS["degraded"])
    fig.update_layout(title="WG vs CT — False-Pass Detection",
                      xaxis_title="Warm Germination (%)", yaxis_title="Cool Test (%)")
    return apply_theme(fig)


def seasonal_trend_chart(data: list[dict]) -> go.Figure:
    if not data:
        return go.Figure()
    seasons = [d["season"] for d in data]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=seasons, y=[d.get("degraded_pct", 0) for d in data],
                          name="Degraded %", marker_color=COLORS["degraded"], opacity=0.8))
    fig.add_trace(go.Scatter(x=seasons, y=[d.get("mean_ct", 0) for d in data],
                              mode="lines+markers", name="Mean CT",
                              marker_color=COLORS["accent_green"], yaxis="y2"))
    fig.update_layout(title="Seasonal Quality Trends",
                      xaxis_title="Season",
                      yaxis=dict(title="Degraded %"),
                      yaxis2=dict(title="Mean CT", overlaying="y", side="right"))
    return apply_theme(fig)


def survival_curve(km_points: list[dict], title: str = "Kaplan-Meier Survival Curve") -> go.Figure:
    if not km_points:
        return go.Figure()
    times    = [p["time"]     for p in km_points]
    survival = [p["survival"] for p in km_points]
    lower    = [p.get("lower_ci", p["survival"]) for p in km_points]
    upper    = [p.get("upper_ci", p["survival"]) for p in km_points]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times + times[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor=f"{COLORS['accent_blue']}20",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=survival, mode="lines",
        line=dict(color=COLORS["accent_blue"], width=3, shape="hv"),
        name="Survival probability",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["at_risk"],
                  annotation_text="50% survival", annotation_font_color=COLORS["at_risk"])
    fig.update_layout(title=title, xaxis_title="Growing Seasons",
                      yaxis_title="P(CT ≥ 60)", yaxis_range=[0, 1])
    return apply_theme(fig)


def multi_survival_curves(curves: dict[str, list[dict]], title: str) -> go.Figure:
    import plotly.colors as pc
    fig = go.Figure()
    palette = [COLORS["accent_green"], COLORS["accent_blue"], COLORS["accent_amber"],
               COLORS["accent_teal"], COLORS["accent_orange"], COLORS["degraded"]]
    for i, (group, points) in enumerate(curves.items()):
        if not points:
            continue
        color = palette[i % len(palette)]
        times    = [p["time"]     for p in points]
        survival = [p["survival"] for p in points]
        fig.add_trace(go.Scatter(x=times, y=survival, mode="lines",
                                  line=dict(color=color, width=2, shape="hv"),
                                  name=str(group)))
    fig.update_layout(title=title, xaxis_title="Growing Seasons",
                      yaxis_title="P(CT ≥ 60)", yaxis_range=[0, 1])
    return apply_theme(fig)


def forest_plot(hazard_ratios: list[dict]) -> go.Figure:
    if not hazard_ratios:
        return go.Figure()
    features = [d["feature"] for d in hazard_ratios]
    hr       = [d["hazard_ratio"] for d in hazard_ratios]
    lower    = [d.get("lower_ci", d["hazard_ratio"] * 0.8) for d in hazard_ratios]
    upper    = [d.get("upper_ci", d["hazard_ratio"] * 1.2) for d in hazard_ratios]
    colors   = [COLORS["degraded"] if h > 1 else COLORS["high_quality"] for h in hr]

    fig = go.Figure()
    for i, (feat, h, lo, hi, c) in enumerate(zip(features, hr, lower, upper, colors)):
        fig.add_trace(go.Scatter(x=[lo, hi], y=[feat, feat], mode="lines",
                                  line=dict(color=c, width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[h], y=[feat], mode="markers",
                                  marker=dict(size=10, color=c), showlegend=False))
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["text_secondary"])
    fig.update_layout(title="Cox PH Hazard Ratios (>1 = increases degradation risk)",
                      xaxis_title="Hazard Ratio", xaxis_type="log")
    return apply_theme(fig)


def rm_radar_chart(band_data: list[dict]) -> go.Figure:
    if not band_data:
        return go.Figure()
    categories = ["Mean CT", "Degraded %", "Lot Count"]
    fig = go.Figure()
    for d in band_data:
        vals = [d.get("mean_ct", 0) / 100, 1 - d.get("degraded_pct", 0) / 100,
                min(d.get("lot_count", 0) / 10000, 1)]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]],
                                       fill="toself", name=d.get("rm_band", "")))
    fig.update_layout(title="rm-Band Comparison (Radar)", polar=dict(bgcolor=COLORS["bg_card"]))
    return apply_theme(fig)
