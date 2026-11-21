"""Shared UI widgets, Plotly dark editorial charts, and visual scouting cards."""
from __future__ import annotations

import html
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_loader import PILLAR_LABELS, season_label
from theme import format_eur, img_tag, short_league


def league_season_filters(
    frame: pd.DataFrame,
    key_prefix: str,
    include_league: bool = True,
    include_season: bool = True,
):
    filtered = frame.copy()
    left, right = st.columns(2)
    league = season = None
    if include_league and "league" in filtered:
        raw_vals = filtered["league"].dropna().astype(str).unique()
        clean_map = {short_league(v): v for v in raw_vals}
        options = sorted(clean_map.keys())
        with left:
            league = st.selectbox("League", ["All leagues"] + options, key=f"{key_prefix}_league")
        if league != "All leagues":
            filtered = filtered[filtered["league"].map(short_league) == league]
    if include_season and "season" in filtered:
        values = sorted([int(v) for v in filtered["season"].dropna().unique()], reverse=True)
        labels = {season_label(v): v for v in values}
        with right:
            selected = st.selectbox("Season", ["All seasons"] + list(labels), key=f"{key_prefix}_season")
        if selected != "All seasons":
            season = labels[selected]
            filtered = filtered[pd.to_numeric(filtered["season"], errors="coerce") == season]
    return filtered, league, season


def entity_selectbox(frame: pd.DataFrame, column: str, label: str, key: str, default_value: str | None = None):
    if frame.empty or column not in frame:
        st.info(f"No {label.lower()} data available for the selected filters.")
        return None
    options = sorted(frame[column].dropna().astype(str).unique())
    if not options:
        st.info(f"No {label.lower()} values available.")
        return None
    idx = options.index(default_value) if (default_value and default_value in options) else 0
    return st.selectbox(label, options, index=idx, key=key)


def pillar_dict_from_row(row: pd.Series) -> dict[str, float]:
    """Extract the 6 non-technical 0-99th percentile scouting pillars from a player row."""
    out = {}
    for label, col, _ in PILLAR_LABELS:
        val = pd.to_numeric(row.get(col, 50), errors="coerce")
        out[label] = 50.0 if pd.isna(val) else float(val)
    return out


def radar_chart(series: list[tuple[str, dict[str, float]]], title: str, height: int = 430) -> go.Figure:
    """Build a crisp obsidian-dark Plotly radar chart."""
    palette = [
        ("#38BDF8", "rgba(56, 189, 248, 0.18)"),
        ("#F59E0B", "rgba(245, 158, 11, 0.18)"),
        ("#10B981", "rgba(16, 185, 129, 0.18)"),
    ]
    fig = go.Figure()
    for idx, (name, values) in enumerate(series):
        labels = list(values.keys())
        vals = [float(v) for v in values.values()]
        if not labels:
            continue
        line_c, fill_c = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=fill_c,
                line=dict(color=line_c, width=2.5),
                marker=dict(size=6, color=line_c),
                name=name,
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#FAFAFA")),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E4E4E7", family="Inter, sans-serif"),
        polar=dict(
            bgcolor="#121318",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#A1A1AA"),
                gridcolor="#27272A",
                linecolor="#3F3F46",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#FAFAFA"),
                gridcolor="#27272A",
                linecolor="#3F3F46",
            ),
        ),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(color="#FAFAFA")),
        margin=dict(l=45, r=45, t=75, b=35),
    )
    return fig


def render_player_hero_card(row: pd.Series) -> None:
    """Render an executive player dossier card with portrait, crest, and bio pills."""
    name = html.escape(str(row.get("player", "Unknown Player")))
    team = html.escape(str(row.get("team", "Unknown Club")))
    league = html.escape(short_league(row.get("league", "")))
    pos = html.escape(str(row.get("position", "")))
    subp = html.escape(str(row.get("sub_position", "")))
    archetype = html.escape(str(row.get("archetype_name", subp)))
    nat = html.escape(str(row.get("nationality") or "N/A"))
    foot = html.escape(str(row.get("foot") or "N/A").title())
    age = row.get("age_2025")
    age_str = f"{float(age):.1f} yrs" if pd.notna(age) else "N/A"
    ht = row.get("height_cm")
    ht_str = f"{int(ht)} cm" if pd.notna(ht) else "N/A"
    mv_str = format_eur(row.get("market_value_eur"))
    mins = int(pd.to_numeric(row.get("minutes_90", row.get("minutes", 0)), errors="coerce") or 0)
    apps = int(pd.to_numeric(row.get("apps", 0), errors="coerce") or 0)
    starts = int(pd.to_numeric(row.get("starts", 0), errors="coerce") or 0)
    s_lbl = season_label(row.get("season", ""))

    portrait = img_tag(row.get("image_url"), str(row.get("player", "")), css_class="ff-portrait", is_club=False)
    crest = img_tag(row.get("club_crest_url"), str(row.get("team", "")), css_class="ff-crest-sm", is_club=True)

    st.markdown(
        f'<div class="ff-card ff-card-emerald">'
        f'<div class="ff-entity-wrap">'
        f'{portrait}'
        f'<div style="flex:1;min-width:260px;">'
        f'<div style="font-size:1.6rem;font-weight:800;color:#FAFAFA;line-height:1.15;">{name}</div>'
        f'<div style="color:#A1A1AA;font-size:0.96rem;margin:5px 0 10px 0;">'
        f'{crest}<strong style="color:#E4E4E7;">{team}</strong> · {league} · Season {html.escape(s_lbl)}'
        f'</div>'
        f'<div>'
        f'<span class="ff-pill ff-pill-cyan">{archetype}</span>'
        f'<span class="ff-pill ff-pill-slate">{pos} ({subp})</span>'
        f'<span class="ff-pill ff-pill-green">Market Value: {mv_str}</span>'
        f'<span class="ff-pill ff-pill-slate">Age: {age_str}</span>'
        f'<span class="ff-pill ff-pill-slate">Foot: {foot}</span>'
        f'<span class="ff-pill ff-pill-slate">Height: {ht_str}</span>'
        f'<span class="ff-pill ff-pill-slate">Nation: {nat}</span>'
        f'<span class="ff-pill ff-pill-slate">{mins:,} mins ({apps} apps, {starts} starts)</span>'
        f'</div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


def render_club_hero_card(row: pd.Series) -> None:
    """Render an executive club tactical dossier card with crest, manager, and stadium."""
    team = html.escape(str(row.get("team", "Unknown Club")))
    tm_name = html.escape(str(row.get("tm_club_name", team)))
    league = html.escape(short_league(row.get("league", "")))
    identity = html.escape(str(row.get("tactical_identity", "Balanced Hybrid System")))
    coach = html.escape(str(row.get("coach_name") or "N/A"))
    stadium = html.escape(str(row.get("stadium_name") or "N/A"))
    seats = row.get("stadium_seats")
    seats_str = f" ({int(seats):,} cap.)" if pd.notna(seats) else ""
    avg_age = row.get("average_age")
    age_str = f"{float(avg_age):.1f} yrs" if pd.notna(avg_age) else "N/A"
    sq_size = row.get("squad_size")
    sq_str = f"{int(sq_size)} players" if pd.notna(sq_size) else "N/A"
    net_tr = html.escape(str(row.get("net_transfer_record") or "N/A"))
    s_lbl = season_label(row.get("season", ""))

    crest = img_tag(row.get("crest_url"), str(row.get("team", "")), css_class="ff-crest-lg", is_club=True)

    st.markdown(
        f'<div class="ff-card ff-card-emerald">'
        f'<div class="ff-entity-wrap">'
        f'{crest}'
        f'<div style="flex:1;min-width:260px;">'
        f'<div style="font-size:1.6rem;font-weight:800;color:#FAFAFA;line-height:1.15;">{team}</div>'
        f'<div style="color:#A1A1AA;font-size:0.92rem;margin:4px 0 10px 0;">'
        f'{tm_name} · {league} · Season {html.escape(s_lbl)}'
        f'</div>'
        f'<div>'
        f'<span class="ff-pill ff-pill-cyan">{identity}</span>'
        f'<span class="ff-pill ff-pill-slate">Manager: {coach}</span>'
        f'<span class="ff-pill ff-pill-slate">Stadium: {stadium}{seats_str}</span>'
        f'<span class="ff-pill ff-pill-slate">Squad: {sq_str} (Avg {age_str})</span>'
        f'<span class="ff-pill ff-pill-amber">Net Transfer: {net_tr}</span>'
        f'</div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )