"""Club Profile — Tactical Identity, League Percentiles & Key Squad Players."""
from __future__ import annotations

import html
import pandas as pd
import streamlit as st
from components import entity_selectbox, league_season_filters, radar_chart, render_club_hero_card
from data_loader import build_clean_club_table, build_club_tactical_summary, load_club_features, load_player_features, season_label
from theme import callout, footer_note, format_eur, img_tag, ordinal, page_header, render_percentile_bar

page_header(
    "Team Tactical DNA",
    "Club Playing-Style Profile",
    "Inspect any Top-5 League club's crest, manager, tactical identity, league percentiles, and core squad contributors.",
)

clubs = load_club_features()
players = load_player_features()
if clubs.empty:
    st.warning("Club dataset is unavailable. Check `data/processed/club_ui_metadata.parquet`.")
    st.stop()

filtered, _, _ = league_season_filters(clubs, "club")
team = entity_selectbox(filtered, "team", "Select Club", "club_name")
if team is None:
    st.stop()

row = filtered[filtered["team"].astype(str) == team].sort_values("season", ascending=False).iloc[0]
sel_season = int(row["season"])

# Club Hero Card with Crest, Manager & Stadium
render_club_hero_card(row)

# Plain-English Tactical Summary
callout(build_club_tactical_summary(row), "info")

k = st.columns(5)
k[0].metric("Ball Possession", f"{float(row.get('poss_share', 0))*100:.1f}%", f"{ordinal(int(row.get('pct_poss_share', 50)))} pct", delta_color="off", delta_arrow="off")
k[1].metric("Pressing (PPDA)", f"{float(row.get('ppda', 0)):.2f}", f"{ordinal(int(row.get('pct_ppda', 50)))} pct (lower = stronger)", delta_color="off", delta_arrow="off")
k[2].metric("Final-Third Tilt", f"{float(row.get('field_tilt', 0))*100:.1f}%", f"{ordinal(int(row.get('pct_field_tilt', 50)))} pct", delta_color="off", delta_arrow="off")
k[3].metric("Defensive Line Height", f"{float(row.get('def_line_height_m', 0)):.1f} m", f"{ordinal(int(row.get('pct_def_line_height_m', 50)))} pct", delta_color="off", delta_arrow="off")
k[4].metric("Long-Ball Share", f"{float(row.get('long_ball_share', 0))*100:.1f}%", f"{ordinal(int(row.get('pct_long_ball_share', 50)))} pct", delta_color="off", delta_arrow="off")

st.markdown(f"### Tactical DNA — Top-5 League Comparison ({season_label(sel_season)})")
c_left, c_right = st.columns([1.05, 1.15])

club_pillars = {
    "Ball Possession": float(row.get("pct_poss_share", 50)),
    "Pressing Intensity": float(row.get("pct_ppda", 50)),
    "Attacking Territory": float(row.get("pct_field_tilt", 50)),
    "Progressive Passing": float(row.get("pct_prog_pass_share", 50)),
    "High Defensive Line": float(row.get("pct_def_line_height_m", 50)),
    "Attacking Width": float(row.get("pct_wide_touch_share", 50)),
}

with c_left:
    st.plotly_chart(
        radar_chart([(f"{team} ({season_label(sel_season)})", club_pillars)], f"{team} — Tactical Shape", height=440),
        width="stretch",
    )
    st.caption("Percentiles compare this club against all 96 clubs across Europe's Top 5 Leagues in that season.")

with c_right:
    st.markdown(
        '<div class="ff-card">'
        + render_percentile_bar("Ball Possession Share", row.get("pct_poss_share", 50), f"{float(row.get('poss_share', 0))*100:.1f}%", "Share of match passes controlled by this club")
        + render_percentile_bar("Pressing Intensity (Inverse PPDA)", row.get("pct_ppda", 50), f"{float(row.get('ppda', 0)):.2f} PPDA", "Passes allowed per defensive action — lower means stronger pressing")
        + render_percentile_bar("Final-Third Territory (Field Tilt)", row.get("pct_field_tilt", 50), f"{float(row.get('field_tilt', 0))*100:.1f}%", "Share of final-third passes owned vs. opponents")
        + render_percentile_bar("Progressive Passing Share", row.get("pct_prog_pass_share", 50), f"{float(row.get('prog_pass_share', 0))*100:.1f}%", "Frequency of passes advancing the ball rapidly")
        + render_percentile_bar("High Defensive Line", row.get("pct_def_line_height_m", 50), f"{float(row.get('def_line_height_m', 0)):.1f} m", "Average pitch height of defensive actions")
        + render_percentile_bar("Direct Long-Ball Tendency", row.get("pct_long_ball_share", 50), f"{float(row.get('long_ball_share', 0))*100:.1f}%", "Reliance on long vertical passes over short build-up")
        + "</div>",
        unsafe_allow_html=True,
    )

# Squad Roster Visual Grid for this Club & Season
if not players.empty:
    squad = players[
        (players["team"].astype(str) == team)
        & (pd.to_numeric(players["season"], errors="coerce") == sel_season)
    ].sort_values("minutes_90", ascending=False)
    if not squad.empty:
        st.markdown(f"### Core Squad Contributors ({season_label(sel_season)})")
        top_sq = squad.head(8)
        for row_idx in range(0, len(top_sq), 4):
            chunk = top_sq.iloc[row_idx : row_idx + 4]
            cols = st.columns(4)
            for col, (_, pr) in zip(cols, chunk.iterrows()):
                p_img = img_tag(pr.get("image_url"), str(pr["player"]), css_class="ff-portrait", is_club=False)
                mins = int(pd.to_numeric(pr.get("minutes_90", 0), errors="coerce") or 0)
                with col:
                    st.markdown(
                        f'<div class="ff-card" style="text-align:center;padding:16px;">'
                        f'{p_img}'
                        f'<div style="font-size:1.0rem;font-weight:700;color:#FAFAFA;margin-top:8px;">{html.escape(str(pr["player"]))}</div>'
                        f'<div style="color:#A1A1AA;font-size:0.82rem;margin:3px 0 8px 0;">{html.escape(str(pr.get("sub_position", pr.get("position", ""))))} · {mins:,} mins</div>'
                        f'<span class="ff-pill ff-pill-cyan">{html.escape(str(pr.get("archetype_name", "")))}</span>'
                        f'<div style="color:#34D399;font-size:0.82rem;font-weight:700;margin-top:6px;">{format_eur(pr.get("market_value_eur"))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

with st.expander("View complete club tactical metrics table"):
    clean_tbl = build_clean_club_table(row)
    st.dataframe(clean_tbl, width="stretch", hide_index=True)

footer_note()