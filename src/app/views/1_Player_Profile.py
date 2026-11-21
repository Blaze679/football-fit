"""Player Profile — Visual Scouting Dossier with Percentiles, Plain-English Traits & Player Comparison."""
from __future__ import annotations

import html
import pandas as pd
import streamlit as st
from components import (
    entity_selectbox,
    league_season_filters,
    pillar_dict_from_row,
    radar_chart,
    render_player_hero_card,
)
from data_loader import PILLAR_LABELS, build_clean_player_table, load_player_features, season_label
from theme import fix_ordinals, footer_note, ordinal, page_header, render_percentile_bar

page_header(
    "Individual Scouting Dossier",
    "Player Playing-Style Profile",
    "Explore any player's portrait, market valuation, position-peer percentiles (1st–99th), and plain-English scouting traits.",
)

players = load_player_features()
if players.empty:
    st.warning("Player dataset is unavailable. Check `data/processed/player_ui_metadata.parquet`.")
    st.stop()

f_col1, f_col2 = st.columns([2, 1])
with f_col1:
    filtered, _, _ = league_season_filters(players, "player")
with f_col2:
    pos_opts = ["All positions"] + sorted(filtered["position"].dropna().astype(str).unique())
    sel_pos = st.selectbox("Position group", pos_opts, key="player_pos_filter")
    if sel_pos != "All positions":
        filtered = filtered[filtered["position"].astype(str) == sel_pos]

s_col1, s_col2 = st.columns([2, 1])
with s_col1:
    search_q = st.text_input("Quick search player by name", placeholder="e.g. Palmer, Zubimendi, Salah, Pedri...", key="player_search")
    if search_q.strip():
        filtered = filtered[filtered["player"].astype(str).str.contains(search_q.strip(), case=False, na=False)]
with s_col2:
    minimum = st.slider("Minimum minutes played", 0, 3000, 450, 90, key="player_min_slider")
    filtered = filtered[pd.to_numeric(filtered["minutes_90"], errors="coerce").fillna(0) >= minimum]

name = entity_selectbox(filtered, "player", "Select Player", "player_name")
if name is None:
    st.stop()

rows = filtered[filtered["player"].astype(str) == name].sort_values("season", ascending=False)
teams = sorted(rows["team"].dropna().astype(str).unique())
team = st.selectbox("Club (player appeared for multiple teams in selection)", teams, key="player_team") if len(teams) > 1 else teams[0]
row = rows[rows["team"].astype(str) == team].iloc[0]

# Hero card with headshot + club crest + bio pills
render_player_hero_card(row)

# Strengths & Watch Areas Banner
st.markdown(
    f'<div class="ff-card">'
    f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
    f'<div style="flex:1;min-width:260px;">'
    f'<div style="font-size:0.76rem;font-weight:700;color:#34D399;text-transform:uppercase;letter-spacing:0.05em;">Standout Strengths (vs. Same-Position Peers)</div>'
    f'<div style="color:#FAFAFA;font-size:0.98rem;font-weight:600;margin-top:4px;">{html.escape(fix_ordinals(row.get("top_strengths", "Balanced profile")))}</div>'
    f'</div>'
    f'<div style="flex:1;min-width:260px;">'
    f'<div style="font-size:0.76rem;font-weight:700;color:#FBBF24;text-transform:uppercase;letter-spacing:0.05em;">Areas Below Position Average</div>'
    f'<div style="color:#D4D4D8;font-size:0.94rem;margin-top:4px;">{html.escape(fix_ordinals(row.get("watch_areas", "None")))}</div>'
    f'</div></div></div>',
    unsafe_allow_html=True,
)

# Top-line headline KPIs
k = st.columns(5)
if str(row.get("position")) == "Goalkeeper":
    k[0].metric("Saves / 90", f"{float(row.get('gk_saves_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_gk_saves_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[1].metric("Crosses Claimed / 90", f"{float(row.get('gk_claims_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_gk_claims_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[2].metric("Sweeper Actions / 90", f"{float(row.get('gk_sweeper_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_gk_sweeper_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[3].metric("Pass Accuracy", f"{float(row.get('pass_acc', 0))*100:.1f}%", f"{ordinal(int(row.get('pct_pass_acc', 50)))} pct", delta_color="off", delta_arrow="off")
    k[4].metric("Minutes Played", f"{int(row.get('minutes_90', 0)):,}", f"{int(row.get('apps', 0))} apps", delta_color="off", delta_arrow="off")
else:
    k[0].metric("Goals + Assists / 90", f"{float(row.get('ga_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_ga_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[1].metric("Key Passes / 90", f"{float(row.get('key_passes_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_key_passes_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[2].metric("Prog. Passes / 90", f"{float(row.get('prog_passes_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_prog_passes_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[3].metric("1v1 Dribbles / 90", f"{float(row.get('takeons_att_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_takeons_att_p90', 50)))} pct", delta_color="off", delta_arrow="off")
    k[4].metric("Tackles + Int. / 90", f"{float(row.get('tackles_int_p90', 0)):.2f}", f"{ordinal(int(row.get('pct_tackles_int_p90', 50)))} pct", delta_color="off", delta_arrow="off")

st.markdown("### Six-Pillar Style Profile (Compared to Same-Position Peers)")

# Optional Head-to-Head Overlay Comparison
cmp_row = None
if st.toggle("Overlay a second player for head-to-head comparison", key="player_cmp_toggle"):
    peer_pool = players[
        (players["position"].astype(str) == str(row.get("position", "")))
        & (pd.to_numeric(players["minutes_90"], errors="coerce").fillna(0) >= 450)
        & (players["player"].astype(str) != name)
    ].sort_values(["season", "minutes_90"], ascending=[False, False])
    if not peer_pool.empty:
        peer_pool = peer_pool.assign(
            _label=peer_pool["player"].astype(str) + " (" + peer_pool["team"].astype(str) + " · " + peer_pool["season"].map(season_label) + ")"
        )
        sel_cmp = st.selectbox("Compare against", peer_pool["_label"].tolist(), key="player_cmp_sel")
        cmp_row = peer_pool[peer_pool["_label"] == sel_cmp].iloc[0]

left_col, right_col = st.columns([1.05, 1.15])

with left_col:
    series = [(f"{row['player']} ({season_label(row['season'])})", pillar_dict_from_row(row))]
    if cmp_row is not None:
        series.append((f"{cmp_row['player']} ({season_label(cmp_row['season'])})", pillar_dict_from_row(cmp_row)))
    chart_title = f"{row['player']} vs. {cmp_row['player']}" if cmp_row is not None else f"{row['player']} — Style Shape"
    st.plotly_chart(radar_chart(series, chart_title, height=440), width="stretch")
    st.caption("Each axis shows where the player ranks from 1st to 99th percentile against players in the same position and season.")

with right_col:
    bars_html = '<div class="ff-card">'
    for label, col, desc in PILLAR_LABELS:
        bars_html += render_percentile_bar(label, row.get(col, 50), subtitle=desc)
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)

st.markdown("### Detailed Per-90 Breakdown")
d1, d2 = st.columns(2)
with d1:
    st.markdown(
        '<div class="ff-card">'
        '<div style="font-weight:700;color:#38BDF8;margin-bottom:12px;">Attacking & Ball Progression</div>'
        + render_percentile_bar("Shots per 90", row.get("pct_shots_p90", 50), f"{float(row.get('shots_p90', 0)):.2f} / 90")
        + render_percentile_bar("Penalty-Box Touches per 90", row.get("pct_box_touches_p90", 50), f"{float(row.get('box_touches_p90', 0)):.2f} / 90")
        + render_percentile_bar("Final-Third Entry Passes per 90", row.get("pct_f3_passes_succ_p90", 50), f"{float(row.get('f3_passes_succ_p90', 0)):.2f} / 90")
        + render_percentile_bar("Throughballs per 90", row.get("pct_throughballs_p90", 50), f"{float(row.get('throughballs_p90', 0)):.2f} / 90")
        + render_percentile_bar("Crosses per 90", row.get("pct_crosses_p90", 50), f"{float(row.get('crosses_p90', 0)):.2f} / 90")
        + "</div>",
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        '<div class="ff-card">'
        '<div style="font-weight:700;color:#FBBF24;margin-bottom:12px;">Possession, Pressing & Aerial Duels</div>'
        + render_percentile_bar("Pass Completion Rate", row.get("pct_pass_acc", 50), f"{float(row.get('pass_acc', 0))*100:.1f}%")
        + render_percentile_bar("Ball Retention (Low Dispossessions)", row.get("pct_dispossessed_p90", 50), f"{float(row.get('dispossessed_p90', 0)):.2f} lost / 90")
        + render_percentile_bar("High-Press Defensive Actions", row.get("pct_high_press_actions_p90", 50), f"{float(row.get('high_press_actions_p90', 0)):.2f} / 90")
        + render_percentile_bar("Loose-Ball Recoveries per 90", row.get("pct_ball_recoveries_p90", 50), f"{float(row.get('ball_recoveries_p90', 0)):.2f} / 90")
        + render_percentile_bar("Aerial Duel Win Rate", row.get("pct_aerial_win_rate", 50), f"{float(row.get('aerial_win_rate', 0))*100:.1f}% won")
        + "</div>",
        unsafe_allow_html=True,
    )

with st.expander("View complete scouting metrics table"):
    clean_tbl = build_clean_player_table(row)
    st.dataframe(clean_tbl, width="stretch", hide_index=True)

footer_note()