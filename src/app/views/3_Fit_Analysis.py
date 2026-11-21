"""Fit Analysis — 1-vs-1 transfer evaluator and bidirectional scouting rankers."""
from __future__ import annotations

import html
import numpy as np
import pandas as pd
import streamlit as st
from components import entity_selectbox, league_season_filters, radar_chart
from data_loader import (
    compute_fit,
    load_club_features,
    load_player_features,
    rank_clubs_for_player,
    rank_players_for_club,
    season_label,
)
from theme import callout, fit_tier, footer_note, format_eur, img_tag, page_header, short_league

page_header(
    "Tactical Compatibility Engine · Top-5 European Leagues",
    "Player-to-Club Fit & Scouting",
    "Evaluate a transfer scenario, find the clubs that best match a player's style, or shortlist signings for a club. "
    "Scores measure tactical style similarity, not a guarantee of transfer outcome.",
)

players, clubs = load_player_features(), load_club_features()
if players.empty or clubs.empty:
    st.warning("Processed player or club data is missing. Check the files in `data/processed`.")
    st.stop()

outfield = players[
    players["position"].isin(["Attack", "Midfield", "Defender"])
    & (pd.to_numeric(players["minutes_90"], errors="coerce").fillna(0) >= 450)
].copy()
clubs_2425 = clubs[pd.to_numeric(clubs["season"], errors="coerce") == 2425].copy()

MODES = ["Evaluate a transfer", "Best clubs for a player", "Best signings for a club"]
mode = st.segmented_control("What do you want to do?", MODES, default=MODES[0], key="fit_mode") or MODES[0]
st.write("")


def score_card(rank_html: str, title_html: str, sub_html: str, pills_html: str, score: float) -> None:
    tier_lbl, tier_hex, pill_cls = fit_tier(score)
    st.markdown(
        f'<div class="ff-card" style="padding:14px 18px;border-left:4px solid {tier_hex};margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">'
        f'<div><strong style="font-size:1.05rem;color:#FAFAFA;">{rank_html}{title_html}</strong> '
        f'<span style="color:#A1A1AA;font-size:0.86rem;">{sub_html}</span><div style="margin-top:4px;">{pills_html}</div></div>'
        f'<div style="text-align:right;"><span style="font-size:1.35rem;font-weight:800;color:{tier_hex};">{score:.1f} / 100</span> '
        f'<span class="ff-pill {pill_cls}">{tier_lbl}</span></div></div></div>',
        unsafe_allow_html=True,
    )


# ====================================================================================
# MODE 1: Evaluate one transfer
# ====================================================================================
if mode == MODES[0]:
    presets = [
        ("Zubimendi → Arsenal", "Martín Zubimendi", "Arsenal"),
        ("Díaz → Bayern", "Luis Díaz", "Bayern"),
        ("Truffert → Bournemouth", "Adrien Truffert", "Bournemouth"),
        ("João Pedro → Chelsea", "João Pedro", "Chelsea"),
        ("Wirtz → Liverpool", "Florian Wirtz", "Liverpool"),
    ]
    st.caption("One-click examples from the Summer 2025 window (scored using 2024-25 baseline profiles):")
    for col, (label, p_name, c_name) in zip(st.columns(len(presets)), presets):
        if col.button(label, width="stretch", key=f"preset_{p_name}"):
            st.session_state.update(fit1_p_league="All leagues", fit1_p_season="All seasons", fit1_p_search="", fit1_c_league="All leagues")
            if p_name in outfield["player"].astype(str).values:
                st.session_state["fit1_player_sel"] = p_name
            if c_name in clubs_2425["team"].astype(str).values:
                st.session_state["fit1_club_sel"] = c_name
            st.rerun()

    left, right = st.columns(2)
    with left:
        st.markdown("##### 1 · Select Player")
        p_filt, _, _ = league_season_filters(outfield, "fit1_p")
        search = st.text_input("Search by name", key="fit1_p_search", placeholder="e.g. Zubimendi")
        if search.strip():
            p_filt = p_filt[p_filt["player"].astype(str).str.contains(search.strip(), case=False, na=False)]
        pname = entity_selectbox(p_filt, "player", "Outfield player (450+ min)", "fit1_player_sel")
    with right:
        st.markdown("##### 2 · Select Target Club")
        c_filt, _, _ = league_season_filters(clubs_2425, "fit1_c", include_season=False)
        target = entity_selectbox(c_filt, "team", "Destination club (2024-25 Top-5 profile)", "fit1_club_sel")
        st.caption("Target clubs use their 2024-25 tactical profile (newly promoted 2025-26 clubs lack a 2024-25 Top-5 baseline).")

    if not (pname and target):
        st.info("Choose a player and a target club.")
        st.stop()

    prows = p_filt[p_filt["player"].astype(str) == pname].sort_values("season", ascending=False)
    origins = list(dict.fromkeys(prows["team"].dropna().astype(str)))
    origin = st.selectbox("Player's club context", origins, key="fit1_origin") if len(origins) > 1 else origins[0]
    prow = prows[prows["team"].astype(str) == origin].iloc[0]

    res = compute_fit(prow, target)
    if "error" in res:
        st.error(res["error"])
        st.stop()
    if res.get("is_goalkeeper"):
        st.info(res["note"])
        st.stop()

    trow, score = res["target_row"], res["fit_score"]
    tier_lbl, tier_hex, pill_cls = fit_tier(score)
    all_clubs = rank_clubs_for_player(int(prow["player_id"]), int(prow["season"]), str(prow["team"]), top_n=None)
    rank_txt = ""
    if not all_clubs.empty and target in set(all_clubs["team"]):
        rk = int(all_clubs.loc[all_clubs["team"] == target, "rank"].iloc[0])
        rank_txt = f"Ranks #{rk} of {len(all_clubs)} Top-5 clubs for this player"

    peer_rows = players[pd.to_numeric(players["player_id"], errors="coerce") == int(res["closest_peer_id"])]
    peer_img_url = peer_rows.iloc[0]["image_url"] if not peer_rows.empty else None
    sub = html.escape(str(prow.get("sub_position", "")))
    st.markdown(
        f'<div class="ff-card" style="border-top:4px solid {tier_hex};">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;">'
        f'<div style="display:flex;align-items:center;gap:14px;min-width:240px;">'
        f'{img_tag(prow.get("image_url"), pname)}<div>'
        f'<div style="font-size:1.3rem;font-weight:800;color:#FAFAFA;">{html.escape(pname)}</div>'
        f'<div style="color:#A1A1AA;font-size:0.88rem;">{img_tag(prow.get("club_crest_url"), origin, css_class="ff-crest-sm", is_club=True)}{html.escape(origin)} · {sub}</div>'
        f'<div style="margin-top:4px;"><span class="ff-pill ff-pill-cyan">{html.escape(str(prow.get("archetype_name", "")))}</span></div></div></div>'
        f'<div style="text-align:center;padding:8px 18px;">'
        f'<div style="font-size:0.76rem;font-weight:700;color:#A1A1AA;text-transform:uppercase;letter-spacing:0.06em;">Overall Tactical Fit</div>'
        f'<div style="font-size:2.6rem;font-weight:800;color:{tier_hex};line-height:1.1;">{score:.1f} <span style="font-size:1rem;color:#71717A;">/ 100</span></div>'
        f'<div style="margin-top:6px;"><span class="ff-pill {pill_cls}">{tier_lbl}</span></div>'
        f'<div style="color:#A1A1AA;font-size:0.8rem;margin-top:4px;">{html.escape(rank_txt)}</div></div>'
        f'<div style="display:flex;align-items:center;gap:14px;min-width:250px;">'
        f'{img_tag(trow.get("crest_url"), target, css_class="ff-crest-lg", is_club=True)}<div>'
        f'<div style="font-size:1.25rem;font-weight:800;color:#FAFAFA;">{html.escape(target)}</div>'
        f'<div style="color:#A1A1AA;font-size:0.85rem;">Manager: {html.escape(str(trow.get("coach_name") or "N/A"))}</div>'
        f'<div style="margin-top:4px;"><span class="ff-pill ff-pill-slate">{html.escape(str(trow.get("tactical_identity", "")))}</span></div></div></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Role Compatibility (80%)", f"{res['role_fit_100']:.1f}", f"Closest peer: {res['closest_dest_peer']}", delta_color="off", delta_arrow="off")
    m2.metric("System & Pressing Fit (15%)", f"{res['system_fit_100']:.1f}", "Pressing, possession, tempo", delta_color="off", delta_arrow="off")
    m3.metric("Club Continuity (5%)", f"{res['env_fit_100']:.1f}", f"{origin} vs. {target}", delta_color="off", delta_arrow="off")

    v_col, peer_col = st.columns([1.45, 0.85])
    with v_col:
        st.markdown("#### Scouting Verdict")
        for b in res["bullets"]:
            st.markdown(f"- {b}")
        st.caption(f"Comparison basis: {res['peer_method']} ({season_label(prow.get('season', 2425))} player profile).")
    with peer_col:
        sim = int(round(max(0.0, res["closest_peer_sim"]) * 100))
        st.markdown(
            f'<div class="ff-card" style="text-align:center;">'
            f'<div style="font-size:0.76rem;font-weight:700;color:#A1A1AA;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Closest Stylistic Peer at {html.escape(target)}</div>'
            f'{img_tag(peer_img_url, res["closest_dest_peer"])}'
            f'<div style="font-size:1.1rem;font-weight:800;color:#FAFAFA;margin-top:8px;">{html.escape(res["closest_dest_peer"])}</div>'
            f'<div style="color:#34D399;font-weight:700;font-size:0.92rem;margin-top:4px;">{sim}% style similarity</div></div>',
            unsafe_allow_html=True,
        )

    groups = {
        "Ball Progression": ["prog_passes_p90", "fwd_pass_share", "avg_touch_x"],
        "Chance Creation": ["key_passes_p90", "throughballs_p90", "f3_passes_succ_p90"],
        "Ball Security": ["pass_acc", "takeon_succ_rate", "dispossessed_p90"],
        "Dribbling & Width": ["crosses_p90", "open_cross_share", "avg_touch_y_centrality"],
        "Goal & Box Threat": ["shots_p90", "box_touches_p90", "att_third_touch_share"],
        "Pressing & Defending": ["high_press_actions_p90", "tackles_int_p90", "ball_recoveries_p90"],
    }
    idx = {c: i for i, c in enumerate(res["player_cols"])}
    p_rad = {g: float(np.clip(50 + 14 * np.mean([res["player_z"][idx[c]] for c in cs if c in idx] or [0]), 5, 98)) for g, cs in groups.items()}
    c_rad = {g: float(np.clip(50 + 14 * np.mean([res["peer_z"][idx[c]] for c in cs if c in idx] or [0]), 5, 98)) for g, cs in groups.items()}
    st.plotly_chart(
        radar_chart(
            [(f"{pname} (Style)", p_rad), (f"{target} {prow.get('position')} Role Demand", c_rad)],
            f"Style Overlay: {pname} vs. {target}'s {prow.get('position')} Role",
            height=450,
        ),
        width="stretch",
        theme=None,
    )

    brief_df = pd.DataFrame([{
        "Player": pname,
        "Current Club": origin,
        "Role": prow.get("sub_position", ""),
        "Target Club": target,
        "Overall Fit Score": score,
        "Verdict": tier_lbl,
        "Role Compatibility (80%)": res["role_fit_100"],
        "System Fit (15%)": res["system_fit_100"],
        "Club Continuity (5%)": res["env_fit_100"],
        "Closest Peer at Target": res["closest_dest_peer"],
        "Peer Style Similarity (%)": sim,
    }])
    st.download_button(
        "Download Scouting Brief (CSV)",
        brief_df.to_csv(index=False),
        file_name=f"scouting_brief_{pname.replace(' ', '_')}_to_{target.replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ====================================================================================
# MODE 2: Best clubs for a player
# ====================================================================================
elif mode == MODES[1]:
    st.markdown("##### Which Top-5 clubs suit this player's style?")
    p_filt2, _, _ = league_season_filters(outfield, "fit2_p")
    pname2 = entity_selectbox(p_filt2, "player", "Player", "fit2_player_sel")
    if pname2:
        prow2 = p_filt2[p_filt2["player"].astype(str) == pname2].sort_values("season", ascending=False).iloc[0]
        ranked = rank_clubs_for_player(int(prow2["player_id"]), int(prow2["season"]), str(prow2["team"]), top_n=10)
        if ranked.empty:
            st.info("Could not rank clubs for this player.")
        else:
            st.markdown(f"**Top 10 clubs for {pname2}** ({prow2.get('sub_position')}, {season_label(prow2['season'])} at {prow2.get('team')})")
            for _, r in ranked.iterrows():
                score_card(
                    f"#{int(r['rank'])} ",
                    f"{img_tag(r.get('crest_url'), str(r['team']), css_class='ff-crest-sm', is_club=True)}{html.escape(str(r['team']))}",
                    f"({html.escape(short_league(r['league']))} · Manager: {html.escape(str(r['coach_name'] or 'N/A'))})",
                    f'<span class="ff-pill ff-pill-cyan">{html.escape(str(r["tactical_identity"]))}</span>'
                    f'<span class="ff-pill ff-pill-slate">Closest peer: {html.escape(str(r["closest_dest_peer"]))}</span>',
                    float(r["fit_score"]),
                )
            export_clubs = ranked[["rank", "team", "league", "coach_name", "tactical_identity", "fit_score", "role_fit_100", "system_fit_100", "env_fit_100", "closest_dest_peer"]].rename(columns={
                "rank": "Rank", "team": "Club", "league": "League", "coach_name": "Manager",
                "tactical_identity": "Tactical Identity", "fit_score": "Overall Fit Score",
                "role_fit_100": "Role Fit", "system_fit_100": "System Fit",
                "env_fit_100": "Club Continuity", "closest_dest_peer": "Closest Peer",
            })
            st.download_button(
                "Download Club Ranking (CSV)",
                export_clubs.to_csv(index=False),
                file_name=f"best_clubs_{pname2.replace(' ', '_')}.csv",
                mime="text/csv",
            )

# ====================================================================================
# MODE 3: Best signings for a club
# ====================================================================================
else:
    st.markdown("##### Shortlist the best-fitting targets for a club")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        club3 = entity_selectbox(clubs_2425, "team", "Recruiting club", "fit3_club_sel")
    with c2:
        pos3 = st.selectbox("Position", ["All", "Attack", "Midfield", "Defender"], key="fit3_pos_sel")
    with c3:
        season_lbl = st.selectbox("Scouting pool", ["2024-25", "2025-26"], key="fit3_season_sel")
    with c4:
        max_age = st.slider("Max age", 19, 36, 28, 1, key="fit3_age_sel")
    if club3:
        season3 = 2425 if season_lbl == "2024-25" else 2526
        with st.spinner("Scoring candidates..."):
            short = rank_players_for_club(club3, position=pos3, season=season3, min_minutes=900, max_age=float(max_age), top_n=10)
        if short.empty:
            st.info("No candidates match these filters. Try a wider age range or position.")
        else:
            st.markdown(f"**Top 10 targets for {club3}** ({season_lbl} pool, 900+ minutes)")
            for i, r in short.iterrows():
                age = f"Age {float(r['age_2025']):.1f}" if pd.notna(r.get("age_2025")) else "Age N/A"
                score_card(
                    f"#{i + 1} ",
                    f"{img_tag(r.get('image_url'), str(r['player']), css_class='ff-crest-sm')}{html.escape(str(r['player']))}",
                    f"({html.escape(str(r['team']))} · {html.escape(str(r['sub_position']))} · {age})",
                    f'<span class="ff-pill ff-pill-cyan">{html.escape(str(r["archetype_name"]))}</span>'
                    f'<span class="ff-pill ff-pill-green">Value: {format_eur(r.get("market_value_eur"))}</span>'
                    f'<span class="ff-pill ff-pill-slate">Mirrors: {html.escape(str(r["closest_dest_peer"]))}</span>',
                    float(r["fit_score"]),
                )
            export_short = short[["player", "team", "league", "position", "sub_position", "age_2025", "market_value_display", "archetype_name", "fit_score", "role_fit_100", "system_fit_100", "closest_dest_peer"]].rename(columns={
                "player": "Player", "team": "Current Club", "league": "League", "position": "Position",
                "sub_position": "Role", "age_2025": "Age", "market_value_display": "Market Value",
                "archetype_name": "Tactical Archetype", "fit_score": "Overall Fit Score",
                "role_fit_100": "Role Fit", "system_fit_100": "System Fit", "closest_dest_peer": "Closest Peer",
            })
            st.download_button(
                "Download Target Shortlist (CSV)",
                export_short.to_csv(index=False),
                file_name=f"shortlist_{club3.replace(' ', '_')}.csv",
                mime="text/csv",
            )

callout("Fit tiers (Elite 80+, Strong 70+, Moderate 58+) are practical bands on a 0–100 style similarity scale, not calibrated probabilities.", "amber")
footer_note()