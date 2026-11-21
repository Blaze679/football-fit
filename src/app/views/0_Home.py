"""Overview / landing page."""
from __future__ import annotations

import html
import streamlit as st
from data_loader import (
    available_seasons,
    load_backtest,
    load_club_features,
    load_player_features,
    load_role_vectors,
    season_label,
)
from theme import callout, fit_tier, footer_note, img_tag, page_header

clubs, players, roles, backtest = load_club_features(), load_player_features(), load_role_vectors(), load_backtest()

page_header(
    "European Top-5 Leagues · Tactical Recruitment Analytics",
    "Does this player fit this club?",
    "Football Fit compares how a footballer plays with how a club plays, using match events from the Premier League, "
    "La Liga, Serie A, Bundesliga, and Ligue 1. The fit model is built strictly on <strong>2024-25</strong> data and "
    "validated out-of-sample against the <strong>Summer 2025</strong> transfer window.",
)

n_clubs = clubs["team"].nunique() if not clubs.empty else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clubs Profiled", f"{n_clubs}", f"{len(clubs)} club-seasons", delta_color="off", delta_arrow="off")
k2.metric("Players Profiled", f"{players['player_id'].nunique():,}" if not players.empty else "0", f"{len(players):,} player-seasons", delta_color="off", delta_arrow="off")
k3.metric("2024-25 Style Pool", f"{len(roles):,}", "Outfield players, 450+ min", delta_color="off", delta_arrow="off")
k4.metric("Transfers Backtested", f"{len(backtest):,}", "Summer 2025, outfield", delta_color="off", delta_arrow="off")

st.markdown("### Explore the Studio")
cards = [
    ("Player Profile", "views/1_Player_Profile.py",
     "Headshot, market value, peer percentiles (1st–99th), plain-English strengths, and a six-pillar style radar."),
    ("Club Profile", "views/2_Club_Profile.py",
     "Tactical identity, pressing and territory percentiles, plain-English style summary, and core squad contributors."),
    ("Fit Analysis", "views/3_Fit_Analysis.py",
     "Test one transfer, find the best-fitting clubs for a player, or shortlist signings for any club."),
    ("Style Map & Backtest", "views/4_Backtest_and_Style_Map.py",
     "Interactive 2024-25 tactical archetype map and out-of-sample Summer 2025 transfer backtest."),
]
for col, (title, target, text) in zip(st.columns(4), cards):
    with col:
        st.markdown(
            f'<div class="ff-card" style="height:164px;">'
            f'<div class="ff-card-title">{title}</div>'
            f'<div class="ff-card-text">{text}</div></div>',
            unsafe_allow_html=True,
        )
        st.page_link(target, label=f"Open {title}", icon=":material/arrow_forward:")

if not backtest.empty:
    st.markdown("### Summer 2025 Transfer Case Studies")
    wanted = ["Luis Díaz", "Martín Zubimendi", "Adrien Truffert", "João Pedro"]
    feat = backtest[backtest["player_name"].isin(wanted)]
    if feat.empty:
        feat = backtest.sort_values("fit_score", ascending=False).head(4)
    for col, (_, r) in zip(st.columns(len(feat)), feat.iterrows()):
        tier_lbl, tier_hex, pill_cls = fit_tier(float(r["fit_score"]))
        outcome = str(r.get("label_base_sec9", "N/A"))
        out_pill = {"Success": "ff-pill-green", "Failure": "ff-pill-rose"}.get(outcome, "ff-pill-slate")
        with col:
            st.markdown(
                f'<div class="ff-card" style="border-top:3px solid {tier_hex};text-align:center;">'
                f'{img_tag(r.get("image_url"), str(r["player_name"]), css_class="ff-portrait")}'
                f'<div style="font-size:1.08rem;font-weight:800;color:#FAFAFA;margin-top:10px;">{html.escape(str(r["player_name"]))}</div>'
                f'<div style="color:#A1A1AA;font-size:0.84rem;margin:4px 0 10px 0;">{html.escape(str(r["tm_from_club"]))} → '
                f'{img_tag(r.get("dest_crest_url"), str(r["tm_to_club"]), css_class="ff-crest-sm", is_club=True)}'
                f'<strong style="color:#FAFAFA;">{html.escape(str(r["tm_to_club"]))}</strong></div>'
                f'<div style="font-size:1.68rem;font-weight:800;color:{tier_hex};">{float(r["fit_score"]):.1f} '
                f'<span style="font-size:0.88rem;color:#71717A;">/ 100</span></div>'
                f'<div style="margin-top:6px;"><span class="ff-pill {pill_cls}">{tier_lbl}</span>'
                f'<span class="ff-pill {out_pill}">2025-26: {html.escape(outcome)}</span></div>'
                f'<div style="color:#A1A1AA;font-size:0.8rem;margin-top:8px;">Closest peer: <strong style="color:#FAFAFA;">{html.escape(str(r["closest_dest_peer"]))}</strong></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()
left, right = st.columns([3, 2])
with left:
    st.markdown("### How the Fit Score Works")
    st.markdown(
        """
1. **Fair possession adjustment.** A midfielder at a 42%-possession club touches the ball less than one at a possession-heavy club, so volumes are rescaled to a 50% baseline. The score reflects *how* a player plays, not their old team's ball share.
2. **Role-peer match (80%).** The player's 30-feature style signature is compared against the target club's players in the same role.
3. **System fit (15%).** Checks whether the player's pressing intensity, ball security, and forward passing match the club's pressing height, build-up, and tempo.
4. **Club continuity (5%).** Measures how close the player's current and target clubs are in overall tactical identity.
        """
    )
with right:
    seasons = available_seasons(clubs, players)
    st.markdown("### Data Integrity & Scope")
    callout(
        f"<strong>Seasons:</strong> {' · '.join(season_label(s) for s in seasons)}<br>"
        "<strong>No outcome leakage:</strong> scalers, style space, and club vectors are fitted on 2024-25 only, and goals and assists are excluded from fit inputs.<br>"
        "<strong>Exploratory validation:</strong> multiple score variants and label thresholds were evaluated on one transfer cohort, so backtest numbers are indicative rather than definitive. See <em>Method &amp; Limitations</em>.",
        "info",
    )
footer_note()