"""2024-25 style map and the summer-2025 transfer backtest (statistics computed live, with intervals)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_backtest, load_style_space
from stats_utils import auc_score, bootstrap_ci, perm_pvalue, spearman
from theme import callout, footer_note, page_header, short_league

AXIS = dict(gridcolor="#27272A", zerolinecolor="#3F3F46", linecolor="#3F3F46")


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#121318",
        font=dict(color="#E4E4E7", family="Inter, sans-serif"),
        title_font=dict(size=15, color="#FAFAFA"),
        legend=dict(bgcolor="rgba(18, 19, 24, 0.92)", bordercolor="#27272A", borderwidth=1, font=dict(color="#FAFAFA")),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


page_header(
    "Unsupervised Style Map · Out-of-Sample Validation",
    "Style Map & Summer 2025 Backtest",
    "How Top-5 outfield players cluster into eight tactical archetypes, and how pre-transfer fit scores compared with "
    "real on-pitch outcomes for Summer 2025 signings in 2025-26.",
)
space, backtest = load_style_space(), load_backtest()

# ------------------------------------------------------------------------------------
# 1. STYLE MAP
# ------------------------------------------------------------------------------------
st.markdown(f"### 1 · 2024-25 Player Style Map ({len(space):,} outfield players)")
if space.empty or not {"umap_1", "umap_2"}.issubset(space.columns):
    st.warning("The UMAP style-space file is missing or lacks `umap_1` / `umap_2`.")
else:
    m1, m2, m3 = st.columns([1.2, 1, 1.3])
    with m1:
        color_by = st.radio(
            "Colour players by",
            ["archetype_name", "position", "league_clean"],
            horizontal=True,
            key="map_color",
            format_func=lambda x: {"archetype_name": "Tactical Archetype", "position": "Position", "league_clean": "League"}[x],
        )
    with m2:
        clean_lgs = sorted(space["league"].dropna().map(short_league).unique())
        sel_lg = st.selectbox("League", ["All leagues"] + clean_lgs, key="map_lg")
    with m3:
        spotlight = st.selectbox("Spotlight a player", ["None"] + sorted(space["player"].dropna().astype(str).unique()), key="map_spot")

    plot_df = space.copy()
    if "league_clean" not in plot_df.columns:
        plot_df["league_clean"] = plot_df["league"].map(short_league)
    if sel_lg != "All leagues":
        plot_df = plot_df[plot_df["league_clean"] == sel_lg]

    plot_df = plot_df.assign(
        **{
            color_by: plot_df[color_by].fillna("Unknown").astype(str),
            "Club": plot_df["team"].fillna("Unknown").astype(str),
            "League": plot_df["league_clean"].fillna("Unknown").astype(str),
            "Role": plot_df["sub_position"].fillna("Unknown").astype(str),
            "Tactical Archetype": plot_df["archetype_name"].fillna("Unknown").astype(str),
            "Minutes Played": pd.to_numeric(plot_df["minutes_90"], errors="coerce").fillna(0).astype(int),
        }
    )

    fig = px.scatter(
        plot_df,
        x="umap_1",
        y="umap_2",
        color=color_by,
        hover_name="player",
        hover_data={
            "umap_1": False,
            "umap_2": False,
            color_by: False,
            "Club": True,
            "League": True,
            "Role": True,
            "Tactical Archetype": True,
            "Minutes Played": ":,",
        },
        opacity=0.88,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title=f"{len(plot_df):,} Players in Tactical Style Space",
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#121318")))
    if spotlight != "None":
        sp = space[space["player"].astype(str) == spotlight]
        if not sp.empty:
            s = sp.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[s["umap_1"]],
                    y=[s["umap_2"]],
                    mode="markers+text",
                    text=[f"★ {s['player']} ({s['team']})"],
                    textposition="top center",
                    textfont=dict(color="#FAFAFA", size=13, family="Inter, sans-serif"),
                    marker=dict(size=18, color="#FBBF24", symbol="star", line=dict(width=1.8, color="#09090B")),
                    name="Spotlight",
                )
            )
    fig.update_xaxes(title="Tactical Style Dimension 1", showticklabels=False)
    fig.update_yaxes(title="Tactical Style Dimension 2", showticklabels=False)
    st.plotly_chart(style_fig(fig, 600), width="stretch", theme=None)
    st.caption(
        "Players positioned close together perform similar on-ball, spatial, and defensive actions per 90 minutes. "
        "The two axes represent a 2-D projection (UMAP), so proximity is indicative rather than an exact metric distance."
    )

st.divider()

# ------------------------------------------------------------------------------------
# 2. BACKTEST
# ------------------------------------------------------------------------------------
st.markdown("### 2 · Summer 2025 Transfer Backtest")
if backtest.empty:
    st.warning("The backtest file is missing. Check `data/processed/backtest_results_v3.parquet`.")
    footer_note()
    st.stop()

callout(
    "<strong>Read this first.</strong> This is an <em>exploratory</em> check on a single transfer window. The fit score was built strictly using "
    "2024-25 data, but several score variants and three outcome thresholds were compared on these same transfers, and the 80/15/5 weights are a "
    "domain modelling choice. Confidence intervals below show sampling uncertainty.",
    "amber",
)


@st.cache_data(show_spinner="Computing backtest statistics...")
def compute_stats(df: pd.DataFrame) -> dict:
    out: dict = {"n": len(df)}
    am = df[df["position"].isin(["Attack", "Midfield"])].dropna(subset=["fit_score", "Y_score"])
    if len(am) > 10:
        r, lo, hi = bootstrap_ci(spearman, (am["fit_score"].to_numpy(), am["Y_score"].to_numpy()), 2000)
        out["y"] = dict(n=len(am), r=r, lo=lo, hi=hi, p=perm_pvalue(am["fit_score"].to_numpy(), am["Y_score"].to_numpy()))
    ms = df.dropna(subset=["fit_score", "MS"])
    if len(ms) > 10:
        r, lo, hi = bootstrap_ci(spearman, (ms["fit_score"].to_numpy(), ms["MS"].to_numpy()), 2000)
        out["ms"] = dict(n=len(ms), r=r, lo=lo, hi=hi, p=perm_pvalue(ms["fit_score"].to_numpy(), ms["MS"].to_numpy()))
    rows = []
    for lbl, name in [("label_lenient_sec9", "Lenient"), ("label_base_sec9", "Base"), ("label_strict_sec9", "Strict")]:
        if lbl not in df.columns:
            continue
        sub = df[df[lbl].isin(["Success", "Failure"])].dropna(subset=["fit_score"])
        y = (sub[lbl] == "Success").astype(int).to_numpy()
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        a, lo, hi = bootstrap_ci(auc_score, (y, sub["fit_score"].to_numpy()), 2000)
        row = {
            "Threshold": name,
            "Sample (n)": len(sub),
            "Successes": int(y.sum()),
            "Failures": int((1 - y).sum()),
            "Fit Score AUC": a,
            "95% Confidence Interval": f"{lo:.2f} – {hi:.2f}",
        }
        for col, sign, key in [
            ("expected_ga_p90", 1, "Expected G+A Baseline"),
            ("age_years", -1, "Younger Age Baseline"),
            ("club_strength_gap", -1, "Smaller Club Jump Baseline"),
        ]:
            if col in sub.columns:
                row[key] = auc_score(y, sign * sub[col].fillna(sub[col].median()).to_numpy())
        rows.append(row)
    out["auc"] = pd.DataFrame(rows)
    return out


stats = compute_stats(backtest)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transfers Evaluated", f"{stats['n']}", "Outfield, complete 2024-25 profiles", delta_color="off", delta_arrow="off")
if "y" in stats:
    y = stats["y"]
    k2.metric("Fit vs. Impact (Attack + Mid)", f"ρ = {y['r']:+.2f}", f"95% CI {y['lo']:+.2f} to {y['hi']:+.2f} · n={y['n']}", delta_color="off", delta_arrow="off")
if "ms" in stats:
    m = stats["ms"]
    k3.metric("Fit vs. Minutes Share", f"ρ = {m['r']:+.2f}", f"95% CI {m['lo']:+.2f} to {m['hi']:+.2f} · n={m['n']}", delta_color="off", delta_arrow="off")
if not stats["auc"].empty:
    b = stats["auc"][stats["auc"]["Threshold"] == "Base"]
    if not b.empty:
        k4.metric("Base-Threshold AUC", f"{b['Fit Score AUC'].iloc[0]:.2f}", f"95% CI {b['95% Confidence Interval'].iloc[0]} · n={int(b['Sample (n)'].iloc[0])}", delta_color="off", delta_arrow="off")

if "y" in stats:
    y = stats["y"]
    verdict = (
        "The confidence interval includes zero, so this correlation could arise by chance."
        if y["lo"] <= 0 <= y["hi"]
        else "The confidence interval excludes zero, indicating a positive relationship."
    )
    callout(
        f"<strong>Primary measure.</strong> Spearman rank correlation between the pre-transfer fit score and 2025-26 impact "
        f"(goals + assists per 90 relative to expectation) for Attack and Midfield signings: ρ = {y['r']:+.2f}, permutation p = {y['p']:.3f}. {verdict}"
    )

st.markdown("#### Discrimination Between Successful and Failed Signings (ROC-AUC)")
if stats["auc"].empty:
    st.info("Not enough successes and failures to compute AUC.")
else:
    tbl = stats["auc"].copy()
    num_cols = [c for c in tbl.columns if "AUC" in c or "Baseline" in c]
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in num_cols}), width="stretch", hide_index=True)
    st.caption("AUC 0.50 equals random guessing; 1.00 is perfect separation. Defenders are evaluated on minutes share only.")
    callout(
        "<strong>About the baselines.</strong> A baseline below 0.50 points in the opposite direction for this label. Because impact is measured "
        "<em>relative to</em> expected goals + assists, players arriving with high prior output face higher expectations and tend to score lower by construction.",
        "info",
    )

fc1, fc2, fc3 = st.columns(3)
with fc1:
    psel = st.selectbox("Position", ["All positions"] + sorted(backtest["position"].dropna().astype(str).unique()), key="bt_pos")
with fc2:
    dsel = st.selectbox("Destination club", ["All clubs"] + sorted(backtest["tm_to_club"].dropna().astype(str).unique()), key="bt_dest")
with fc3:
    lsel = st.selectbox("2025-26 outcome", ["All outcomes", "Success", "Failure", "Ambiguous"], key="bt_lbl")

shown = backtest.copy()
if psel != "All positions":
    shown = shown[shown["position"].astype(str) == psel]
if dsel != "All clubs":
    shown = shown[shown["tm_to_club"].astype(str) == dsel]
if lsel != "All outcomes" and "label_base_sec9" in shown:
    shown = shown[shown["label_base_sec9"].astype(str) == lsel]

plot_bt = shown.rename(columns={
    "tm_from_club": "Previous Club",
    "tm_to_club": "New Club",
    "position": "Position",
    "MS": "Minutes Share",
    "fit_score": "Pre-Transfer Fit Score",
    "Y_score": "2025-26 Impact vs. Expected",
    "label_base_sec9": "Season Verdict",
})

cl, cr = st.columns(2)
with cl:
    sc = plot_bt.dropna(subset=["Pre-Transfer Fit Score", "2025-26 Impact vs. Expected"])
    fig_sc = px.scatter(
        sc,
        x="Pre-Transfer Fit Score",
        y="2025-26 Impact vs. Expected",
        color="Season Verdict",
        hover_name="player_name",
        hover_data={
            "Previous Club": True,
            "New Club": True,
            "Position": True,
            "Minutes Share": ":.1%",
            "Pre-Transfer Fit Score": ":.1f",
            "2025-26 Impact vs. Expected": ":+.2f",
        },
        color_discrete_map={"Success": "#10B981", "Failure": "#F43F5E", "Ambiguous": "#71717A"},
        title="Fit Score vs. 2025-26 On-Pitch Impact",
    )
    fig_sc.add_hline(y=0, line_dash="dash", line_color="#52525B")
    fig_sc.update_traces(marker=dict(size=9, line=dict(width=0.8, color="#121318")))
    st.plotly_chart(style_fig(fig_sc, 420), width="stretch", theme=None)

with cr:
    bx = plot_bt[plot_bt["Season Verdict"].isin(["Success", "Failure"])]
    fig_bx = px.box(
        bx,
        x="Season Verdict",
        y="Pre-Transfer Fit Score",
        color="Season Verdict",
        points="all",
        hover_name="player_name",
        hover_data={"Previous Club": True, "New Club": True, "Pre-Transfer Fit Score": ":.1f"},
        color_discrete_map={"Success": "#10B981", "Failure": "#F43F5E"},
        title="Fit Score Distribution by 2025-26 Outcome",
    )
    fig_bx.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig_bx, 420), width="stretch", theme=None)

st.markdown(f"#### Evaluated Transfers ({len(shown)} shown)")
cols = [
    "player_name", "position", "sub_position", "tm_from_club", "tm_to_club",
    "fit_score", "role_fit_100", "system_fit_100", "closest_dest_peer", "MS", "Y_score", "label_base_sec9",
]
cols = [c for c in cols if c in shown.columns]
table = shown[cols].sort_values("fit_score", ascending=False).rename(columns={
    "player_name": "Player",
    "position": "Position",
    "sub_position": "Role",
    "tm_from_club": "Previous Club",
    "tm_to_club": "New Club",
    "fit_score": "Fit Score",
    "role_fit_100": "Role Fit",
    "system_fit_100": "System Fit",
    "closest_dest_peer": "Closest Peer",
    "MS": "Squad Minutes Share",
    "Y_score": "Impact vs. Expected",
    "label_base_sec9": "2025-26 Outcome",
})
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download Evaluated Transfers (CSV)", table.to_csv(index=False), file_name="backtest_transfers.csv", mime="text/csv")
footer_note()