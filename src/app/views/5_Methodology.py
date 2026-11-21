"""Method and limitations — what was done, what the numbers mean, and what they cannot say."""
from __future__ import annotations

import streamlit as st
from theme import callout, footer_note, page_header

page_header(
    "Transparency & Research Design",
    "Method & Limitations",
    "How the data was collected and processed, how the fit score is constructed, how it was tested, and where it should not be over-interpreted.",
)

st.markdown("### Pipeline at a Glance")
c1, c2, c3, c4 = st.columns(4)
for col, title, text in [
    (c1, "1 · Collect", "WhoScored match events for 2024-25 and 2025-26 across Europe's top five leagues (3,504 matches), paired with Transfermarkt transfer, position, and age records."),
    (c2, "2 · Clean", "Pitch coordinates converted to metres; player minutes rebuilt from line-ups, substitutions, and cards; goals and assists verified from match events."),
    (c3, "3 · Model", "30 player style features and 22 club style features; possession-adjusted and standardised on 2024-25 only; fit score built from role, system, and club continuity."),
    (c4, "4 · Test", "Permanent Summer 2025 signings evaluated out-of-sample using 2025-26 minutes share and output versus expectation (calibrated on non-transferring peers)."),
]:
    with col:
        st.markdown(
            f'<div class="ff-card" style="min-height:185px;"><div class="ff-card-title">{title}</div><div class="ff-card-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("### How the Fit Score Is Constructed")
st.markdown(
    """
- **Player style** is represented by 30 per-90 and spatial ratio features (touches, progression, chance creation, take-ons, pressing, aerial duels, and average pitch coordinates). On-ball volumes are rescaled to a 50% possession baseline and defensive volumes to the matching share, so a player is neither rewarded nor penalised for their club's ball share.
- **Club style** is represented by 22 tactical features (possession share, pressing intensity, final-third territory, directness, circulation tempo, width, and defensive-line height).
- **Fit score** = $50 + 50 \\times (0.80 \\times \\text{role similarity} + 0.15 \\times \\text{system fit} + 0.05 \\times \\text{club continuity})$, bounded on a 0–100 scale. Role similarity measures cosine similarity between the player and the target club's players in the same role; system fit checks pressing, possession, and tempo alignment; continuity compares origin and target club styles.
- The component weights (80 / 15 / 5) and the three system axes are **domain modelling choices**, not parameters fitted on transfer outcomes.
    """
)

st.markdown("### How It Was Tested")
st.markdown(
    """
- **Cohort:** permanent Summer 2025 transfers between clubs that competed in a Top-5 league in both 2024-25 and 2025-26, after excluding inferred loans, players aged 21 or under, and low-minute samples. Goalkeepers are tracked on minutes share but are not scored by the outfield style model.
- **Outcome:** squad minutes share at the new club, plus goals + assists per 90 compared against an expectation calibrated on players who remained at their club (with empirical Bayes shrinkage toward the position mean and an age curve). Defenders are judged on minutes share only.
- **Leakage control:** all scalers, style clusters, and club vectors are fitted strictly on 2024-25 data. Goals and assists are excluded from fit inputs, and 2025-26 data is never used to score a transfer.
    """
)

st.markdown("### Limitations")
callout(
    "<strong>Small sample size.</strong> The backtest covers 133 transfers, so correlations and AUCs carry wide confidence intervals. "
    "<strong>Multiple variants compared.</strong> Three score variants and three outcome thresholds were examined on the same cohort, so reported metrics are exploratory.",
    "amber",
)
st.markdown(
    """
- **Loans are inferred.** Public Transfermarkt tables lack an explicit loan flag; loans and loan returns were identified from reverse moves, so edge cases may remain.
- **No injury or contract data.** A player who missed matches through injury looks identical in minutes data to one dropped on form.
- **Event data only.** Match events capture on-ball actions and defensive interventions, not off-ball runs, physical load, or dressing-room dynamics.
- **Style is not quality.** A high fit indicates similar tactical habits to the target club's players in that role, not that the player meets the club's overall quality threshold.
- **Many causes of transfer outcomes.** Managerial changes, squad injuries, and tactical shifts can alter a signing's trajectory independently of pre-transfer style fit.
- **Promoted clubs.** Clubs without a 2024-25 Top-5 profile cannot serve as destination clubs in the 2024-25 baseline model.
- **Season mixing in scouting mode.** When 2025-26 player profiles are evaluated against 2024-25 club profiles, that cross-season pairing is a deliberate baseline approximation.
    """
)
footer_note()