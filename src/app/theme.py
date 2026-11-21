"""Obsidian charcoal & electric cyan styling, clean avatar/crest framing, and card components."""
from __future__ import annotations

import html
import re
import urllib.parse
import pandas as pd
import streamlit as st


def inject_global_css() -> None:
    """Obsidian dark theme built on top of .streamlit/config.toml."""
    st.markdown(
        """
<style>
.block-container {
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1380px;
    font-variant-numeric: tabular-nums;
}
section[data-testid="stSidebar"] {
    background-color: #070709;
    border-right: 1px solid #27272A;
}
footer { visibility: hidden; }
h1, h2, h3, h4 { letter-spacing: -0.015em; color: #FAFAFA; }

/* KPI Metric Cards */
div[data-testid="stMetric"] {
    background: #121318;
    border: 1px solid #27272A;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
div[data-testid="stMetricLabel"] p {
    color: #A1A1AA !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetricValue"] {
    color: #FAFAFA !important;
    font-weight: 800 !important;
    font-variant-numeric: tabular-nums;
}

/* Obsidian & Cyan Hero Banner */
.ff-hero {
    background: linear-gradient(135deg, #101116 0%, #141923 55%, #0D2538 100%);
    border: 1px solid #27272A;
    border-left: 4px solid #38BDF8;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
}
.ff-hero-tag {
    display: inline-block;
    background: rgba(56, 189, 248, 0.14);
    color: #7DD3FC;
    border: 1px solid rgba(56, 189, 248, 0.32);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.ff-hero h1 {
    margin: 0 0 6px 0;
    padding: 0;
    font-size: 1.95rem;
    font-weight: 800;
    color: #FAFAFA;
}
.ff-hero p { margin: 0; color: #D4D4D8; font-size: 0.97rem; line-height: 1.55; }

/* Elevated Carbon Cards */
.ff-card {
    background: #121318;
    border: 1px solid #27272A;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.30);
    color: #E4E4E7;
}
.ff-card-emerald { border-top: 2px solid #38BDF8; }
.ff-card-cyan    { border-top: 2px solid #38BDF8; }
.ff-card-violet  { border-top: 2px solid #71717A; }
.ff-card-amber   { border-top: 2px solid #F59E0B; }
.ff-card-rose    { border-top: 2px solid #F43F5E; }
.ff-card-title   { font-size: 1.04rem; font-weight: 700; margin-bottom: 6px; color: #FAFAFA; }
.ff-card-text    { color: #A1A1AA; font-size: 0.90rem; line-height: 1.52; }

/* Callout Boxes */
.ff-callout {
    border-left: 3px solid #38BDF8;
    background: rgba(56, 189, 248, 0.08);
    border-radius: 8px;
    padding: 12px 16px;
    color: #E4E4E7;
    font-size: 0.91rem;
    line-height: 1.55;
    margin: 8px 0 14px 0;
}
.ff-callout-amber {
    border-left-color: #F59E0B;
    background: rgba(245, 158, 11, 0.09);
    color: #FDE68A;
}
.ff-callout-rose {
    border-left-color: #F43F5E;
    background: rgba(244, 63, 94, 0.09);
    color: #FECDD3;
}
.ff-footer {
    color: #71717A;
    font-size: 0.78rem;
    margin-top: 2.2rem;
    border-top: 1px solid #27272A;
    padding-top: 0.85rem;
}

/* Clean Studio Portraits & Club Crests */
.ff-entity-wrap { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.ff-portrait, .ff-crest-lg, .ff-crest-sm { display: inline-block; background-repeat: no-repeat; }
.ff-portrait {
    width: 90px; height: 90px; border-radius: 50%;
    background-size: contain;
    background-position: center bottom;
    background-color: #18181B;
    border: 2px solid #3F3F46;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    flex-shrink: 0;
}
.ff-crest-lg {
    width: 84px; height: 84px;
    background-color: #F4F4F5;
    background-size: contain;
    background-position: center;
    background-origin: content-box;
    border-radius: 14px;
    padding: 8px;
    box-sizing: border-box;
    border: 1px solid #3F3F46;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    flex-shrink: 0;
}
.ff-crest-sm {
    width: 24px; height: 24px;
    background-color: #F4F4F5;
    background-size: contain;
    background-position: center;
    background-origin: content-box;
    border-radius: 6px;
    padding: 2px;
    box-sizing: border-box;
    vertical-align: middle;
    margin-right: 6px;
}

/* Restrained Semantic Pills */
.ff-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.77rem;
    font-weight: 600;
    margin: 3px 5px 3px 0;
}
.ff-pill-green  { background: rgba(16, 185, 129, 0.14); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.30); }
.ff-pill-cyan   { background: rgba(56, 189, 248, 0.14); color: #7DD3FC; border: 1px solid rgba(56, 189, 248, 0.30); }
.ff-pill-violet { background: rgba(167, 139, 250, 0.14); color: #C4B5FD; border: 1px solid rgba(167, 139, 250, 0.30); }
.ff-pill-amber  { background: rgba(245, 158, 11, 0.14); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.30); }
.ff-pill-rose   { background: rgba(244, 63, 94, 0.14);  color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.30); }
.ff-pill-slate  { background: #18181B; color: #D4D4D8; border: 1px solid #27272A; }

/* Percentile Bars */
.ff-bar-row { margin-bottom: 12px; }
.ff-bar-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; gap: 10px; }
.ff-bar-title { color: #FAFAFA; font-weight: 600; font-size: 0.89rem; }
.ff-bar-sub   { color: #A1A1AA; font-size: 0.76rem; }
.ff-bar-val   { font-weight: 700; font-size: 0.85rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
.ff-bar-track { width: 100%; height: 8px; background: #27272A; border-radius: 999px; overflow: hidden; }
.ff-bar-fill  { height: 100%; border-radius: 999px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def page_header(tag: str, title: str, subtitle: str) -> None:
    """Consistent obsidian hero banner used on every page."""
    st.markdown(
        f'<div class="ff-hero"><div class="ff-hero-tag">{html.escape(tag)}</div>'
        f"<h1>{html.escape(title)}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info") -> None:
    """Highlighted note box. kind: info | amber | rose."""
    cls = {"info": "", "amber": " ff-callout-amber", "rose": " ff-callout-rose"}.get(kind, "")
    st.markdown(f'<div class="ff-callout{cls}">{text}</div>', unsafe_allow_html=True)


def footer_note() -> None:
    st.markdown(
        '<div class="ff-footer">Football Fit · Tactical Recruitment Analytics · WhoScored match events and Transfermarkt '
        "public data (2024-25 &amp; 2025-26) · Fit scores measure tactical style similarity, not a guarantee of transfer outcome.</div>",
        unsafe_allow_html=True,
    )


def fallback_avatar_uri(name: str, bg_hex: str = "#18181B", fg_hex: str = "#38BDF8") -> str:
    """Return an inline SVG data-URI avatar with player/club initials."""
    parts = [p[0].upper() for p in str(name).split() if p]
    initials = "".join(parts[:2]) or "FF"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">'
        f'<rect width="120" height="120" rx="60" fill="{bg_hex}"/>'
        f'<text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="{fg_hex}" font-family="Inter,Arial,sans-serif" font-size="40" font-weight="800">'
        f'{html.escape(initials)}</text></svg>'
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def img_tag(url: str | None, fallback_name: str, css_class: str = "ff-portrait", is_club: bool = False) -> str:
    """Render photo/crest cleanly without stacking SVG text underneath transparent PNG cutouts."""
    label = html.escape(str(fallback_name))
    has_url = url is not None and str(url).strip() not in ("", "None", "nan", "<NA>")
    if has_url:
        clean_url = str(url).strip().replace("'", "%27").replace('"', "%22")
        style = f' style="background-image:url(\'{clean_url}\');"'
    elif not is_club:
        style = (
            f' style="background-image:url(\'{fallback_avatar_uri(fallback_name)}\');'
            f'background-size:cover;background-position:center;"'
        )
    else:
        style = ""
    return f'<span class="{css_class}" role="img" aria-label="{label}" title="{label}"{style}></span>'


def ordinal(n: int | float) -> str:
    """1 -> 1st, 2 -> 2nd, 11 -> 11th, 22 -> 22nd."""
    n = int(round(float(n)))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fix_ordinals(text: str) -> str:
    """Repair strings such as '91th pct' / '2th pct' into proper ordinals."""
    return re.sub(r"(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal(m.group(1)), str(text))


def percentile_color(pct: int | float) -> tuple[str, str]:
    """Return (hex_color, human_tier_label) calibrated for dark cards."""
    val = pd.to_numeric(pct, errors="coerce")
    p = 50.0 if pd.isna(val) else float(val)
    if p >= 80:
        return "#10B981", "Elite"
    if p >= 60:
        return "#38BDF8", "Above Avg"
    if p >= 40:
        return "#F59E0B", "Average"
    return "#FB7185", "Below Avg"


def fit_tier(score: float) -> tuple[str, str, str]:
    """Return (tier_label, hex_color, pill_class) for a 0-100 fit score."""
    if score >= 80.0:
        return "Elite Tactical Fit", "#10B981", "ff-pill-green"
    if score >= 70.0:
        return "Strong Tactical Fit", "#38BDF8", "ff-pill-cyan"
    if score >= 58.0:
        return "Moderate / Adaptable Fit", "#F59E0B", "ff-pill-amber"
    return "Low Tactical Fit", "#F43F5E", "ff-pill-rose"


def render_percentile_bar(title: str, pct: int | float, raw_display: str = "", subtitle: str = "") -> str:
    """Return HTML for a clean, non-technical horizontal percentile bar."""
    val = pd.to_numeric(pct, errors="coerce")
    p_float = 50.0 if pd.isna(val) else float(val)
    p_int = int(max(1, min(99, round(p_float))))
    color, tier = percentile_color(p_int)
    sub_html = f'<div class="ff-bar-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    raw_txt = f"{html.escape(raw_display)} · " if raw_display else ""
    return (
        f'<div class="ff-bar-row">'
        f'<div class="ff-bar-head">'
        f'<div><span class="ff-bar-title">{html.escape(title)}</span>{sub_html}</div>'
        f'<div class="ff-bar-val" style="color:{color};">{raw_txt}{ordinal(p_int)} pct ({tier})</div>'
        f'</div>'
        f'<div class="ff-bar-track"><div class="ff-bar-fill" style="width:{p_int}%;background:{color};"></div></div>'
        f'</div>'
    )


def format_eur(val: float | int | None) -> str:
    """Format a Euro market value cleanly."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if v != v:
        return "N/A"
    if v <= 0:
        return "Undisclosed"
    if v >= 1_000_000:
        return f"€{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"€{v / 1_000:.0f}K"
    return f"€{v:.0f}"


def short_league(name: str | None) -> str:
    """'ENG-Premier League' -> 'Premier League'."""
    s = str(name or "")
    for prefix in ("ENG-", "ESP-", "ITA-", "GER-", "FRA-"):
        s = s.replace(prefix, "")
    return s