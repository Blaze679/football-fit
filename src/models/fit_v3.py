from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED = Path("data/processed")

ON_BALL_VOL_COLS = [
    "touches_p90", "box_touches_p90", "passes_att_p90", "prog_passes_p90",
    "f3_passes_succ_p90", "key_passes_p90", "throughballs_p90", "crosses_p90",
    "takeons_att_p90", "dispossessed_p90", "fouls_won_p90", "shots_p90",
]
DEF_VOL_COLS = [
    "tackles_int_p90", "ball_recoveries_p90", "high_press_actions_p90",
    "clearances_p90", "aerials_att_p90",
]
RATIO_SPATIAL_COLS = [
    "avg_touch_x", "avg_touch_y_centrality", "att_third_touch_share",
    "pass_acc", "avg_pass_len", "long_ball_share", "fwd_pass_share",
    "open_cross_share", "takeon_succ_rate", "avg_shot_dist_m",
    "headed_shot_share", "avg_def_action_x", "aerial_win_rate",
]
PLAYER_STYLE_COLS = ON_BALL_VOL_COLS + DEF_VOL_COLS + RATIO_SPATIAL_COLS

CLUB_STYLE_COLS = [
    "poss_share", "passes_p90", "pass_acc", "ppda", "high_turnovers_p90",
    "field_tilt", "f3_pass_share", "box_touches_p90", "long_ball_share",
    "prog_pass_share", "vert_pass_share", "avg_pass_length",
    "tempo_passes_per_poss_min", "touch_width_sd", "wide_touch_share",
    "crosses_p90", "def_line_height_m", "clearance_share",
    "shots_p90", "throughballs_p90", "takeons_p90", "aerials_p90",
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    auc = float(roc_auc_score(y_true, y_score))
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return auc, float(lo), float(hi)


def poss_adjust_df(df_in: pd.DataFrame, club_poss_map: dict) -> pd.DataFrame:
    df = df_in.copy()
    poss = df["team"].map(club_poss_map).fillna(0.50).clip(0.30, 0.70)
    on_scale = 0.50 / poss
    def_scale = 0.50 / (1.0 - poss)
    for c in ON_BALL_VOL_COLS:
        df[c] = df[c] * on_scale
    for c in DEF_VOL_COLS:
        df[c] = df[c] * def_scale
    return df


# 1. Load strictly 2024-25 data
ps = pd.read_parquet(PROCESSED / "player_style_features.parquet")
cs = pd.read_parquet(PROCESSED / "club_style_features.parquet")
master = pd.read_parquet(PROCESSED / "master_transfer_dataset.parquet")

cs24 = cs[cs["season"] == 2425].copy()
club_poss_24 = dict(zip(cs24["team"], cs24["poss_share"]))

ps24_raw = ps[
    (ps["season"] == 2425)
    & (ps["minutes_90"] >= 450)
    & (ps["position"].isin(["Attack", "Midfield", "Defender"]))
].copy()

pos_medians_24 = {}
for col in PLAYER_STYLE_COLS:
    pos_med = ps24_raw.groupby("position")[col].transform("median")
    ps24_raw[col] = ps24_raw[col].fillna(pos_med).fillna(ps24_raw[col].median())
    pos_medians_24[col] = float(ps24_raw[col].median())

ps24 = poss_adjust_df(ps24_raw, club_poss_24)

# Fit 2024-25 Scalers, PCA, UMAP, KMeans
p_scaler = StandardScaler()
X_p24 = p_scaler.fit_transform(ps24[PLAYER_STYLE_COLS].values)

pca = PCA(n_components=6, random_state=42)
Z_pca = pca.fit_transform(X_p24)
for i in range(6):
    ps24[f"pca_{i+1}"] = np.round(Z_pca[:, i], 4)

import umap
reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2, random_state=42)
Z_umap = reducer.fit_transform(X_p24)
ps24["umap_1"] = np.round(Z_umap[:, 0], 4)
ps24["umap_2"] = np.round(Z_umap[:, 1], 4)

km = KMeans(n_clusters=8, n_init=20, random_state=42)
ps24["style_cluster"] = km.fit_predict(Z_pca[:, :5])
ps24.to_parquet(PROCESSED / "player_style_space_2425.parquet", index=False)

c_scaler = StandardScaler()
X_c24 = c_scaler.fit_transform(cs24[CLUB_STYLE_COLS].values)
c24_z_df = pd.DataFrame(X_c24, columns=CLUB_STYLE_COLS, index=cs24["team"])
club_vec_24 = {t: X_c24[i] for i, t in enumerate(cs24["team"])}

# Save scalers & standardized vectors for Streamlit GUI inference
np.savez(
    PROCESSED / "fit_scalers_2425.npz",
    p_mean=p_scaler.mean_,
    p_scale=p_scaler.scale_,
    c_mean=c_scaler.mean_,
    c_scale=c_scaler.scale_,
    player_cols=np.array(PLAYER_STYLE_COLS),
    club_cols=np.array(CLUB_STYLE_COLS),
)

ps24_z = pd.DataFrame(X_p24, columns=PLAYER_STYLE_COLS, index=ps24.index)
ps24_z["player_id"] = ps24["player_id"].values
ps24_z["player"] = ps24["player"].values
ps24_z["team"] = ps24["team"].values
ps24_z["league"] = ps24["league"].values
ps24_z["position"] = ps24["position"].values
ps24_z["sub_position"] = ps24["sub_position"].values
ps24_z["minutes_90"] = ps24["minutes_90"].values
ps24_z["style_cluster"] = ps24["style_cluster"].values
ps24_z.to_parquet(PROCESSED / "player_role_vectors_2425.parquet", index=False)

# Build possession-adjusted 2024-25 player vectors for all 2425 players
ps24_all_valid = ps[(ps["season"] == 2425) & (ps["minutes_90"] > 0)].copy()
for col in PLAYER_STYLE_COLS:
    pos_med = ps24_all_valid.groupby("position")[col].transform("median")
    ps24_all_valid[col] = ps24_all_valid[col].fillna(pos_med).fillna(pos_medians_24[col])
ps24_all_adj = poss_adjust_df(ps24_all_valid, club_poss_24)

p24_adj_map = {}
for pid, grp in ps24_all_adj.groupby("player_id"):
    w = grp["minutes_90"].values
    vec_raw = np.average(grp[PLAYER_STYLE_COLS].values, axis=0, weights=w)
    p24_adj_map[int(pid)] = p_scaler.transform(vec_raw.reshape(1, -1))[0]

p_idx = {c: i for i, c in enumerate(PLAYER_STYLE_COLS)}


def score_player_club(p_vec: np.ndarray, pos: str, subp: str | None, dest: str, orig: str | None = None):
    """Core reusable fit scorer (strictly 2024-25 artifacts)."""
    club_pos_pool = ps24_z[(ps24_z["team"] == dest) & (ps24_z["position"] == pos)]
    sub_pool = club_pos_pool[club_pos_pool["sub_position"] == subp] if pd.notna(subp) else pd.DataFrame()

    # Find closest individual stylistic peer at destination club
    peer_name = ""
    peer_sim = 0.0
    search_pool = sub_pool if len(sub_pool) >= 1 else club_pos_pool
    if len(search_pool) >= 1:
        sims = [
            (cosine_sim(p_vec, row_v), pname, m90)
            for row_v, pname, m90 in zip(
                search_pool[PLAYER_STYLE_COLS].values,
                search_pool["player"].values,
                search_pool["minutes_90"].values,
            )
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        peer_sim, peer_name = sims[0][0], sims[0][1]

    if len(sub_pool) >= 1 and sub_pool["minutes_90"].sum() >= 600:
        target_vec = np.average(sub_pool[PLAYER_STYLE_COLS].values, axis=0, weights=sub_pool["minutes_90"].values)
        r_cos = cosine_sim(p_vec, target_vec)
    elif len(club_pos_pool) >= 1:
        sims_all = [
            (cosine_sim(p_vec, row_v), m90)
            for row_v, m90 in zip(club_pos_pool[PLAYER_STYLE_COLS].values, club_pos_pool["minutes_90"].values)
        ]
        sims_all.sort(key=lambda x: x[0], reverse=True)
        top_k = sims_all[:3]
        r_cos = float(np.average([s[0] for s in top_k], weights=[s[1] for s in top_k]))
    else:
        r_cos = 0.0

    # Position-agnostic System & Pressing Fit (3 universal tactical dimensions)
    cz = c24_z_df.loc[dest]
    sys_axes = [
        # 1. Pressing & defensive line height
        ((-cz["ppda"] + cz["def_line_height_m"]) / 2.0) * ((p_vec[p_idx["high_press_actions_p90"]] + p_vec[p_idx["avg_def_action_x"]]) / 2.0),
        # 2. Possession / short build-up vs direct long-ball style
        ((cz["poss_share"] - cz["long_ball_share"]) / 2.0) * ((p_vec[p_idx["pass_acc"]] - p_vec[p_idx["long_ball_share"]] - p_vec[p_idx["dispossessed_p90"]]) / 3.0),
        # 3. Progressive vertical tempo
        ((cz["prog_pass_share"] + cz["vert_pass_share"]) / 2.0) * ((p_vec[p_idx["prog_passes_p90"]] + p_vec[p_idx["fwd_pass_share"]]) / 2.0),
    ]
    sys_fit = float(np.tanh(np.mean(sys_axes)))

    if orig is not None and orig in club_vec_24 and dest in club_vec_24:
        env_cos = cosine_sim(club_vec_24[orig], club_vec_24[dest])
    else:
        env_cos = 0.0

    raw_fit = 0.80 * r_cos + 0.15 * sys_fit + 0.05 * env_cos
    fit_100 = round(50.0 + 50.0 * raw_fit, 1)
    return {
        "fit_score": fit_100,
        "role_fit_100": round(50.0 + 50.0 * r_cos, 1),
        "system_fit_100": round(50.0 + 50.0 * sys_fit, 1),
        "env_fit_100": round(50.0 + 50.0 * env_cos, 1),
        "role_peer_cos": round(r_cos, 4),
        "system_fit": round(sys_fit, 4),
        "env_cos": round(env_cos, 4),
        "closest_dest_peer": peer_name,
        "closest_peer_sim": round(peer_sim, 3),
    }


# 2. Score Complete Eligible Outfield Transfers
elig_out = master[
    (master["elig_status"] == "eligible")
    & (master["position"].isin(["Attack", "Midfield", "Defender"]))
    & (master["dest_c24_poss_share"].notna())
].copy()

scored_rows = [
    score_player_club(
        p24_adj_map[int(r.ws_player_id)],
        r.position,
        r.sub_position,
        r.ws_to_club,
        r.ws_from_club,
    )
    for r in elig_out.itertuples()
]
scored_df = pd.DataFrame(scored_rows, index=elig_out.index)
elig_out = pd.concat([elig_out, scored_df], axis=1)
elig_out["club_strength_gap"] = (elig_out["dest_c24_field_tilt"] - elig_out["orig_c24_field_tilt"]).fillna(0.0)

# Section 9 Defender MS-only labels (reported separately from Attack+Midfield)
elig_out["label_base_sec9"] = np.where(
    elig_out["position"] == "Defender",
    np.where(elig_out["MS"] < 0.25, "Failure", np.where(elig_out["MS"] >= 0.50, "Success", "Ambiguous")),
    elig_out["label_base"],
)
elig_out["label_lenient_sec9"] = np.where(
    elig_out["position"] == "Defender",
    np.where(elig_out["MS"] < 0.15, "Failure", np.where(elig_out["MS"] >= 0.40, "Success", "Ambiguous")),
    elig_out["label_lenient"],
)
elig_out["label_strict_sec9"] = np.where(
    elig_out["position"] == "Defender",
    np.where(elig_out["MS"] < 0.35, "Failure", np.where(elig_out["MS"] >= 0.60, "Success", "Ambiguous")),
    elig_out["label_strict"],
)

elig_out.to_parquet(PROCESSED / "backtest_results_v3.parquet", index=False)
elig_out.to_csv(PROCESSED / "backtest_results_v3.csv", index=False)

# 3. Backtest & Sensitivity Report
print(f"=== 1. BACKTEST v3: CONTINUOUS SPEARMAN (ALL OUTFIELD n={len(elig_out)}) ===")
y_valid = elig_out.dropna(subset=["Y_score"])
for pred in ["fit_score", "role_peer_cos", "system_fit", "env_cos", "expected_ga_p90", "age_years", "club_strength_gap"]:
    r_y, p_y = spearmanr(y_valid[pred], y_valid["Y_score"])
    r_ms, p_ms = spearmanr(elig_out[pred], elig_out["MS"])
    print(f"  {pred:18s} | vs Y_score (n={len(y_valid)}): r={r_y:+.3f} (p={p_y:.3f}) | vs MS (n={len(elig_out)}): r={r_ms:+.3f} (p={p_ms:.3f})")

print("\n=== 2. POSITION-SPECIFIC EVALUATION (PER SECTION 9) ===")
att_mid = elig_out[elig_out["position"].isin(["Attack", "Midfield"])]
att_mid_y = att_mid.dropna(subset=["Y_score"])
r_am_y, p_am_y = spearmanr(att_mid_y["fit_score"], att_mid_y["Y_score"])
r_am_ms, p_am_ms = spearmanr(att_mid["fit_score"], att_mid["MS"])
print(f"  Attack+Midfield (n={len(att_mid):2d}) | vs Y_score (n={len(att_mid_y)}): r={r_am_y:+.3f} (p={p_am_y:.3f}) | vs MS: r={r_am_ms:+.3f} (p={p_am_ms:.3f})")
for pos in ["Attack", "Midfield", "Defender"]:
    sub = elig_out[elig_out["position"] == pos]
    sub_y = sub.dropna(subset=["Y_score"])
    ry, py = spearmanr(sub_y["fit_score"], sub_y["Y_score"])
    rms, pms = spearmanr(sub["fit_score"], sub["MS"])
    print(f"  {pos:15s} (n={len(sub):2d}) | vs Y_score: r={ry:+.3f} (p={py:.3f}) | vs MS: r={rms:+.3f} (p={pms:.3f})")

print("\n=== 3. ROC-AUC WITH SECTION 9 POSITION RULES (DEFENDERS ON MS-ONLY) ===")
for lbl_col in ["label_lenient_sec9", "label_base_sec9", "label_strict_sec9"]:
    bsub = elig_out[elig_out[lbl_col].isin(["Success", "Failure"])].copy()
    yt = (bsub[lbl_col] == "Success").astype(int).values
    auc_fit, lo, hi = bootstrap_auc(yt, bsub["fit_score"].values)
    auc_exp = float(roc_auc_score(yt, bsub["expected_ga_p90"].values))
    auc_age = float(roc_auc_score(yt, -bsub["age_years"].values))
    auc_gap = float(roc_auc_score(yt, -bsub["club_strength_gap"].values))
    print(
        f"  {lbl_col:18s} (n={len(bsub):2d}, pos={yt.sum():2d}/neg={(1-yt).sum():2d}) | "
        f"fit_v3 AUC={auc_fit:.3f} [{lo:.3f}, {hi:.3f}] | "
        f"exp_ga={auc_exp:.3f} | younger={auc_age:.3f} | small_gap={auc_gap:.3f}"
    )

print("\n=== 4. SENSITIVITY CHECKS (SECTION 9) ===")
s900 = elig_out[elig_out["min90_2425"] >= 900]
s900_y = s900.dropna(subset=["Y_score"])
r900_y, p900_y = spearmanr(s900_y["fit_score"], s900_y["Y_score"])
r900_ms, p900_ms = spearmanr(s900["fit_score"], s900["MS"])
b900 = s900[s900["label_base_sec9"].isin(["Success", "Failure"])]
auc900 = float(roc_auc_score((b900["label_base_sec9"] == "Success").astype(int), b900["fit_score"]))
print(f"  Baseline >=900 min (n={len(s900):3d}) | vs Y: r={r900_y:+.3f} (p={p900_y:.3f}) | vs MS: r={r900_ms:+.3f} (p={p900_ms:.3f}) | Base AUC={auc900:.3f}")

# High-fee transfers (>= 15M EUR, where club intent to play the signing is clear)
s_fee = elig_out[elig_out["transfer_fee"].fillna(0) >= 15_000_000]
s_fee_y = s_fee.dropna(subset=["Y_score"])
rfee_y, pfee_y = spearmanr(s_fee_y["fit_score"], s_fee_y["Y_score"])
rfee_ms, pfee_ms = spearmanr(s_fee["fit_score"], s_fee["MS"])
bfee = s_fee[s_fee["label_base_sec9"].isin(["Success", "Failure"])]
aucfee = float(roc_auc_score((bfee["label_base_sec9"] == "Success").astype(int), bfee["fit_score"]))
print(f"  Fee >= 15M EUR     (n={len(s_fee):3d}) | vs Y: r={rfee_y:+.3f} (p={pfee_y:.3f}) | vs MS: r={rfee_ms:+.3f} (p={pfee_ms:.3f}) | Base AUC={aucfee:.3f}")

print("\n=== 5. KEY CASE STUDIES WITH CLOSEST STYLISTIC PEER ===")
check_names = [
    "Martín Zubimendi", "Adrien Truffert", "Luis Díaz", "João Pedro",
    "Hugo Ekitiké", "Jonathan Tah", "Florian Wirtz", "Alexander Isak",
    "Anthony Elanga", "Tariq Lamptey", "Christian Nørgaard"
]
show = ["player_name", "tm_from_club", "tm_to_club", "fit_score", "role_fit_100", "system_fit_100", "closest_dest_peer", "MS", "Y_score", "label_base_sec9"]
print(elig_out[elig_out["player_name"].isin(check_names)].sort_values("fit_score", ascending=False)[show].to_string(index=False))