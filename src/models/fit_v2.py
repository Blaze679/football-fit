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

for col in PLAYER_STYLE_COLS:
    pos_med = ps24_raw.groupby("position")[col].transform("median")
    ps24_raw[col] = ps24_raw[col].fillna(pos_med).fillna(ps24_raw[col].median())

ps24 = poss_adjust_df(ps24_raw, club_poss_24)

# Fit 2024-25 Player Style Scaler, PCA, UMAP, KMeans
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

# Fit 2024-25 Club Tactical Scaler
c_scaler = StandardScaler()
X_c24 = c_scaler.fit_transform(cs24[CLUB_STYLE_COLS].values)
c24_z_df = pd.DataFrame(X_c24, columns=CLUB_STYLE_COLS, index=cs24["team"])
club_vec_24 = {t: X_c24[i] for i, t in enumerate(cs24["team"])}

# Index destination-club 2024-25 players in standardized style space
ps24_z = pd.DataFrame(X_p24, columns=PLAYER_STYLE_COLS, index=ps24.index)
ps24_z["team"] = ps24["team"].values
ps24_z["position"] = ps24["position"].values
ps24_z["sub_position"] = ps24["sub_position"].values
ps24_z["minutes_90"] = ps24["minutes_90"].values

# Build possession-adjusted 2024-25 player vectors for transfer cohort
ps24_all_valid = ps[(ps["season"] == 2425) & (ps["minutes_90"] > 0)].copy()
for col in PLAYER_STYLE_COLS:
    pos_med = ps24_all_valid.groupby("position")[col].transform("median")
    ps24_all_valid[col] = ps24_all_valid[col].fillna(pos_med).fillna(ps24_raw[col].median())
ps24_all_adj = poss_adjust_df(ps24_all_valid, club_poss_24)

p24_adj_map = {}
for pid, grp in ps24_all_adj.groupby("player_id"):
    w = grp["minutes_90"].values
    vec_raw = np.average(grp[PLAYER_STYLE_COLS].values, axis=0, weights=w)
    p24_adj_map[int(pid)] = p_scaler.transform(vec_raw.reshape(1, -1))[0]

p_idx = {c: i for i, c in enumerate(PLAYER_STYLE_COLS)}

# 2. Compute Fit v2 for Complete Eligible Outfield Transfers
elig_out = master[
    (master["elig_status"] == "eligible")
    & (master["position"].isin(["Attack", "Midfield", "Defender"]))
    & (master["dest_c24_poss_share"].notna())
].copy()

role_peer_cos = []
tactical_align = []
env_cos_list = []

for r in elig_out.itertuples():
    pid = int(r.ws_player_id)
    p_vec = p24_adj_map[pid]
    dest = r.ws_to_club
    orig = r.ws_from_club
    pos = r.position
    subp = r.sub_position

    # A. Destination club role-peer similarity (same sub_position first; else top-3 closest peers in broad position)
    club_pos_pool = ps24_z[(ps24_z["team"] == dest) & (ps24_z["position"] == pos)]
    sub_pool = club_pos_pool[club_pos_pool["sub_position"] == subp] if pd.notna(subp) else pd.DataFrame()

    if len(sub_pool) >= 1 and sub_pool["minutes_90"].sum() >= 600:
        target_vec = np.average(sub_pool[PLAYER_STYLE_COLS].values, axis=0, weights=sub_pool["minutes_90"].values)
        r_cos = cosine_sim(p_vec, target_vec)
    elif len(club_pos_pool) >= 1:
        # Compare against up to 3 closest stylistic peers in that broad position at the destination club
        sims = [
            (cosine_sim(p_vec, row_v), m90)
            for row_v, m90 in zip(club_pos_pool[PLAYER_STYLE_COLS].values, club_pos_pool["minutes_90"].values)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = sims[:3]
        r_cos = float(np.average([s[0] for s in top_k], weights=[s[1] for s in top_k]))
    else:
        r_cos = 0.0
    role_peer_cos.append(r_cos)

    # B. Explicit Player-to-Club Tactical Demand Alignment (z-score dot-product alignment on 6 tactical axes)
    cz = c24_z_df.loc[dest]
    axes = [
        # 1. High press & high line demand <-> player pressing & defensive height
        ((-cz["ppda"] + cz["def_line_height_m"]) / 2.0) * ((p_vec[p_idx["high_press_actions_p90"]] + p_vec[p_idx["avg_def_action_x"]]) / 2.0),
        # 2. Possession & short build-up <-> passing accuracy & ball security
        ((cz["poss_share"] - cz["long_ball_share"]) / 2.0) * ((p_vec[p_idx["pass_acc"]] - p_vec[p_idx["dispossessed_p90"]]) / 2.0),
        # 3. Territory & box dominance <-> final-third & box presence
        ((cz["field_tilt"] + cz["box_touches_p90"]) / 2.0) * ((p_vec[p_idx["att_third_touch_share"]] + p_vec[p_idx["box_touches_p90"]]) / 2.0),
        # 4. Crossing / aerial directness <-> crossing or aerial target traits
        ((cz["crosses_p90"] + cz["long_ball_share"]) / 2.0) * max(p_vec[p_idx["crosses_p90"]], p_vec[p_idx["aerial_win_rate"]]),
        # 5. Vertical / progressive tempo <-> progressive & forward passing
        ((cz["prog_pass_share"] + cz["vert_pass_share"]) / 2.0) * ((p_vec[p_idx["prog_passes_p90"]] + p_vec[p_idx["fwd_pass_share"]]) / 2.0),
        # 6. 1v1 / penetration <-> take-on volume & throughballs
        ((cz["takeons_p90"] + cz["throughballs_p90"]) / 2.0) * ((p_vec[p_idx["takeons_att_p90"]] + p_vec[p_idx["throughballs_p90"]]) / 2.0),
    ]
    # Squash mean axis product into [-1, 1] via tanh
    t_align = float(np.tanh(np.mean(axes)))
    tactical_align.append(t_align)

    # C. Origin-to-Destination Club Tactical Environment Continuity
    if orig in club_vec_24 and dest in club_vec_24:
        env_cos_list.append(cosine_sim(club_vec_24[orig], club_vec_24[dest]))
    else:
        env_cos_list.append(0.0)

elig_out["role_peer_cos"] = np.round(role_peer_cos, 4)
elig_out["tactical_align"] = np.round(tactical_align, 4)
elig_out["env_cos"] = np.round(env_cos_list, 4)

# Fit Score v2 (0-100 scale): 55% role-peer fit + 30% tactical demand alignment + 15% club environment continuity
raw_v2 = 0.55 * elig_out["role_peer_cos"] + 0.30 * elig_out["tactical_align"] + 0.15 * elig_out["env_cos"]
elig_out["fit_score_v2"] = np.round(50.0 + 50.0 * raw_v2, 1)
elig_out["club_strength_gap"] = (elig_out["dest_c24_field_tilt"] - elig_out["orig_c24_field_tilt"]).fillna(0.0)

elig_out.to_parquet(PROCESSED / "backtest_results_v2.parquet", index=False)
elig_out.to_csv(PROCESSED / "backtest_results_v2.csv", index=False)

# 3. Backtest v2 Evaluation
print(f"=== 1. BACKTEST v2: CONTINUOUS SPEARMAN CORRELATIONS (n={len(elig_out)}) ===")
y_valid = elig_out.dropna(subset=["Y_score"])
for pred in ["fit_score_v2", "role_peer_cos", "tactical_align", "env_cos", "expected_ga_p90", "age_years", "club_strength_gap"]:
    r_y, p_y = spearmanr(y_valid[pred], y_valid["Y_score"])
    r_ms, p_ms = spearmanr(elig_out[pred], elig_out["MS"])
    print(f"  {pred:18s} | vs Y_score (n={len(y_valid)}): r={r_y:+.3f} (p={p_y:.3f}) | vs MS (n={len(elig_out)}): r={r_ms:+.3f} (p={p_ms:.3f})")

print("\n=== 2. SPEARMAN BY POSITION GROUP (fit_score_v2) ===")
for pos in ["Attack", "Midfield", "Defender"]:
    sub_all = elig_out[elig_out["position"] == pos]
    sub_y = sub_all.dropna(subset=["Y_score"])
    r_y, p_y = spearmanr(sub_y["fit_score_v2"], sub_y["Y_score"])
    r_ms, p_ms = spearmanr(sub_all["fit_score_v2"], sub_all["MS"])
    print(f"  {pos:8s} (n={len(sub_all):2d}) | fit_v2 vs Y: r={r_y:+.3f} (p={p_y:.3f}) | fit_v2 vs MS: r={r_ms:+.3f} (p={p_ms:.3f})")

print("\n=== 3. BINARY CLASSIFICATION ROC-AUC (SUCCESS vs FAILURE, 1000 BOOTSTRAPS) ===")
for lbl_col in ["label_lenient", "label_base", "label_strict"]:
    bsub = elig_out[elig_out[lbl_col].isin(["Success", "Failure"])].copy()
    yt = (bsub[lbl_col] == "Success").astype(int).values
    auc_fit, lo, hi = bootstrap_auc(yt, bsub["fit_score_v2"].values)
    auc_exp = float(roc_auc_score(yt, bsub["expected_ga_p90"].values))
    auc_age = float(roc_auc_score(yt, -bsub["age_years"].values))
    auc_gap = float(roc_auc_score(yt, -bsub["club_strength_gap"].values))
    print(
        f"  {lbl_col:13s} (n={len(bsub):2d}, pos={yt.sum():2d}/neg={(1-yt).sum():2d}) | "
        f"fit_v2 AUC={auc_fit:.3f} [{lo:.3f}, {hi:.3f}] | "
        f"exp_ga={auc_exp:.3f} | younger={auc_age:.3f} | small_gap={auc_gap:.3f}"
    )

print("\n=== 4. HIGH-PROFILE & KEY CASE CHECKS (fit_score_v2) ===")
check_names = [
    "Martín Zubimendi", "Adrien Truffert", "Luis Díaz", "João Pedro",
    "Hugo Ekitiké", "Jonathan Tah", "Florian Wirtz", "Alexander Isak",
    "Anthony Elanga", "Tariq Lamptey", "Christian Nørgaard"
]
show = ["player_name", "position", "tm_from_club", "tm_to_club", "fit_score_v2", "role_peer_cos", "tactical_align", "env_cos", "MS", "Y_score", "label_base"]
print(elig_out[elig_out["player_name"].isin(check_names)].sort_values("fit_score_v2", ascending=False)[show].to_string(index=False))