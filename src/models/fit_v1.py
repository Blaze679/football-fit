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

PLAYER_STYLE_COLS = [
    "touches_p90", "avg_touch_x", "avg_touch_y_centrality",
    "att_third_touch_share", "box_touches_p90",
    "passes_att_p90", "pass_acc", "avg_pass_len", "long_ball_share",
    "prog_passes_p90", "f3_passes_succ_p90", "fwd_pass_share",
    "key_passes_p90", "throughballs_p90", "crosses_p90", "open_cross_share",
    "takeons_att_p90", "takeon_succ_rate", "dispossessed_p90", "fouls_won_p90",
    "shots_p90", "avg_shot_dist_m", "headed_shot_share",
    "tackles_int_p90", "ball_recoveries_p90", "high_press_actions_p90",
    "avg_def_action_x", "clearances_p90", "aerials_att_p90", "aerial_win_rate",
]

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


# 1. Load 2024-25 data ONLY for fitting style space & role demand profiles
ps = pd.read_parquet(PROCESSED / "player_style_features.parquet")
cs = pd.read_parquet(PROCESSED / "club_style_features.parquet")
master = pd.read_parquet(PROCESSED / "master_transfer_dataset.parquet")

ps24 = ps[
    (ps["season"] == 2425)
    & (ps["minutes_90"] >= 450)
    & (ps["position"].isin(["Attack", "Midfield", "Defender"]))
].copy()

# Impute rare NaNs (e.g. avg_shot_dist_m for 0-shot defenders) using 2024-25 position median
for col in PLAYER_STYLE_COLS:
    pos_med = ps24.groupby("position")[col].transform("median")
    ps24[col] = ps24[col].fillna(pos_med).fillna(ps24[col].median())

p_scaler = StandardScaler()
X_p24 = p_scaler.fit_transform(ps24[PLAYER_STYLE_COLS].values)

# PCA + UMAP + KMeans role archetypes on 2024-25 outfield pool
pca = PCA(n_components=6, random_state=42)
Z_pca = pca.fit_transform(X_p24)
for i in range(6):
    ps24[f"pca_{i+1}"] = np.round(Z_pca[:, i], 4)

try:
    import umap
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=2, random_state=42)
    Z_umap = reducer.fit_transform(X_p24)
    ps24["umap_1"] = np.round(Z_umap[:, 0], 4)
    ps24["umap_2"] = np.round(Z_umap[:, 1], 4)
    umap_ok = True
except Exception as e:
    ps24["umap_1"] = ps24["pca_1"]
    ps24["umap_2"] = ps24["pca_2"]
    umap_ok = False
    print(f"[warn] UMAP fallback to PCA: {e}")

km = KMeans(n_clusters=8, n_init=20, random_state=42)
ps24["style_cluster"] = km.fit_predict(Z_pca[:, :5])
ps24.to_parquet(PROCESSED / "player_style_space_2425.parquet", index=False)

print("=== 1. 2024-25 OUTFIELD STYLE SPACE ===")
print(f"Players in 2024-25 style space (>=450 min90): {len(ps24)} | UMAP active: {umap_ok}")
print("PCA explained variance ratio (PC1..PC6):", [round(float(x), 3) for x in pca.explained_variance_ratio_])
print("\nCluster x Position breakdown:")
print(pd.crosstab(ps24["style_cluster"], ps24["position"]).to_string())

# 2. Build 2024-25 Club-Position Demand Profiles (standardized space)
ps24_scaled = pd.DataFrame(X_p24, columns=PLAYER_STYLE_COLS, index=ps24.index)
ps24_scaled["team"] = ps24["team"]
ps24_scaled["position"] = ps24["position"]
ps24_scaled["sub_position"] = ps24["sub_position"]
ps24_scaled["minutes_90"] = ps24["minutes_90"]

def weighted_profile(grp: pd.DataFrame) -> np.ndarray:
    w = grp["minutes_90"].values
    return np.average(grp[PLAYER_STYLE_COLS].values, axis=0, weights=w)

broad_profiles = {}
broad_rows = []
for (team, pos), grp in ps24_scaled.groupby(["team", "position"]):
    vec = weighted_profile(grp)
    broad_profiles[(team, pos)] = vec
    broad_rows.append({"team": team, "position": pos, "n_players": len(grp), **{c: round(float(v), 4) for c, v in zip(PLAYER_STYLE_COLS, vec)}})

sub_profiles = {}
for (team, pos, subp), grp in ps24_scaled.dropna(subset=["sub_position"]).groupby(["team", "position", "sub_position"]):
    if len(grp) >= 2 and grp["minutes_90"].sum() >= 1350:
        sub_profiles[(team, pos, subp)] = weighted_profile(grp)

pd.DataFrame(broad_rows).to_parquet(PROCESSED / "club_position_profiles_2425.parquet", index=False)

# Fit 2024-25 Club Tactical Scaler
cs24 = cs[cs["season"] == 2425].copy()
c_scaler = StandardScaler()
X_c24 = c_scaler.fit_transform(cs24[CLUB_STYLE_COLS].values)
club_vec_24 = {t: X_c24[i] for i, t in enumerate(cs24["team"])}

# 3. Score Complete Eligible Outfield Transfers (compute_fit v1)
elig_out = master[
    (master["elig_status"] == "eligible")
    & (master["position"].isin(["Attack", "Midfield", "Defender"]))
    & (master["dest_c24_poss_share"].notna())
].copy()

p24_feature_names = [f"p24_{c}" for c in PLAYER_STYLE_COLS]
X_transfer_p = p_scaler.transform(elig_out[p24_feature_names].values)

role_cos_broad = []
role_cos_sub = []
role_dist_broad = []
env_cos_list = []

for idx, (_, r) in enumerate(elig_out.iterrows()):
    p_vec = X_transfer_p[idx]
    dest = r["ws_to_club"]
    orig = r["ws_from_club"]
    pos = r["position"]
    subp = r["sub_position"]

    c_broad = broad_profiles.get((dest, pos), np.zeros_like(p_vec))
    c_sub = sub_profiles.get((dest, pos, subp), c_broad)

    role_cos_broad.append(cosine_sim(p_vec, c_broad))
    role_cos_sub.append(cosine_sim(p_vec, c_sub))
    role_dist_broad.append(float(np.linalg.norm(p_vec - c_broad)))

    if orig in club_vec_24 and dest in club_vec_24:
        env_cos_list.append(cosine_sim(club_vec_24[orig], club_vec_24[dest]))
    else:
        env_cos_list.append(0.0)

elig_out["role_cos_broad"] = np.round(role_cos_broad, 4)
elig_out["role_cos_sub"] = np.round(role_cos_sub, 4)
elig_out["role_dist_broad"] = np.round(role_dist_broad, 4)
elig_out["env_cos"] = np.round(env_cos_list, 4)

# Blended fit_score_v1 (0-100 scale: 75% club-position role fit + 25% tactical continuity)
raw_blend = 0.75 * elig_out["role_cos_sub"] + 0.25 * elig_out["env_cos"]
elig_out["fit_score_v1"] = np.round(50.0 + 50.0 * raw_blend, 1)

# Club-strength gap baseline (dest minus origin 2024-25 field_tilt)
elig_out["club_strength_gap"] = (elig_out["dest_c24_field_tilt"] - elig_out["orig_c24_field_tilt"]).fillna(0.0)

elig_out.to_parquet(PROCESSED / "backtest_results_v1.parquet", index=False)
elig_out.to_csv(PROCESSED / "backtest_results_v1.csv", index=False)

# 4. Backtest Evaluation (Spearman vs Y_score & MS, Bootstrap AUC, Baselines)
print(f"\n=== 2. BACKTEST v1: CONTINUOUS SPEARMAN CORRELATIONS (n={len(elig_out)}) ===")
y_valid = elig_out.dropna(subset=["Y_score"])
for pred in ["fit_score_v1", "role_cos_sub", "role_cos_broad", "env_cos", "expected_ga_p90", "age_years", "club_strength_gap"]:
    r_y, p_y = spearmanr(y_valid[pred], y_valid["Y_score"])
    r_ms, p_ms = spearmanr(elig_out[pred], elig_out["MS"])
    print(f"  {pred:18s} | vs Y_score (n={len(y_valid)}): r={r_y:+.3f} (p={p_y:.3f}) | vs MS (n={len(elig_out)}): r={r_ms:+.3f} (p={p_ms:.3f})")

print("\n=== 3. SPEARMAN BY POSITION GROUP ===")
for pos in ["Attack", "Midfield", "Defender"]:
    sub_all = elig_out[elig_out["position"] == pos]
    sub_y = sub_all.dropna(subset=["Y_score"])
    r_y, p_y = spearmanr(sub_y["fit_score_v1"], sub_y["Y_score"])
    r_ms, p_ms = spearmanr(sub_all["fit_score_v1"], sub_all["MS"])
    print(f"  {pos:8s} (n={len(sub_all):2d}) | fit_v1 vs Y: r={r_y:+.3f} (p={p_y:.3f}) | fit_v1 vs MS: r={r_ms:+.3f} (p={p_ms:.3f})")

print("\n=== 4. BINARY CLASSIFICATION ROC-AUC (SUCCESS vs FAILURE, 1000 BOOTSTRAPS) ===")
for lbl_col in ["label_lenient", "label_base", "label_strict"]:
    bsub = elig_out[elig_out[lbl_col].isin(["Success", "Failure"])].copy()
    yt = (bsub[lbl_col] == "Success").astype(int).values
    auc_fit, lo, hi = bootstrap_auc(yt, bsub["fit_score_v1"].values)
    auc_exp = float(roc_auc_score(yt, bsub["expected_ga_p90"].values))
    auc_age = float(roc_auc_score(yt, -bsub["age_years"].values))
    auc_gap = float(roc_auc_score(yt, -bsub["club_strength_gap"].values))
    print(
        f"  {lbl_col:13s} (n={len(bsub):2d}, pos={yt.sum():2d}/neg={(1-yt).sum():2d}) | "
        f"fit_v1 AUC={auc_fit:.3f} [{lo:.3f}, {hi:.3f}] | "
        f"exp_ga={auc_exp:.3f} | younger={auc_age:.3f} | small_gap={auc_gap:.3f}"
    )

print("\n=== 5. TOP 10 & BOTTOM 10 COHORT TRANSFERS BY fit_score_v1 ===")
show = ["player_name", "position", "tm_from_club", "tm_to_club", "fit_score_v1", "role_cos_sub", "env_cos", "MS", "Y_score", "label_base"]
print("--- Highest Predicted Fit ---")
print(elig_out.sort_values("fit_score_v1", ascending=False)[show].head(10).to_string(index=False))
print("\n--- Lowest Predicted Fit ---")
print(elig_out.sort_values("fit_score_v1", ascending=True)[show].head(10).to_string(index=False))