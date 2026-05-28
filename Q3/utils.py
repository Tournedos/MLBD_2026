import os, json, ast, warnings, argparse, textwrap
from pathlib import Path
from datetime import datetime
import math 

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
 
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
 
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

FEATURE_GROUPS = {
    "Session Consistency": [
        "n_active_days_early", "engagement_span_days",
        "avg_daily_pageviews_early",
    ],
    "Exercise Diversity": [
        "n_exercises_early", "n_exercise_types_early",
        "n_unique_courses_early",
    ],
    "Content Focus": [
        "content_focus_score",
        "avg_score_early", "score_variance_early",
    ],
    "Help-seeking": [
        "hint_rate_early", "n_chats_early",
        "quiz_correct_rate_early",
    ],
}
 
FEATURE_NICE = {
    "n_active_days_early":          "Active Days",
    "engagement_span_days":         "Engagement Span (days)",
    "avg_daily_pageviews_early":    "Avg Daily Page-views",
    "n_exercises_early":            "# Exercises Attempted",
    "n_exercise_types_early":       "Exercise Type Diversity",
    "n_unique_courses_early":       "# Courses Visited",
    "content_focus_score":          "Content Focus Score",
    "avg_score_early":              "Avg Score (early)",
    "score_variance_early":         "Score Variance",
    "hint_rate_early":              "Hint Rate",
    "n_chats_early":                "AI Chatbot Uses",
    "quiz_correct_rate_early":      "Quiz Correct Rate",
    "n_pageviews_early":            "# Page-views",
    "n_unique_pages_early":         "# Unique Pages",
    "time_per_question_early":      "Avg Time / Question (s)",
    "avg_user_msgs_per_chat_early": "Avg User Msgs / Chat",
}


C = {
    "primary":   "#4A90D9",
    "secondary": "#E07B54",
    "success":   "#5CB85C",
    "danger":    "#D9534F",
    "neutral":   "#7F8C8D",
    "bg":        "#F8F9FA",
    "dark":      "#2C3E50",
}
 
COURSE_LABELS = {
    42:   "LZG Math", 3865: "KZG Math",
    5447: "LZG Essay", 3301: "KZG Essay",
    2115: "LZG Text",  5009: "KZG Text",
}


### Loading the data, windowing, data format functions
# Load Data 
def load_data(data_dir):
    required = ["students", "pageviews",
        "math_results", "quiz_results",
        "text_results", "essay_results", "gymitrainer", "comments",
        "course_ids", "math_questions", "text_questions", "quiz_questions"]
    p = Path(data_dir)
    tables = {}

    for name in required:
        print(f"Reading Table {name}")
        path = p/f"{name}.csv"
        df = pd.read_csv(path, low_memory = False)
        tables[name] = df

    return tables

def load_event_data(data_dir):
    p = Path(data_dir)
    tables = {}
    # Load event tables
    event_dir = p/"events"
    event_tables = {}
    required = ["0", "c", "g", "m", "q", "s"]


    for name in required:
        print(f"Reading Table {name}")
        path = event_dir/f"{name}.csv"
        df = pd.read_csv(path, low_memory = False)
        event_tables[name] = df
        del(df)

    return event_tables

def _to_unix(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    mask = out.isna()
    if mask.any():
        parsed = pd.to_datetime(series[mask], errors="coerce", utc=True)
        out[mask] = parsed.astype("int64") // 1_000_000_000
    # If timestamps look like milliseconds (> year 2100 in seconds)
    ms_mask = out > 4_102_444_800
    out[ms_mask] = out[ms_mask] / 1000
    return out

def _ms_to_s(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    ms_mask = s > 4_102_444_800
    s[ms_mask] = s[ms_mask] / 1_000
    return s
### Feature Engineering 
def describe_platform_engagement(df_pageview, df_students, early_fraction=1/3):
    pv = df_pageview.copy()
    pv["ts"] = _to_unix(pv["created_at"])
    pv = pv.dropna(subset=["ts", "user_id"])
    pv["date"] = pd.to_datetime(pv["ts"], unit="s", utc=True).dt.date

    # Join with student registration time
    students = df_students[["user_id", "creation_time"]].drop_duplicates()
    students["reg_ts"] = _to_unix(students["creation_time"])

    per_user = pv.groupby("user_id").agg(
        n_active_days  = ("date",  "nunique"),
        first_activity = ("ts",    "min"),
        last_activity  = ("ts",    "max"),
    ).reset_index()

    per_user = per_user.merge(students, on="user_id", how="left")
    per_user["span_days"] = (per_user["last_activity"] - per_user["first_activity"]) / 86400
    per_user["days_since_reg"] = (per_user["last_activity"] - per_user["reg_ts"]) / 86400

    for col, label in [
        ("n_active_days", "Unique active days"),
        ("span_days",     "Activity span (days)"),
        ("days_since_reg","Days since registration"),
    ]:
        s = per_user[col].dropna()
        print(f"\n{label}")
        print(f"  Mean   : {s.mean():.1f}")
        print(f"  Median : {s.median():.1f}")
        print(f"  p25–p75: {s.quantile(0.25):.1f} – {s.quantile(0.75):.1f}")

    median_span = per_user["span_days"].median()
    suggested_window = max(1, round(median_span * early_fraction))
    print(f"\n{'=' * 45}")
    print(f"Suggested early window ({early_fraction:.0%} of median span): {suggested_window} days")
    print("=" * 45)

    return per_user, median_span

def compute_shannon_entropy(series):
    counts = series.value_counts()
    total = counts.sum()
    if total == 0:
        return 0
    probs = counts / total
    return -sum(p * math.log2(p) for p in probs if p > 0)

def build_outcome(tables: dict, late_ts: dict) -> pd.DataFrame:
    """
    Late-window average normalised score (math + quiz + text + essay).
    Label: 1 = success (≥ median), 0 = struggle (< median).
    """
    all_late_scores = []

    # 1. Collect all valid scores from all tables in one pass
    configs = [("math_results", "timestamp"), ("quiz_results", "time"),
               ("text_results", "timestamp")] #, ("essay_results", "time")]

    for name, ts_col in configs:
        df = tables[name].copy()
        df["ts"] = _to_unix(df[ts_col])
        df["cutoff"] = df["user_id"].map(late_ts)
        #print(df.columns)
        # Keep only the 'late' data
        late_df = df[df["ts"] > df["cutoff"]]#.dropna(subset=["points", "max_points"])
        
        if not late_df.empty:
            late_df["score"] = pd.to_numeric(late_df["points"]) / pd.to_numeric(late_df["max_points"]).replace(0, np.nan)
            all_late_scores.append(late_df[["user_id", "score"]])

    if not all_late_scores: return pd.DataFrame()

    # 2. Average the scores per user
    out = pd.concat(all_late_scores).groupby("user_id")["score"].mean().reset_index()
    out.columns = ["user_id", "late_avg_score"]

    # 3. Create the binary label (1 or 0)
    median = out["late_avg_score"].median()
    out["label"] = (out["late_avg_score"] >= median).astype(int)

    print(f"\n Outcome median score: {median:.3f}")
    return out
 
### Events Feature Engineering
def _features_clicks(df: pd.DataFrame):
    df = df.copy()
    df["ts"] = _ms_to_s(df["timestamp"])

    df["date"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    #df["scrollY"] = pd.to_numeric(df["scrollY"], errors="coerce")
    #df["scrollX"] = pd.to_numeric(df["scrollY"], errors="coerce")

    # Compute coefficient of variation, compute dispersion of the data ? 
    def calc_cv(x):
        gaps = x.sort_values().diff().dropna()
        gaps = gaps[gaps>0]
        return gaps.std()/gaps.mean() if gaps.mean() > 0 else np.nan  
        
    results = df.groupby("user_id").agg(
        clicks__n_total = ("ts", "nunique"),
        clicks__avg_scrollY = ("scrollY", "mean"),
        clicks__session_gap_cv=("ts", calc_cv),
        clicks__scroll_depth_max=("scrollY", lambda x: x.quantile(0.95))
    )

    return results.reset_index()

def _features_heartbeat(df) : 
    '''
    Features we want to extract : 
    - Mean idle time per user 
    - Distance between the last postition and the current one 
    - Scrolling depth
    '''
    df = df.copy()
    df['ts'] = pd.to_datetime(_ms_to_s(df['timestamp']), unit = "s", errors="coerce")
    df['lastActivity_ts'] = pd.to_datetime(_ms_to_s(df['lastActivity']), unit = "s", errors="coerce")
    df['idle_time'] = df["ts"] - df["lastActivity_ts"]

    results = df.groupby("user_id").agg(
        heartbeat__n_total = ("ts", "nunique"),
        heartbeat__avg_idle_time = ("idle_time", "mean"),
        heartbeat__scroll_depth_max = ("scrollY", lambda x: x.quantile(0.95)),
        heartbeat__avg_scrollY=("scrollY", "mean"),
     #  heartbeat__session_gap_cv = ("ts", calc_cv)
    )
    return results.reset_index()

def _features_media(df) :
    """
    From events/m.csv derive per-student:
 
    media__n_play_events        — total play events (media engagement count)
    media__n_unique_media       — distinct media items played
    media__total_watch_min      — estimated total watch/listen time in minutes
                                  (sum of completed intervals between play→pause/end)
    media__completion_rate      — fraction of media items where an 'ended'
                                  event was recorded  (finishes what they start)
    media__n_unique_urls        — distinct pages with media
    media__media_type_diversity — number of distinct mediaType values
                                  (audio / vimeo / youtube breadth)
    media__audio_pct            — fraction of play events that are audio
                                  (separates audio lessons from video)
    """
    df = df.copy()
    df["ts"]           = _ms_to_s(df["timestamp"])
    df["currentTime"]  = pd.to_numeric(df["currentTime"], errors="coerce")
    df["duration"]     = pd.to_numeric(df["duration"], errors="coerce")
    df["eventType"]    = df.get("eventType",  "unknown").fillna("unknown").str.lower()
    df["mediaType"]    = df.get("mediaType",  "unknown").fillna("unknown").str.lower()
    df["mediaId"]      = df.get("mediaId",    "").astype(str)

    df["is_play"] = df["eventType"] == "play"
    df["is_ended"] = df["eventType"] == "ended"

    # Compute completion stats
    completion_stats = df[df["is_ended"]].groupby("user_id")["mediaId"].nunique()

    results = df.groupby("user_id").agg(
        media__n_play_events=("is_play", "sum"),
        media__n_unique_media=("mediaType", "nunique"),
        #media__dominant_media = ("mediaType", "mode"),
        media__total_watch_min=("currentTime", lambda x: x[df.loc[x.index, "is_ended"]].sum() / 60),
        media__n_unique_urls=("url", "nunique") if "url" in df.columns else ("ts", lambda _: np.nan),
        media__media_type_diversity=("mediaType", "nunique"),
        # Audio %: average of is_audio only where event is 'play'
        media__audio_pct=("mediaType", lambda x: (x[df.loc[x.index, "is_play"]] == "audio").mean())
    )

    results["media__completion_rate"] = completion_stats / results["media__n_unique_media"]
    return results.reset_index().fillna(0)

def _features_questions(df, course_ids):
    df = df.copy()
    df["ts"] = _ms_to_s(df["timestamp"])
    df["q_num"] = pd.to_numeric(df.get("questionNumber"), errors="coerce")

    if course_ids is not None:
        url2c = course_ids.set_index("url")["course_id"].to_dict()
        df["course_id"] = df["url"].map(url2c)

    # 2. Calculate "Revisits" upfront
    # We flag rows that are part of a duplicate (user_id, url, q_num) set
    df["is_revisit"] = df.duplicated(subset=["user_id", "url", "q_num"], keep=False)

    # 0 if majority of course_id are in kurzeit_id, else 1
    
    def _maj_track_binary(x):
        kurzeit_ids = [3865, 3301, 5009, 8117]
        langzeit_ids = [42, 5447, 2115, 8117]
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        in_set_ratio = x.isin(kurzeit_ids).mean()
        return 0 if in_set_ratio > 0.5 else 1

    # 3. Main Aggregation
    res = df.groupby("user_id").agg(
        q__n_questions_viewed=("ts", "count"),
        q__n_unique_urls=("url", "nunique") if "url" in df.columns else ("ts", lambda _: np.nan),
        q__maj_track=("course_id", _maj_track_binary),
        q__n_unique_courses=("course_id", "nunique"),
        q__avg_question_number=("q_num", "mean"),
        q__n_active_days=("ts", lambda x: pd.to_datetime(x, unit="s").dt.date.nunique()),
        # Revisit rate logic: % of unique questions that were seen > once
        q__revisit_rate=("is_revisit", "mean")
    )

    return res.reset_index()

def extract_event_features_(data_dir, tables, early_ts):
    """
    Reads each event CSV one at a time: load → filter → extract features → delete.
    Only one large event table lives in memory at once.
    """
    EVENT_COLS = {
        "0": ["pageview_id", "timestamp", "lastActivity", "scrollY"],
        "c": ["pageview_id", "timestamp", "scrollY"],
        "m": ["pageview_id", "timestamp", "currentTime", "duration",
              "eventType", "mediaType", "mediaId"],
        "q": ["pageview_id", "timestamp", "questionNumber"],
    }

    pv         = tables['pageviews'][["id", "user_id", "url"]]
    pid_to_uid = pv.set_index("id")["user_id"].to_dict()
    pid_to_url = pv.set_index("id")["url"].to_dict()
    course_ids = tables['course_ids']
    event_dir  = Path(data_dir) / "events"

    def read_and_filter(name, chunk_size=500_000):
        path      = event_dir / f"{name}.csv"
        wanted    = EVENT_COLS.get(name, [])
        available = pd.read_csv(path, nrows=0).columns.tolist()
        usecols   = [c for c in wanted if c in available] or None
        print(f"  Loading event/{name}.csv  cols={usecols}")
        raw = pd.read_csv(path, usecols=usecols)#, low_memory=False)

        kept = []
        for start in range(0, len(raw), chunk_size):
            chunk = raw.iloc[start : start + chunk_size]
            uid   = chunk["pageview_id"].map(pid_to_uid)
            mask  = uid.notna()
            if not mask.any():
                continue
            filt = chunk.loc[mask].copy()
            filt["user_id"] = uid[mask].astype(int)
            if "url" not in filt.columns:
                filt["url"] = filt["pageview_id"].map(pid_to_url)
            ts_s   = pd.to_numeric(filt["timestamp"], errors="coerce") / 1000
            cutoff = filt["user_id"].map(early_ts)
            kept.append(filt.loc[ts_s <= cutoff])
        del raw  # free the full CSV before extracting features
        return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()

    feat_clicks    = _features_clicks(read_and_filter("c"))
    feat_heartbeat = _features_heartbeat(read_and_filter("0"))
    feat_questions = _features_questions(read_and_filter("q"), course_ids)
    feat_media     = _features_media(read_and_filter("m"))

    features = feat_clicks
    for ft in [feat_heartbeat, feat_questions, feat_media]:
        features = features.merge(ft, on="user_id", how="outer")

    return features.reset_index()


### Plot functions 
def plot_feature_importance(results, X, outpath="feature_importance.png", top_n=15):

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=C["bg"])
    '''
    # ── Random Forest ──────────────────────────────────────────────────────
    rf_importances = np.array([
        est.named_steps["clf"].feature_importances_
        for est in results["Random Forest"]["estimators"]
    ])
    rf_mean = pd.Series(rf_importances.mean(axis=0), index=X.columns)
    rf_std  = pd.Series(rf_importances.std(axis=0),  index=X.columns)
    rf_mean = rf_mean.sort_values(ascending=True).tail(top_n)
    rf_std  = rf_std[rf_mean.index]

    axes[0, 0].barh(rf_mean.index, rf_mean.values, xerr=rf_std.values,
                    color=C["primary"], ecolor="gray", capsize=3)
    axes[0, 0].set_title("Random Forest",
                          color=C["dark"])
    axes[0, 0].set_xlabel("Importance")
    '''
    # ── Logistic Regression ────────────────────────────────────────────────
    lr_coefs = np.array([
        est.named_steps["clf"].coef_[0]
        for est in results["Logistic Regression"]["estimators"]
    ])
    lr_mean = pd.Series(lr_coefs.mean(axis=0), index=X.columns)
    lr_top  = lr_mean.abs().sort_values(ascending=True).tail(top_n)
    colors  = [C["danger"] if lr_mean[f] < 0 else C["success"] for f in lr_top.index]

    axes[0, 1].barh(lr_top.index, lr_mean[lr_top.index].values, color=colors)
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Logistic Regression — Coefficients (mean, 5 folds)\n"
                          "green = success, red = struggle", color=C["dark"])
    axes[0, 1].set_xlabel("Coefficient value")
    
    # ── Gradient Boosting ──────────────────────────────────────────────────
    gb_importances = np.array([
        est.named_steps["clf"].feature_importances_
        for est in results["Gradient Boosting"]["estimators"]
    ])
    gb_mean = pd.Series(gb_importances.mean(axis=0), index=X.columns)
    gb_std  = pd.Series(gb_importances.std(axis=0),  index=X.columns)
    gb_mean = gb_mean.sort_values(ascending=True).tail(top_n)
    gb_std  = gb_std[gb_mean.index]

    axes[1, 0].barh(gb_mean.index, gb_mean.values, xerr=gb_std.values,
                    color=C["primary"], ecolor="gray", capsize=3)
    axes[1, 0].set_title("Gradient Boosting ",
                          color=C["dark"])
    axes[1, 0].set_xlabel("Importance")

    # ── XGBoost (only if available) ────────────────────────────────────────
    if HAS_XGB and "XGBoost" in results:
        xgb_importances = np.array([
            est.named_steps["clf"].feature_importances_
            for est in results["XGBoost"]["estimators"]
        ])
        xgb_mean = pd.Series(xgb_importances.mean(axis=0), index=X.columns)
        xgb_std  = pd.Series(xgb_importances.std(axis=0),  index=X.columns)
        xgb_mean = xgb_mean.sort_values(ascending=True).tail(top_n)
        xgb_std  = xgb_std[xgb_mean.index]

        axes[1, 1].barh(xgb_mean.index, xgb_mean.values, xerr=xgb_std.values,
                        color=C["primary"], ecolor="gray", capsize=3)
        axes[1, 1].set_title("XGBoost",
                              color=C["dark"])
        axes[1, 1].set_xlabel("Importance")
    else:
        axes[1, 1].set_visible(False)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        ax.set_facecolor(C["bg"])

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.show()

### Models functions
def build_models() -> dict:
    steps_base = [("impute", SimpleImputer(strategy="median")),
                  ("scale", StandardScaler())]
    models = {
        "Logistic Regression": Pipeline(steps_base + [
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=1))
        ]),
        "Gradient Boosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42))
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    scale_pos_weight=1, random_state=42, verbosity=0))
        ])
    return models

def evaluate_models(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> dict:
    def _plot_model_comparison(results: dict, out_dir: Path):
        metrics = ["roc_auc", "f1", "bal_acc"]
        metric_labels = ["ROC-AUC", "F1 Score", "Balanced Accuracy"]
        names = list(results.keys())
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=C["bg"])
        fig.suptitle("Model Performance Comparison (5-Fold CV)",
                    fontsize=14, fontweight="bold", color=C["dark"])
        colours = [C["primary"], C["secondary"], C["success"], C["neutral"]]
        for ax, metric, label in zip(axes, metrics, metric_labels):
            vals   = [results[n][metric] for n in names]
            stds   = [results[n].get(metric + "_std", 0) for n in names]
            bars   = ax.barh(names, vals, xerr=stds, color=colours[:len(names)],
                            edgecolor="white", linewidth=0.5, height=0.55,
                            error_kw=dict(ecolor="gray", capsize=4))
            ax.set_xlim(0.4, 1.0)
            ax.set_title(label, fontsize=12, color=C["dark"])
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8,
                    label="Baseline (0.5)")
            for bar, val in zip(bars, vals):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
            ax.tick_params(labelsize=9)
        plt.tight_layout()
        fig.savefig("03_plot_comparisons.png", dpi=150,
                   bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)
        plt.show()
 
    def _plot_roc_curves(results: dict, X: pd.DataFrame, y: pd.Series,
                        cv: StratifiedKFold, out_dir: Path):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=C["bg"])
        colours = [C["primary"], C["secondary"], C["success"], C["neutral"]]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.50)")
        models = build_models()
    
        for (name, res), colour in zip(results.items(), colours):
            # use the best fold estimator
            aucs = res["cv_results"]["test_roc_auc"]
            best_idx = int(np.argmax(aucs))
            best_est = res["estimators"][best_idx]
            splits   = list(cv.split(X, y))
            _, test_idx = splits[best_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            y_prob = best_est.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(
                y_test, y_prob, name=f"{name} (AUC={aucs[best_idx]:.3f})",
                color=colour, ax=ax, lw=2)
    
        ax.set_title("ROC Curves — Best Fold per Model",
                    fontsize=13, fontweight="bold", color=C["dark"])
        ax.legend(fontsize=9, loc="lower right")
        ax.set_facecolor(C["bg"])
        plt.tight_layout()
        fig.savefig("04_roc_curves.png", dpi=150,
                   bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)
        plt.show()
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "f1", "accuracy", "balanced_accuracy"]
    models  = build_models()
    results = {}
 
    print("\n── Cross-validated performance ───────────────────────────")
    print(f"  {'Model':<25} {'ROC-AUC':>9} {'F1':>9} {'Balanced Acc':>13}")
    print("  " + "-"*60)
 
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                return_estimator=True, n_jobs=1)
        results[name] = {
            "cv_results": cv_res,
            "roc_auc":    cv_res["test_roc_auc"].mean(),
            "roc_auc_std":cv_res["test_roc_auc"].std(),
            "f1":         cv_res["test_f1"].mean(),
            "bal_acc":    cv_res["test_balanced_accuracy"].mean(),
            "estimators": cv_res["estimator"],
        }
        print(f"  {name:<25} {results[name]['roc_auc']:.3f}±{results[name]['roc_auc_std']:.3f}"
              f"  {results[name]['f1']:.3f}  {results[name]['bal_acc']:.3f}")
 
    # ── plot comparison ────────────────────────────────────────────────
    _plot_model_comparison(results, out_dir)
 
    # ── ROC curves (best fold per model) ──────────────────────────────
    _plot_roc_curves(results, X, y, cv, out_dir)
 
    return results



def build_time_series_features(data_dir, tables, early_ts, n_bins=3):
    """
    Per-user temporal bin features over the 41-day early window.
    Bins question events and pageviews into n_bins equal slices;
    adds quiz score trajectory (first half vs second half of window).
    Returns a DataFrame with ts__* columns, one row per user.
    """
    WINDOW_S  = 41 * 86400 # 41 days window to seconds
    bin_width = WINDOW_S / n_bins
    valid_ts  = {u: t for u, t in early_ts.items() if pd.notna(t)}

    # ── 1. Question-event bins (events/q.csv — timestamps in ms) ──────────
    pv_map = tables["pageviews"][["id", "user_id"]].set_index("id")["user_id"].to_dict()
    q_path = Path(data_dir) / "events" / "q.csv"
    q_raw  = pd.read_csv(q_path, usecols=["pageview_id", "timestamp"])
    q_raw["user_id"] = q_raw["pageview_id"].map(pv_map)
    q_raw  = q_raw.dropna(subset=["user_id"])
    q_raw["user_id"] = q_raw["user_id"].astype(int)
    q_raw["ts"]      = pd.to_numeric(q_raw["timestamp"], errors="coerce") / 1000
    q_raw["cutoff"]  = q_raw["user_id"].map(valid_ts)
    q_raw["start"]   = q_raw["cutoff"] - WINDOW_S
    q_raw = q_raw.dropna(subset=["cutoff"])
    q_raw = q_raw[(q_raw["ts"] >= q_raw["start"]) & (q_raw["ts"] <= q_raw["cutoff"])]
    q_raw["bin"] = ((q_raw["ts"] - q_raw["start"]) / bin_width).clip(0, n_bins - 1e-9).astype(int)

    q_bins = (q_raw.groupby(["user_id", "bin"]).size()
                   .unstack(fill_value=0)
                   .reindex(columns=range(n_bins), fill_value=0))
    q_bins.columns = [f"ts__q_bin{b+1}" for b in range(n_bins)]
    q_bins["ts__q_trend"]   = q_bins[f"ts__q_bin{n_bins}"] - q_bins["ts__q_bin1"]
    total_q = q_bins[[f"ts__q_bin{b+1}" for b in range(n_bins)]].sum(axis=1).replace(0, np.nan)
    q_bins["ts__q_recency"] = q_bins[f"ts__q_bin{n_bins}"] / total_q

    # ── 2. Pageview bins (created_at may be datetime or string) ───────────
    pv = tables["pageviews"][["user_id", "created_at"]].copy()
    if pd.api.types.is_datetime64_any_dtype(pv["created_at"]):
        pv["ts"] = pv["created_at"].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    else:
        pv["ts"] = _to_unix(pv["created_at"])
    pv["cutoff"] = pv["user_id"].map(valid_ts)
    pv["start"]  = pv["cutoff"] - WINDOW_S
    pv = pv.dropna(subset=["cutoff"])
    pv = pv[(pv["ts"] >= pv["start"]) & (pv["ts"] <= pv["cutoff"])]
    pv["bin"] = ((pv["ts"] - pv["start"]) / bin_width).clip(0, n_bins - 1e-9).astype(int)

    pv_bins = (pv.groupby(["user_id", "bin"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=range(n_bins), fill_value=0))
    pv_bins.columns = [f"ts__pv_bin{b+1}" for b in range(n_bins)]
    pv_bins["ts__pv_trend"]   = pv_bins[f"ts__pv_bin{n_bins}"] - pv_bins["ts__pv_bin1"]
    total_pv = pv_bins[[f"ts__pv_bin{b+1}" for b in range(n_bins)]].sum(axis=1).replace(0, np.nan)
    pv_bins["ts__pv_recency"] = pv_bins[f"ts__pv_bin{n_bins}"] / total_pv

    # ── 3. Quiz score trajectory ───────────────────────────────────────────
    quiz = tables["quiz_results"].copy()
    if pd.api.types.is_datetime64_any_dtype(quiz["time"]):
        quiz["ts"] = quiz["time"].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    else:
        quiz["ts"] = _to_unix(quiz["time"])
    quiz["cutoff"] = quiz["user_id"].map(valid_ts)
    quiz["start"]  = quiz["cutoff"] - WINDOW_S
    quiz = quiz.dropna(subset=["cutoff"])
    quiz = quiz[(quiz["ts"] >= quiz["start"]) & (quiz["ts"] <= quiz["cutoff"])]
    quiz["score"] = (pd.to_numeric(quiz["points"], errors="coerce") /
                     pd.to_numeric(quiz["max_points"], errors="coerce").replace(0, np.nan))
    quiz["first_half"] = quiz["ts"] < (quiz["start"] + WINDOW_S / 2)

    s_first  = quiz[quiz["first_half"]].groupby("user_id")["score"].mean().rename("ts__score_first")
    s_last   = quiz[~quiz["first_half"]].groupby("user_id")["score"].mean().rename("ts__score_last")
    score_df = pd.concat([s_first, s_last], axis=1).reset_index()
    score_df["ts__score_delta"] = score_df["ts__score_last"] - score_df["ts__score_first"]

    # ── 4. Merge all ───────────────────────────────────────────────────────
    result = q_bins.reset_index().merge(pv_bins.reset_index(), on="user_id", how="outer")
    result = result.merge(score_df, on="user_id", how="outer")
    print(f"  Time-series features: {result.shape[1]-1} features for {len(result)} users")
    return result


def plot_group_correlations(df, feature_groups, out_path="group_correlations.png"):
    """Correlation heatmap for each feature group, plotted side by side."""
    groups_avail = {
        name: [c for c in cols if c in df.columns]
        for name, cols in feature_groups.items()
        if sum(c in df.columns for c in cols) >= 2
    }
    n   = len(groups_avail)
    fig = plt.figure(figsize=(5 * n, 4.5), facecolor=C["bg"])
    fig.suptitle("Within-group Feature Correlations",
                 fontsize=13, fontweight="bold", color=C["dark"])

    for i, (name, cols) in enumerate(groups_avail.items(), 1):
        ax  = fig.add_subplot(1, n, i)
        sub = df[cols].fillna(0)
        sub = sub.loc[:, sub.std() > 0]  # drop zero-variance columns
        corr   = sub.corr()
        labels = [c.split("__")[-1] for c in sub.columns]
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 7}, linewidths=0.4,
                    cbar_kws={"shrink": 0.7})
        ax.set_title(name, fontsize=10, fontweight="bold", color=C["dark"])
        ax.tick_params(labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.show()

def plot_correlation_heatmap(df_final, group_cols, group_name):
    print(group_name)
    sub = df_final[group_cols]
    sub = sub.loc[:, sub.std()>0]
    corr_matrix = sub.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Heatmap for {group_name}")
    plt.savefig(f"Correlation_Matrices/Correlation matrix for {group_name}")

def evaluate_by_group(df, feature_groups, label_col="label",
                      out_path="group_ablation.png"):
    """
    Ablation study: train RandomForest on each feature group separately,
    then on all groups combined. Uses the same model throughout for fair comparison.
    """
    y  = df[label_col].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _run(cols):
        available = [c for c in cols if c in df.columns]
        if not available:
            return None
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=1)),
        ])
        cv_res = cross_validate(pipe, df[available], y, cv=cv,
                                scoring=["roc_auc", "f1", "balanced_accuracy"],
                                n_jobs=1)
        return {
            "roc_auc":     cv_res["test_roc_auc"].mean(),
            "roc_auc_std": cv_res["test_roc_auc"].std(),
            "f1":          cv_res["test_f1"].mean(),
            "bal_acc":     cv_res["test_balanced_accuracy"].mean(),
            "n_features":  len(available),
        }

    results = {}
    for name, cols in feature_groups.items():
        print(f"  [{name}]...", end=" ", flush=True)
        res = _run(cols)
        if res:
            results[name] = res
            print(f"AUC={res['roc_auc']:.3f}")

    all_cols = [c for cols in feature_groups.values() for c in cols]
    print(f"  [All Combined]...", end=" ", flush=True)
    results["All Combined"] = _run(all_cols)
    print(f"AUC={results['All Combined']['roc_auc']:.3f}")

    # ── Print table ───────────────────────────────────────────────────────
    print("\n── Group Ablation Study ─────────────────────────────────────")
    print(f"  {'Group':<22} {'N':>4}  {'ROC-AUC':>12}  {'F1':>6}  {'Bal Acc':>8}")
    print("  " + "─" * 60)
    for name, res in results.items():
        if res:
            print(f"  {name:<22} {res['n_features']:>4}  "
                  f"{res['roc_auc']:.3f}±{res['roc_auc_std']:.3f}  "
                  f"{res['f1']:.3f}  {res['bal_acc']:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    names  = [n for n, r in results.items() if r]
    aucs   = [results[n]["roc_auc"]     for n in names]
    stds   = [results[n]["roc_auc_std"] for n in names]
    colors = [C["secondary"] if n == "All Combined" else C["primary"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C["bg"])
    bars = ax.barh(names, aucs, xerr=stds, color=colors,
                   ecolor="gray", capsize=4, height=0.55, edgecolor="white")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("ROC-AUC (5-fold CV)")
    ax.set_title("Predictive Power by Feature Group",
                 fontsize=13, fontweight="bold", color=C["dark"])
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_facecolor(C["bg"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.show()

    return results


def train_bidirectional_lstm(features_df: pd.DataFrame,
                             labels_series: pd.Series,
                             num_weeks: int = 10,
                             test_size: float = 0.2,
                             random_state: int = 42,
                             lstm_units: int = 64,
                             dense_units: int = 32,
                             dropout_rate: float = 0.5,
                             learning_rate: float = 1e-3,
                             batch_size: int = 32,
                             epochs: int = 10):

    n_samples, n_features = features_df.shape
    if n_features % num_weeks != 0:
        raise ValueError(f"Expected total features to be divisible by num_weeks={num_weeks}, "
                         f"but got {n_features} total features.")
    n_metrics = n_features // num_weeks

    X = features_df.values.reshape(n_samples, num_weeks, n_metrics)
    y = labels_series.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_flat = X_train.reshape(-1, n_metrics)
    X_test_flat  = X_test.reshape(-1,  n_metrics)

    imputer = SimpleImputer(strategy="median")
    X_train_flat = imputer.fit_transform(X_train_flat)
    X_test_flat  = imputer.transform(X_test_flat)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat  = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(X_train.shape)
    X_test  = X_test_flat.reshape(X_test.shape)

    model = Sequential([
        Bidirectional(
            LSTM(lstm_units, return_sequences=False),
            input_shape=(num_weeks, n_metrics)
        ),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return model