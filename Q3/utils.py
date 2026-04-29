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
 
from sklearn.model_selection import StratifiedKFold, cross_validate
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
        path = p/f"{name}.csv"
        df = pd.read_csv(path, low_memory = False)
        tables[name] = df

    # Load event tables
    event_dir = p/"events"
    event_tables = {}
    required = ["0", "c", "g", "m", "q", "s"]


    for name in required:
        path = event_dir/f"{name}.csv"
        df = pd.read_csv(path, low_memory = False)
        event_tables[name] = df

    return tables, event_tables

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

    # 3. Main Aggregation
    res = df.groupby("user_id").agg(
        q__n_questions_viewed=("ts", "count"),
        q__n_unique_urls=("url", "nunique") if "url" in df.columns else ("ts", lambda _: np.nan),
        q__n_unique_courses=("course_id", "nunique"),
        q__avg_question_number=("q_num", "mean"),
        q__n_active_days=("ts", lambda x: pd.to_datetime(x, unit="s").dt.date.nunique()),
        # Revisit rate logic: % of unique questions that were seen > once
        q__revisit_rate=("is_revisit", "mean")
    )

    return res.reset_index()

def extract_event_features_(event_tables, tables, early_ts):
    df_pageview = tables['pageviews'].copy()
    course_ids  = tables['course_ids']

    def merge_pageview(event_df):
        return event_df.merge(df_pageview, left_on="pageview_id", right_on="id")

    def apply_window(df, ts_col="timestamp"):
        # Event timestamps are in ms → convert to seconds for comparison
        # early_ts values are already in Unix seconds
        event_ts_s = pd.to_numeric(df[ts_col], errors="coerce") / 1000
        cutoff_s   = df["user_id"].map(early_ts)
        return df[event_ts_s <= cutoff_s].copy()

    df_clicks    = apply_window(merge_pageview(event_tables['c']))
    df_heartbeat = apply_window(merge_pageview(event_tables['0']))
    df_media     = apply_window(merge_pageview(event_tables['m']))
    df_questions = apply_window(merge_pageview(event_tables['q']))

    feat_tables = [
        _features_clicks(df_clicks),
        _features_heartbeat(df_heartbeat),
        _features_questions(df_questions, course_ids),
        _features_media(df_media),
    ]

    features = feat_tables[0]
    for ft in feat_tables[1:]:
        print(f"Length before merge: {len(features)}")
        features = features.merge(ft, on="user_id", how="outer")
        print(f"Length after merge : {len(features)}")

    return features.reset_index()


### Plot functions 
def plot_feature_importance(results, X, outpath="feature_importance.png", top_n=15):

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=C["bg"])

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
        axes[1, 1].set_title("XGBoost ",
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