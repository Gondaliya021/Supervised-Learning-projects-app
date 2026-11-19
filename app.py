"""
CSV -> EDA -> Model -> Prediction  (Streamlit + scikit-learn)

You choose only the target column (y). Everything else is automatic:
  * useful feature columns are picked (IDs, constants, mostly-empty and duplicate columns are dropped)
  * each column gets the encoding that fits it (not one-hot for everything)
  * missing values, scaling and the train/test split are handled for you
"""

# adjested r2 is not

import io
import re
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, RobustScaler, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # XGBoost is optional: pip install xgboost
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------------------------------------------------------- settings
MAX_ONEHOT = 10        # categorical columns with <= this many values -> one-hot
MAX_CATEGORIES = 50    # categorical columns with more values than this are dropped
MISSING_DROP = 0.5     # drop columns with more than 50% missing
CORR_DROP = 0.95       # drop numeric columns that duplicate another column
NA_STRINGS = ["", "nan", "NaN", "None", "NA", "N/A", "null", "NULL", "-"]


# ------------------------------------------------------------ data loading
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read this file as a CSV.")


def to_number(s: pd.Series) -> pd.Series:
    """Turn text like '$1,200' or '12%' into numbers; unreadable values and infinity become missing."""
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(r"[,$€£₹%]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, turn 'numbers stored as text' into numbers, make text consistent."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.map(lambda v: str(v) if pd.notna(v) else np.nan)
        elif pd.api.types.is_numeric_dtype(s):
            df[c] = s.replace([np.inf, -np.inf], np.nan)
        elif pd.api.types.is_datetime64_any_dtype(s):
            continue
        else:
            t = s.map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
            t = t.mask(t.isin(NA_STRINGS))
            num = to_number(t)
            if t.notna().sum() > 0 and num.notna().sum() >= 0.95 * t.notna().sum():
                df[c] = num
            else:
                df[c] = t
    return df


@st.cache_data(show_spinner=False)
def load_data(raw: bytes) -> pd.DataFrame:
    return fix_types(read_csv_bytes(raw))


# -------------------------------------------------- automatic column choice
def auto_select_features(df: pd.DataFrame, target: str):
    """Return (kept_columns, {dropped_column: reason})."""
    n = len(df)
    keep, dropped = [], {}
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        nun = s.nunique(dropna=True)
        is_num = pd.api.types.is_numeric_dtype(s)
        if pd.api.types.is_datetime64_any_dtype(s):
            dropped[c] = "date column"
        elif s.isna().mean() > MISSING_DROP:
            dropped[c] = f"{s.isna().mean():.0%} of values missing"
        elif nun <= 1:
            dropped[c] = "same value in every row"
        elif n > 20 and ((is_num and not pd.api.types.is_float_dtype(s) and nun == n) or (not is_num and nun >= 0.95 * n)):
            dropped[c] = "looks like an ID (every value different)"
        elif not is_num and nun > MAX_CATEGORIES:
            dropped[c] = f"too many different values ({nun})"
        else:
            keep.append(c)

    # drop numeric columns that carry the same information as another one
    num = [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) > 1:
        corr = df[num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        for c in upper.columns:
            partners = [p for p in upper.index[upper[c] > CORR_DROP] if p in keep]
            if partners and c in keep:
                dropped[c] = f"duplicates '{partners[0]}' (correlation > {CORR_DROP})"
                keep.remove(c)
    return keep, dropped


# ------------------------------------------------------- encoding choices
ENC_CHOICES = ["Auto", "Numeric", "One-hot", "Binary 0/1", "Label", "Ordered (custom order)",
               "Frequency", "Target (mean of y)", "Drop"]
CHOICE_TO_GROUP = {
    "Numeric": "num", "One-hot": "onehot", "Binary 0/1": "binary", "Label": "ordinal",
    "Ordered (custom order)": "ordered", "Frequency": "freq", "Target (mean of y)": "target",
}
GROUPS = ["num", "binary", "onehot", "ordinal", "ordered", "freq", "target"]
CAT_GROUPS = ["binary", "onehot", "ordinal", "ordered", "freq", "target"]
GROUP_TEXT = {
    "num": "Numeric: fill missing with median, then scale",
    "binary": "2 values: converted to 0 / 1",
    "onehot": "Few categories: one-hot encoding",
    "ordinal": "Many categories: label encoding (one number per category)",
    "ordered": "Ordered categories: numbers follow your custom order",
    "freq": "Frequency encoding (how common each category is)",
    "target": "Target encoding (average of y for each category)",
}


def cat_str(v) -> str:
    """One consistent text form for category values (3.0 and 3 both become '3')."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def auto_group(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "num"
    nun = s.nunique(dropna=True)
    if nun <= 2:
        return "binary"
    return "onehot" if nun <= MAX_ONEHOT else "ordinal"


def check_choice(s: pd.Series, choice: str):
    """Is this encoding allowed for this column? Returns (ok, reason)."""
    nun = s.nunique(dropna=True)
    if choice == "Numeric" and not pd.api.types.is_numeric_dtype(s):
        return False, "is text, so it cannot be used as a number"
    if choice == "Binary 0/1" and nun > 2:
        return False, f"has {nun} different values (binary needs 2)"
    if choice in ("One-hot", "Ordered (custom order)") and nun > MAX_CATEGORIES:
        return False, f"has {nun} different values (this encoding allows at most {MAX_CATEGORIES})"
    return True, ""


def sorted_values(s: pd.Series) -> list:
    vals = {cat_str(v) for v in s.dropna()}
    if pd.api.types.is_numeric_dtype(s):
        return sorted(vals, key=float)
    return sorted(vals)


def make_plan(X: pd.DataFrame, overrides: dict = None, orders: dict = None) -> dict:
    """Choose an encoding for every column. Automatic unless you override it."""
    overrides, orders = overrides or {}, orders or {}
    plan = {g: [] for g in GROUPS}
    plan["orders"], plan["notes"] = {}, []
    for c in X.columns:
        group = auto_group(X[c])
        choice = overrides.get(c, "Auto")
        if choice not in ("Auto", "Drop"):
            ok, why = check_choice(X[c], choice)
            if ok:
                group = CHOICE_TO_GROUP[choice]
            else:
                plan["notes"].append(f"`{c}` {why}. The automatic encoding is used instead.")
        plan[group].append(c)
        if group == "ordered":
            uniq = sorted_values(X[c])
            given = [v for v in orders.get(c, []) if v in uniq]
            plan["orders"][c] = given + [v for v in uniq if v not in given]
    return plan


def plan_table(X: pd.DataFrame, plan: dict) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        for c in plan[g]:
            how = GROUP_TEXT[g]
            if g == "ordered":
                how += f" ({' < '.join(plan['orders'][c][:6])}{' ...' if len(plan['orders'][c]) > 6 else ''})"
            rows.append({"column": c, "unique values": int(X[c].nunique()), "encoding": how})
    return pd.DataFrame(rows)


def align_features(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Make new data have the same column types as the training data."""
    out = df.copy()
    for c in plan["num"]:
        out[c] = to_number(out[c])
    for g in CAT_GROUPS:
        for c in plan[g]:
            out[c] = out[c].map(lambda v: cat_str(v) if pd.notna(v) else np.nan)
    return out


# ------------------------------------------------------------------ models
def guess_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "Regression"
    return "Classification"


def get_models(task: str) -> dict:
    if task == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=2.0),
            "Lasso": Lasso(alpha=0.5, max_iter=2000),
            "Polynomial (degree 2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "KNN": KNeighborsRegressor(n_jobs=-1),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1)
        return models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(random_state=42, n_jobs=-1)
    return models


OUTLIER_OPTIONS = {
    "Keep as is": "none",
    "Cap extreme values (IQR rule)": "cap",
    "Robust scaling (median & IQR)": "robust",
}


class IQRCapper(BaseEstimator, TransformerMixin):
    """Pull extreme values back to the edge of the normal range.
    The range (Q1 - k*IQR ... Q3 + k*IQR) is learned from the TRAINING rows only."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.k * iqr, -np.inf)  # iqr == 0 -> leave the column alone
        self.upper_ = np.where(iqr > 0, q3 + self.k * iqr, np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by how common it is in the training rows (0 for unseen categories)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X[c].value_counts(normalize=True).to_dict() for c in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X[c].map(m).fillna(0.0).to_numpy(dtype=float) for c, m in zip(X.columns, self.maps_)])


def build_pipeline(model, plan: dict, outliers: str = "none") -> Pipeline:
    def imp():
        return SimpleImputer(strategy="most_frequent")

    unseen = dict(handle_unknown="use_encoded_value", unknown_value=-1)
    parts = []
    if plan["num"]:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if outliers == "cap":
            steps.append(("cap", IQRCapper()))
        steps.append(("scale", RobustScaler() if outliers == "robust" else StandardScaler()))
        parts.append(("num", Pipeline(steps), plan["num"]))
    if plan["binary"]:
        parts.append(("bin", Pipeline([("imp", imp()), ("enc", OrdinalEncoder(**unseen))]), plan["binary"]))
    if plan["onehot"]:
        parts.append(("oh", Pipeline([("imp", imp()), ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), plan["onehot"]))
    if plan["ordinal"]:
        parts.append(("ord", Pipeline([("imp", imp()), ("enc", OrdinalEncoder(**unseen))]), plan["ordinal"]))
    if plan["ordered"]:
        cats = [plan["orders"][c] for c in plan["ordered"]]
        parts.append(("ordd", Pipeline([("imp", imp()), ("enc", OrdinalEncoder(categories=cats, **unseen))]), plan["ordered"]))
    if plan["freq"]:
        parts.append(("freq", Pipeline([("imp", imp()), ("enc", FrequencyEncoder())]), plan["freq"]))
    if plan["target"]:
        parts.append(("tgt", Pipeline([("imp", imp()), ("enc", TargetEncoder(random_state=42))]), plan["target"]))
    return Pipeline([("prep", ColumnTransformer(parts, sparse_threshold=0)), ("model", model)])


# -------------------------------------------------- feature relevance (EDA)
@st.cache_data(show_spinner=False)
def feature_relevance(X: pd.DataFrame, y: pd.Series, task: str, num_cols: tuple) -> pd.Series:
    """How strongly each column is related to y (0-100). Uses mutual information."""
    if len(X) > 5000:
        idx = X.sample(5000, random_state=0).index
        X, y = X.loc[idx], y.loc[idx]
    enc = pd.DataFrame(index=X.index)
    discrete = []
    for c in X.columns:
        if c in num_cols:
            enc[c] = X[c].fillna(X[c].median())
            discrete.append(False)
        else:
            enc[c] = X[c].astype("category").cat.codes
            discrete.append(True)
    enc = enc.fillna(0)
    if task == "Regression":
        scores = mutual_info_regression(enc, y, discrete_features=discrete, random_state=0)
    else:
        scores = mutual_info_classif(enc, pd.factorize(y)[0], discrete_features=discrete, random_state=0)
    s = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    return (s / s.max() * 100) if s.max() > 0 else s


# ------------------------------------------------------------------ EDA charts
BLUE, ORANGE, RED, GREEN = "#4C78A8", "#F58518", "#E45756", "#54A24B"


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def grid(n, ncols=3, w=4.2, h=3.2):
    """A grid of small charts, one per column."""
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def outlier_pct(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100)


def class_colors(classes):
    return {cl: plt.cm.tab10(i % 10) for i, cl in enumerate(classes)}


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "Date"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "Number"
        else:
            kind = "Category"
        inside = ""
        if s.notna().any():
            if kind == "Number":
                inside = f"min {s.min():.4g}  ·  median {s.median():.4g}  ·  max {s.max():.4g}"
            elif kind == "Category":
                inside = ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(3).items())
        rows.append({"Column": c, "Type": kind, "Missing %": round(s.isna().mean() * 100, 1),
                     "Unique values": int(s.nunique()), "What is inside": inside})
    return pd.DataFrame(rows)


def make_insights(df, target, task, dropped, rel, num_cols, y, used_cols=None):
    out = []
    dup = int(df.duplicated().sum())
    out.append(f"🔁 **{dup:,} duplicate rows** ({dup / len(df):.1%})." if dup else "✅ No duplicate rows.")
    miss = df[used_cols if used_cols else list(df.columns)].isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        out.append(f"❓ **{len(miss)} column(s) have missing values.** Worst: `{miss.index[0]}` ({miss.iloc[0]:.0%}). They are filled in automatically before training.")
    else:
        out.append("✅ No missing values in the columns that will be used.")
    skewed = [c for c in num_cols if abs(df[c].skew()) > 1]
    if skewed:
        out.append("📈 **Skewed columns** (a few very large or very small values): " + ", ".join(f"`{c}`" for c in skewed[:6]))
    heavy = [c for c in num_cols if outlier_pct(df[c]) > 5]
    if heavy:
        out.append("🎯 **Many outliers** (more than 5% of rows): " + ", ".join(f"`{c}`" for c in heavy[:6]))
    if dropped:
        out.append(f"🧹 **{len(dropped)} column(s) will be left out of X automatically** (IDs, constants, duplicates...). See the *Prepare & Train* tab.")
    if target is not None and y is not None and len(y):
        if task == "Classification":
            share = y.value_counts(normalize=True)
            if share.min() < 0.1:
                out.append(f"⚖️ **Imbalanced target:** class `{share.idxmin()}` is only {share.min():.0%} of the rows, so accuracy alone can be misleading.")
            else:
                out.append("⚖️ The target classes are reasonably balanced.")
        elif abs(y.skew()) > 1:
            out.append(f"📈 **The target `{target}` is skewed** (skew = {y.skew():.1f}).")
        if rel is not None and len(rel):
            out.append(f"⭐ **Most related to `{target}`:** " + ", ".join(f"`{c}`" for c in rel.index[:3]))
    return out


def plot_missing(df, cols):
    miss = (df[cols].isna().mean() * 100)
    miss = miss[miss > 0].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(miss) + 1.5))
    ax.barh(miss.index.astype(str), miss.values, color=RED)
    for i, v in enumerate(miss.values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of rows missing")
    ax.set_xlim(0, max(miss.max() * 1.2, 5))
    return fig


def plot_target(y, target, task):
    if task == "Regression":
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 1]})
        a1.hist(y, bins=30, color=BLUE, edgecolor="white")
        a1.axvline(y.mean(), color=RED, ls="--", lw=2, label=f"average = {y.mean():.2f}")
        a1.axvline(y.median(), color=ORANGE, lw=2, label=f"middle (median) = {y.median():.2f}")
        a1.set_xlabel(target)
        a1.set_ylabel("Number of rows")
        a1.set_title("How the values are spread")
        a1.legend(fontsize=8)
        a2.boxplot(y.dropna(), patch_artist=True, boxprops=dict(facecolor=BLUE, alpha=0.6))
        a2.set_xticks([])
        a2.set_title("Boxplot (dots = outliers)")
    else:
        counts = y.value_counts().head(15)[::-1]
        colors = class_colors(list(counts.index))
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(counts) + 1.8))
        ax.barh(counts.index.astype(str), counts.values, color=[colors[c] for c in counts.index])
        for i, v in enumerate(counts.values):
            ax.text(v, i, f" {v:,}  ({v / len(y):.0%})", va="center", fontsize=9)
        ax.set_xlim(0, counts.max() * 1.25)
        ax.set_xlabel("Number of rows")
        ax.set_title(f"How many rows in each class of `{target}`")
    return fig


def plot_relevance(rel, k):
    top = rel.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(top) + 1.6))
    ax.barh(top.index.astype(str), top.values, color=GREEN)
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Relevance to y  (100 = strongest)")
    return fig


def plot_hists(data, cols):
    fig, axes = grid(len(cols))
    for ax, c in zip(axes, cols):
        s = data[c].dropna()
        if s.empty:
            continue
        ax.hist(s, bins=min(30, max(5, s.nunique())), color=BLUE, edgecolor="white")
        ax.axvline(s.median(), color=ORANGE, ls="--", lw=1.8)
        ax.set_title(f"{c}\nskew {s.skew():.1f}  ·  outliers {outlier_pct(s):.0f}%", fontsize=9)
        ax.tick_params(labelsize=8)
    return fig


def plot_box(data, cols):
    z = [((data[c] - data[c].mean()) / (data[c].std() or 1)).dropna() for c in cols]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(cols) + 1.8))
    bp = ax.boxplot(z, vert=False, patch_artist=True, flierprops=dict(marker=".", markersize=4, markerfacecolor=RED, markeredgecolor=RED))
    for b in bp["boxes"]:
        b.set(facecolor=BLUE, alpha=0.5)
    ax.set_yticks(range(1, len(cols) + 1))
    ax.set_yticklabels(cols)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Standardised value (0 = average)")
    return fig


def plot_cat_counts(data, cols):
    fig, axes = grid(len(cols), ncols=2, w=5, h=3)
    for ax, c in zip(axes, cols):
        vc = data[c].value_counts().head(8)[::-1]
        ax.barh(vc.index.astype(str), vc.values, color=BLUE)
        for i, v in enumerate(vc.values):
            ax.text(v, i, f" {v / data[c].notna().sum():.0%}", va="center", fontsize=8)
        ax.set_xlim(0, vc.max() * 1.2)
        ax.set_title(f"{c}  ({data[c].nunique()} categories)", fontsize=9)
        ax.tick_params(labelsize=8)
    return fig


def plot_rel_num(data, cols, target, task):
    fig, axes = grid(len(cols))
    if task == "Regression":
        for ax, c in zip(axes, cols):
            d = data[[c, target]].dropna()
            if len(d) > 2000:
                d = d.sample(2000, random_state=0)
            ax.scatter(d[c], d[target], s=8, alpha=0.35, color=BLUE)
            r = d[c].corr(d[target])
            if d[c].nunique() > 1:
                m, b = np.polyfit(d[c], d[target], 1)
                xs = np.array([d[c].min(), d[c].max()])
                ax.plot(xs, m * xs + b, color=RED, lw=2)
            ax.set_title(f"{c}  (correlation {r:+.2f})", fontsize=9)
            ax.set_xlabel(c, fontsize=8)
            ax.set_ylabel(target, fontsize=8)
            ax.tick_params(labelsize=8)
    else:
        y = data[target]
        classes = y.value_counts().head(8).index.tolist()
        colors = class_colors(classes)
        for ax, c in zip(axes, cols):
            groups = [data.loc[y == cl, c].dropna() for cl in classes]
            bp = ax.boxplot(groups, patch_artist=True, showfliers=False)
            for patch, cl in zip(bp["boxes"], classes):
                patch.set_facecolor(colors[cl])
            ax.set_xticks(range(1, len(classes) + 1))
            ax.set_xticklabels([str(cl) for cl in classes], rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{c} by class", fontsize=9)
            ax.tick_params(labelsize=8)
    return fig


def plot_rel_cat(data, cols, target, task):
    fig, axes = grid(len(cols), ncols=2, w=5.5, h=3.3)
    if task == "Regression":
        overall = data[target].mean()
        for ax, c in zip(axes, cols):
            g = data.groupby(c)[target].agg(["mean", "count"])
            g = g.sort_values("count", ascending=False).head(10).sort_values("mean")
            ax.barh(g.index.astype(str), g["mean"], color=BLUE)
            ax.axvline(overall, color=RED, ls="--", lw=1.5)
            for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
                ax.text(m, i, f" n={n}", va="center", fontsize=7)
            ax.set_title(f"Average {target} by {c}", fontsize=9)
            ax.tick_params(labelsize=8)
        fig.text(0.5, 0.0, "red dashed line = overall average", ha="center", fontsize=8, color=RED)
    else:
        y = data[target]
        classes = y.value_counts().head(8).index.tolist()
        colors = class_colors(classes)
        for ax, c in zip(axes, cols):
            top = data[c].value_counts().head(8).index
            d = data[data[c].isin(top) & y.isin(classes)]
            ct = pd.crosstab(d[c], d[target], normalize="index").reindex(columns=classes, fill_value=0) * 100
            left = np.zeros(len(ct))
            for cl in classes:
                ax.barh(ct.index.astype(str), ct[cl].values, left=left, color=colors[cl])
                left += ct[cl].values
            ax.set_xlim(0, 100)
            ax.set_xlabel("% of rows in each class", fontsize=8)
            ax.set_title(f"{c}", fontsize=9)
            ax.tick_params(labelsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[cl]) for cl in classes]
        fig.legend(handles, [str(cl) for cl in classes], loc="upper center", ncol=min(5, len(classes)), fontsize=8,
                   bbox_to_anchor=(0.5, 1.03), title=target)
    return fig


def plot_corr(data, cols):
    corr = data[cols].corr()
    n = len(cols)
    size = max(5, 0.7 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig, corr


# -------------------------------------------------------------------- EDA tab
def render_eda(df, target, task, features, plan, dropped):
    has_target = target is not None and bool(features)
    data = df.dropna(subset=[target]) if target is not None else df
    y = data[target] if target is not None else None

    rel = None
    if has_target:
        rel = feature_relevance(data[features], y, task, tuple(plan["num"]))
        num_cols = [c for c in rel.index if c in plan["num"]]
        cat_cols = [c for c in rel.index if c not in plan["num"]]
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_datetime64_any_dtype(df[c]) and 1 < df[c].nunique() <= MAX_CATEGORIES]
    num_cols = [c for c in num_cols if data[c].notna().any()]

    k = st.slider("How many columns to show in each chart group", 3, 12, 6,
                  help="Columns are ordered from most to least related to y.")
    t_over, t_target, t_num, t_cat, t_rel, t_corr = st.tabs(
        ["🧭 Overview", "🎯 Target (y)", "🔢 Numeric columns", "🔤 Category columns", "🔗 Relationship with y", "🌡️ Correlation"]
    )

    # ---- overview
    with t_over:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.markdown("#### Quick findings")
        for line in make_insights(df, target, task, dropped, rel, num_cols, y, (features + [target]) if has_target else None):
            st.markdown(f"- {line}")
        st.markdown("#### First rows")
        st.dataframe(df.head(10))
        st.markdown("#### Every column at a glance")
        st.dataframe(column_profile(df))
        if df.isna().any().any():
            st.markdown("#### Missing values")
            show(plot_missing(df, list(df.columns)))
        with st.expander("Detailed summary statistics"):
            st.dataframe(df.describe(include="all").T)

    # ---- target
    with t_target:
        if target is None:
            st.info("👈 Choose the target column (y) in the sidebar.")
        else:
            st.markdown(f"#### What you want to predict: `{target}` ({task})")
            if task == "Regression":
                m = st.columns(6)
                for col, (lab, val) in zip(m, [("Smallest", y.min()), ("Middle (median)", y.median()), ("Average", y.mean()),
                                               ("Largest", y.max()), ("Skew", y.skew()), ("Outliers %", outlier_pct(y))]):
                    col.metric(lab, f"{val:,.2f}")
            show(plot_target(y, target, task))
            if task == "Regression":
                st.caption("If the average (red) is far from the median (orange), a few extreme values are pulling the average. "
                           "Dots in the boxplot are outliers.")
            elif y.value_counts(normalize=True).min() < 0.1:
                st.caption("⚠️ Some classes are rare, so accuracy alone can be misleading. Look at recall and F1 too.")
            else:
                st.caption("Classes are reasonably balanced.")

    # ---- numeric
    with t_num:
        cols = num_cols[:k]
        if not cols:
            st.info("No numeric columns.")
        else:
            st.markdown("#### How each number column is spread")
            show(plot_hists(data, cols))
            st.caption("Orange dashed line = median. **skew** far from 0 means a long tail on one side. "
                       "**outliers** = % of rows far outside the normal range.")
            if len(cols) > 1:
                st.markdown("#### Outliers side by side")
                show(plot_box(data, cols))
                st.caption("Each column is rescaled so they fit on one axis. Red dots are outliers.")

    # ---- categories
    with t_cat:
        cols = cat_cols[:k]
        if not cols:
            st.info("No category (text) columns.")
        else:
            st.markdown("#### Most common values in each category column")
            show(plot_cat_counts(data, cols))
            st.caption("Labels show the share of rows. Only the 8 most common values are drawn.")

    # ---- relationship with y
    with t_rel:
        if not has_target:
            st.info("👈 Choose the target column (y) in the sidebar.")
        else:
            st.markdown("#### Which columns matter most?")
            show(plot_relevance(rel, k))
            st.caption("Score from 0 to 100 that works for both numbers and categories, "
                       "and also catches curved (non-linear) relationships.")
            rn = [c for c in num_cols if c in rel.index][:min(k, 6)]
            if rn:
                st.markdown("#### Numeric columns vs y")
                show(plot_rel_num(data, rn, target, task))
                st.caption("Red line = overall trend. A clear slope means a useful column." if task == "Regression"
                           else "If the boxes sit at different heights, the column helps separate the classes.")
            rc = cat_cols[:min(k, 4)]
            if rc:
                st.markdown("#### Category columns vs y")
                show(plot_rel_cat(data, rc, target, task))
                st.caption("If bars differ a lot from category to category, the column is useful." if task == "Regression"
                           else "If the colour mix changes between categories, the column helps separate the classes.")

    # ---- correlation
    with t_corr:
        cols = num_cols[:max(k, 6)]
        if has_target and task == "Regression" and target not in cols:
            cols = cols + [target]
        if len(cols) < 2:
            st.info("Need at least two numeric columns.")
        else:
            fig, corr = plot_corr(data, cols)
            show(fig)
            st.caption("Red = rise together, blue = one rises while the other falls, white = no linear link.")
            pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("correlation").reset_index()
            pairs.columns = ["Column A", "Column B", "Correlation"]
            pairs = pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index).head(5)
            st.markdown("#### Strongest pairs")
            st.dataframe(pairs.round(2))
            if (pairs["Correlation"].abs() > 0.8).any():
                st.caption("⚠️ Two columns with a correlation above 0.8 carry nearly the same information.")


# ------------------------------------------------------------ data cleaning
def is_text(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)


def norm_text(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def find_spelling_variants(df: pd.DataFrame) -> dict:
    """{column: {normalised_value: [different spellings]}} for words written in more than one way."""
    out = {}
    for c in df.columns:
        if not is_text(df[c]):
            continue
        vals = df[c].dropna().astype(str).unique()
        if len(vals) > 300:
            continue
        groups = {}
        for v in vals:
            groups.setdefault(norm_text(v), []).append(v)
        bad = {k: v for k, v in groups.items() if len(v) > 1}
        if bad:
            out[c] = bad
    return out


def suggest_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Valid-range suggestions: ages 0-120, and 'no negatives' when only a few values are negative."""
    rows = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or s.notna().sum() == 0:
            continue
        lo = hi = np.nan
        if re.search(r"(^|_)age($|_)", c.lower()):
            lo, hi = 0.0, 120.0
        elif (s < 0).any() and (s < 0).mean() < 0.01 and (s >= 0).mean() > 0.98:
            lo = 0.0
        rows.append({"Column": c, "Lowest in data": float(s.min()), "Highest in data": float(s.max()),
                     "Min allowed": lo, "Max allowed": hi})
    return pd.DataFrame(rows, columns=["Column", "Lowest in data", "Highest in data", "Min allowed", "Max allowed"])


def clean_data(df, remove_duplicates=False, normalize=False, value_maps=None, ranges=None, range_action="missing"):
    """Apply the chosen cleaning steps. Returns (clean_df, report) where report is a list of dicts."""
    out = df.copy()
    report = []
    if normalize:
        merged = 0
        for c in [c for c in out.columns if is_text(out[c])]:
            before = out[c].nunique()
            out[c] = out[c].map(lambda v: norm_text(v) if pd.notna(v) else v)
            merged += before - out[c].nunique()
        report.append({"Step": "Make text consistent", "What changed": f"{merged} spelling variants merged (lowercase, trimmed, single spaces)"})
    for c, m in (value_maps or {}).items():
        if c in out.columns and m:
            changed = int(out[c].isin(list(m)).sum())
            out[c] = out[c].map(lambda v, m=m: m.get(v, v) if pd.notna(v) else v)
            report.append({"Step": f"Rename values in `{c}`", "What changed": f"{changed} cells changed ({', '.join(f'{k} → {v}' for k, v in list(m.items())[:4])})"})
    if remove_duplicates:
        n0 = len(out)
        out = out.drop_duplicates()
        report.append({"Step": "Remove duplicate rows", "What changed": f"{n0 - len(out)} rows removed"})
    if ranges:
        drop_mask = pd.Series(False, index=out.index)
        for c, (lo, hi) in ranges.items():
            if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
                continue
            mask = pd.Series(False, index=out.index)
            if pd.notna(lo):
                mask |= out[c] < lo
            if pd.notna(hi):
                mask |= out[c] > hi
            if mask.any():
                if range_action == "missing":
                    out.loc[mask, c] = np.nan
                else:
                    drop_mask |= mask
                report.append({"Step": f"Valid range for `{c}`", "What changed": f"{int(mask.sum())} values outside {lo if pd.notna(lo) else '−∞'} … {hi if pd.notna(hi) else '∞'}"})
        if drop_mask.any():
            out = out[~drop_mask]
            report.append({"Step": "Remove rows with invalid values", "What changed": f"{int(drop_mask.sum())} rows removed"})
    return out, report


def render_cleaning(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Optional cleaning controls. Nothing changes unless you tick something."""
    variants = find_spelling_variants(df)
    n_dup = int(df.duplicated().sum())
    sug = suggest_ranges(df)
    n_bad = 0
    for _, r in sug.iterrows():
        s = df[r["Column"]]
        if pd.notna(r["Min allowed"]):
            n_bad += int((s < r["Min allowed"]).sum())
        if pd.notna(r["Max allowed"]):
            n_bad += int((s > r["Max allowed"]).sum())
    issues = (n_dup > 0) + bool(variants) + (n_bad > 0)
    label = "🧹 Data cleaning (optional)" + (f"  ·  ⚠️ {issues} possible issue(s) found" if issues else "  ·  no obvious problems")

    with st.expander(label):
        k = lambda name: f"{name}_{file_name}"  # noqa: E731
        c1, c2 = st.columns(2)
        remove_dup = c1.checkbox(f"Remove duplicate rows ({n_dup:,} found)", key=k("dup"))
        normalize = c2.checkbox(f"Make text consistent ({len(variants)} column(s) with mixed spelling)", key=k("norm"),
                                help="Lowercases text, trims spaces and merges 'Male', 'male ' and 'MALE' into one value.")
        for c, groups in list(variants.items())[:4]:
            example = list(groups.values())[0]
            st.caption(f"`{c}`: written several ways, for example {', '.join(repr(v) for v in example[:4])}")

        base, _ = clean_data(df, normalize=normalize)

        st.markdown("**Rename or merge values** (for example `M` → `male`)")
        text_cols = [c for c in base.columns if is_text(base[c]) and 1 < base[c].nunique() <= 60]
        maps = st.session_state.setdefault(f"maps_{file_name}", {})
        col = st.selectbox("Column to edit", ["(none)"] + text_cols, key=k("mapcol"))
        if col != "(none)":
            vc = base[col].value_counts()
            stored = maps.get(col, {})
            tbl = pd.DataFrame({"Value": vc.index.astype(str), "Rows": vc.values,
                                "Replace with": [stored.get(v, "") for v in vc.index.astype(str)]})
            edited = st.data_editor(tbl, disabled=["Value", "Rows"], hide_index=True, key=f"map_{file_name}_{col}")
            maps[col] = {v: r.strip() for v, r in zip(edited["Value"], edited["Replace with"])
                         if isinstance(r, str) and r.strip() and r.strip() != v}

        st.markdown("**Valid ranges** (impossible values such as an age of −5 or 999)")
        use_ranges, ranges, action = False, {}, "missing"
        if len(sug):
            use_ranges = st.checkbox("Treat values outside these ranges as errors", key=k("userng"))
            st.caption("Suggestions come from column names (age: 0–120) and from rare negative numbers. Edit or clear any of them.")
            rng = st.data_editor(sug, disabled=["Column", "Lowest in data", "Highest in data"], hide_index=True, key=k("rng"),
                                 column_config={"Min allowed": st.column_config.NumberColumn(), "Max allowed": st.column_config.NumberColumn()})
            if use_ranges:
                ranges = {r["Column"]: (r["Min allowed"], r["Max allowed"]) for _, r in rng.iterrows()
                          if pd.notna(r["Min allowed"]) or pd.notna(r["Max allowed"])}
                choice = st.radio("What to do with those values", ["Set to missing (filled in automatically)", "Remove those rows"],
                                  horizontal=True, key=k("action"))
                action = "missing" if choice.startswith("Set") else "drop"

        clean, report = clean_data(df, remove_dup, normalize, {c: m for c, m in maps.items() if m}, ranges, action)
        if report:
            st.markdown(f"**What was cleaned** — rows: {len(df):,} → {len(clean):,}")
            st.dataframe(pd.DataFrame(report), hide_index=True)
        else:
            st.caption("Nothing is changed until you tick an option above.")
    return clean


# -------------------------------------------------------- leakage detection
STOP_TOKENS = {"total", "count", "amount", "num", "number", "avg", "mean", "min", "max", "net", "daily",
               "pct", "percent", "the", "and", "for", "per", "gram", "grams", "item", "items"}


def name_stems(name) -> set:
    toks = re.split(r"[^a-z0-9]+", str(name).lower())
    return {t[:5] for t in toks if len(t) >= 4 and t not in STOP_TOKENS}


def as_codes(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.median() if s.notna().any() else 0)
    return s.astype("category").cat.codes


def skill_score(X: pd.DataFrame, y: pd.Series, task: str) -> float:
    """0 = no better than guessing, 1 = perfect. R² for regression, accuracy gain over the majority class otherwise."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    if task == "Regression":
        m = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=0).fit(Xtr, ytr)
        return float(max(0.0, m.score(Xte, yte)))
    m = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=0).fit(Xtr, ytr)
    base = ytr.value_counts(normalize=True).iloc[0]
    return float(max(0.0, (m.score(Xte, yte) - base) / (1 - base))) if base < 1 else 0.0


def detect_leakage(df: pd.DataFrame, target: str, features: list, task: str) -> dict:
    """Look for columns that give away y. Returns {'table', 'auto_exclude', 'formula_msg'}."""
    data = df.dropna(subset=[target])
    if len(data) > 3000:
        data = data.sample(3000, random_state=0)
    y = data[target].astype(float) if task == "Regression" else pd.Series(pd.factorize(data[target])[0], index=data.index)
    t_stems = name_stems(target)
    rows = []
    for c in features:
        if data[c].nunique() < 2:
            continue
        skill = skill_score(as_codes(data[c]).to_frame(), y, task)
        same_name = bool(t_stems & name_stems(c))
        if skill >= 0.99:
            risk, why = "High", "this column alone predicts y almost perfectly"
        elif same_name and skill >= 0.5:
            risk, why = "High", "its name matches y and it explains most of y, so it is probably calculated from y"
        elif skill >= 0.9:
            risk, why = "Medium", "very strong on its own: check it is really known before y is known"
        elif same_name:
            risk, why = "Low", "its name looks related to y"
        else:
            continue
        rows.append({"Column": c, "Alone explains y": f"{skill:.0%}", "Risk": risk, "Why": why, "_s": skill})
    table = pd.DataFrame(rows, columns=["Column", "Alone explains y", "Risk", "Why", "_s"])
    if len(table):
        order = table["Risk"].map({"High": 0, "Medium": 1, "Low": 2})
        table = table.assign(_o=order).sort_values(["_o", "_s"], ascending=[True, False]).drop(columns=["_o", "_s"])
    exclude = table.loc[table["Risk"] == "High", "Column"].tolist() if len(table) else []

    formula_msg = ""
    rest = [c for c in features if c not in exclude and data[c].nunique() > 1]
    if rest:
        enc = pd.DataFrame({c: as_codes(data[c]) for c in rest})
        if task == "Regression":
            num = [c for c in rest if pd.api.types.is_numeric_dtype(data[c])]
            if num:
                Xtr, Xte, ytr, yte = train_test_split(enc[num], y, test_size=0.3, random_state=0)
                r2 = LinearRegression().fit(Xtr, ytr).score(Xte, yte)
                if r2 >= 0.995:
                    formula_msg = (f"The remaining numeric columns predict `{target}` almost perfectly with a plain straight-line formula "
                                   f"(R² = {r2:.4f}). `{target}` is probably calculated from them, so high scores are expected and "
                                   "do not prove the model is smart.")
        elif skill_score(enc, y, task) >= 0.99:
            formula_msg = (f"The remaining columns almost perfectly determine `{target}`. A very high accuracy is expected "
                           "and may not mean much.")
    return {"table": table, "auto_exclude": exclude, "formula_msg": formula_msg}


@st.cache_data(show_spinner=False)
def detect_leakage_cached(df, target, features: tuple, task):
    return detect_leakage(df, target, list(features), task)


# ----------------------------------------- evaluation, tuning, explanation
SCORING = {"Regression": "r2", "Classification": "accuracy"}


def make_cv(task: str, y: pd.Series, folds: int):
    if task == "Classification" and y.value_counts().min() >= folds:
        return StratifiedKFold(folds, shuffle=True, random_state=42)
    return KFold(folds, shuffle=True, random_state=42)


def score_pipe(pipe, X_tr, X_te, y_tr, y_te, task) -> dict:
    """Measure a fitted pipeline on the training rows and on the unseen test rows."""
    p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
    if task == "Regression":
        return dict(pipe=pipe, preds=p_te, train=r2_score(y_tr, p_tr), test=r2_score(y_te, p_te),
                    MAE=mean_absolute_error(y_te, p_te), RMSE=float(np.sqrt(mean_squared_error(y_te, p_te))))
    return dict(pipe=pipe, preds=p_te, train=accuracy_score(y_tr, p_tr), test=accuracy_score(y_te, p_te),
                F1=f1_score(y_te, p_te, average="weighted", zero_division=0))


def fit_model(model, plan, X_tr, X_te, y_tr, y_te, task, outliers="none", cv_folds=0):
    t0 = time.time()
    pipe = build_pipeline(model, plan, outliers)
    pipe.fit(X_tr, y_tr)
    res = score_pipe(pipe, X_tr, X_te, y_tr, y_te, task)
    res.update(cv_mean=np.nan, cv_std=np.nan, params=None)
    if cv_folds:
        try:  # cross-validation on the TRAINING rows only; the test rows stay untouched
            sc = cross_val_score(pipe, X_tr, y_tr, cv=make_cv(task, y_tr, cv_folds), scoring=SCORING[task])
            res.update(cv_mean=float(sc.mean()), cv_std=float(sc.std()))
        except Exception:
            pass
    res["secs"] = time.time() - t0
    return res


def param_space(name: str):
    p = "model__"
    return {
        "Ridge": {p + "alpha": loguniform(0.01, 100)},
        "Lasso": {p + "alpha": loguniform(0.001, 10)},
        "Logistic Regression": {p + "C": loguniform(0.01, 100)},
        "Decision Tree": {p + "max_depth": [3, 5, 8, 12, 20, None], p + "min_samples_leaf": [1, 2, 5, 10, 20]},
        "Random Forest": {p + "n_estimators": [100, 200, 300], p + "max_depth": [None, 5, 10, 20],
                          p + "min_samples_leaf": [1, 2, 5], p + "max_features": ["sqrt", 0.5, 1.0]},
        "Gradient Boosting": {p + "n_estimators": [100, 200, 300], p + "learning_rate": [0.03, 0.05, 0.1, 0.2],
                              p + "max_depth": [2, 3, 4, 5], p + "subsample": [0.7, 0.85, 1.0]},
        "AdaBoost": {p + "n_estimators": [50, 100, 200, 300], p + "learning_rate": [0.03, 0.1, 0.3, 1.0]},
        "KNN": {p + "n_neighbors": [3, 5, 7, 9, 11, 15, 21], p + "weights": ["uniform", "distance"], p + "p": [1, 2]},
        "XGBoost": {p + "n_estimators": [100, 200, 300], p + "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    p + "max_depth": [3, 4, 6, 8], p + "subsample": [0.7, 0.85, 1.0], p + "colsample_bytree": [0.7, 0.85, 1.0]},
    }.get(name)


def tune_model(name, model, plan, outliers, X_tr, X_te, y_tr, y_te, task, n_iter=12, folds=3):
    """Random search for better settings. Uses only the training rows; the test rows judge the result."""
    space = param_space(name)
    if space is None:
        return None
    t0 = time.time()
    search = RandomizedSearchCV(build_pipeline(clone(model), plan, outliers), space, n_iter=n_iter,
                                cv=make_cv(task, y_tr, folds), scoring=SCORING[task], random_state=42, n_jobs=1)
    search.fit(X_tr, y_tr)
    res = score_pipe(search.best_estimator_, X_tr, X_te, y_tr, y_te, task)
    res.update(cv_mean=float(search.best_score_), cv_std=float(search.cv_results_["std_test_score"][search.best_index_]),
               secs=time.time() - t0,
               params={k.replace("model__", ""): (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                       for k, v in search.best_params_.items()})
    return res


def leaderboard(fitted: dict, task: str) -> pd.DataFrame:
    score = "R²" if task == "Regression" else "Accuracy"
    rows = []
    for name, r in fitted.items():
        row = {"Model": name, f"Train {score}": round(r["train"], 4), f"Test {score}": round(r["test"], 4)}
        if pd.notna(r.get("cv_mean", np.nan)):
            row[f"CV {score} (mean)"] = round(r["cv_mean"], 4)
            row["CV ±"] = round(r["cv_std"], 4)
        if task == "Regression":
            row.update({"MAE": round(r["MAE"], 3), "RMSE": round(r["RMSE"], 3)})
        else:
            row["F1"] = round(r["F1"], 4)
        row["Overfitting?"] = "⚠️ yes" if r["train"] - r["test"] > 0.10 else "no"
        row["Time (s)"] = round(r["secs"], 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    key = f"CV {score} (mean)" if f"CV {score} (mean)" in df and df[f"CV {score} (mean)"].notna().all() else f"Test {score}"
    return df.sort_values(key, ascending=False).reset_index(drop=True)


def best_model_name(fitted: dict) -> str:
    use_cv = all(pd.notna(r.get("cv_mean", np.nan)) for r in fitted.values())
    return max(fitted, key=lambda k: fitted[k]["cv_mean"] if use_cv else fitted[k]["test"])


# ------------------------------------------------------------- explanation
def importance_table(pipe, X_te, y_te, task, repeats=5) -> pd.DataFrame:
    """Permutation importance: how much the score drops when one column is shuffled."""
    if len(X_te) > 2000:
        idx = X_te.sample(2000, random_state=0).index
        X_te, y_te = X_te.loc[idx], y_te.loc[idx]
    r = permutation_importance(pipe, X_te, y_te, scoring=SCORING[task], n_repeats=repeats, random_state=42, n_jobs=1)
    return (pd.DataFrame({"Column": X_te.columns, "Importance": r.importances_mean, "Spread": r.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, task: str, k: int = 12):
    top = imp.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(top) + 1.6))
    ax.barh(top["Column"].astype(str), top["Importance"], xerr=top["Spread"], color=GREEN, error_kw=dict(lw=1, capsize=2))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"Drop in {'R²' if task == 'Regression' else 'accuracy'} when the column is shuffled")
    return fig


def partial_dependence_curve(pipe, X_ref, col, plan, task, class_index=None, n_grid=20, sample=300):
    """Average prediction as one column is moved across its range (all other columns stay as they are)."""
    base = X_ref.sample(min(sample, len(X_ref)), random_state=0)
    if col in plan["num"]:
        vals = np.unique(np.nanpercentile(X_ref[col].dropna(), np.linspace(2, 98, n_grid)))
    else:
        vals = X_ref[col].value_counts().head(12).index.tolist()
    ys = []
    for v in vals:
        tmp = base.copy()
        tmp[col] = v
        ys.append(float(pipe.predict(tmp).mean()) if task == "Regression" else float(pipe.predict_proba(tmp)[:, class_index].mean()))
    return vals, ys


def plot_pdp(vals, ys, col, is_num, label):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if is_num:
        ax.plot(vals, ys, marker="o", color=BLUE)
    else:
        ax.bar([str(v) for v in vals], ys, color=BLUE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel(label)
    return fig


def local_effects(pipe, row, X_ref, plan, task):
    """How much each input moves THIS prediction compared with a typical value (approximate, one column at a time)."""
    feats = list(row.columns)
    typical = {c: (X_ref[c].median() if c in plan["num"] else X_ref[c].mode().iloc[0]) for c in feats}
    if task == "Regression":
        def f(r):
            return float(pipe.predict(r)[0])
        label = "Change in the prediction"
    else:
        k = int(np.argmax(pipe.predict_proba(row)[0]))

        def f(r):
            return float(pipe.predict_proba(r)[0][k])
        label = "Change in the probability of the predicted class"
    full = f(row)
    eff = {}
    for c in feats:
        tmp = row.copy()
        tmp[c] = typical[c]
        eff[c] = full - f(tmp)
    return pd.Series(eff).sort_values(key=lambda s: s.abs(), ascending=False), label


def plot_effects(eff: pd.Series, label: str, k: int = 8):
    top = eff.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(top) + 1.4))
    ax.barh(top.index.astype(str), top.values, color=[GREEN if v >= 0 else RED for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(label)
    return fig


# ------------------------------------------------------- encoding editor UI
def render_encoding_editor(df, features, key):
    """Optional per-column override of the automatic encoding."""
    overrides, orders = {}, {}
    if not features:
        return overrides, orders
    auto = make_plan(df[features])
    short = {"num": "Numeric", "binary": "Binary 0/1", "onehot": "One-hot", "ordinal": "Label"}
    tbl = pd.DataFrame([{"Column": c, "Unique values": int(df[c].nunique()), "Automatic": short[g], "Your choice": "Auto"}
                        for g in GROUPS for c in auto[g]])
    sig = abs(hash(tuple(features))) % 10**8
    with st.expander("⚙️ Advanced: change how a column is encoded (optional)"):
        st.caption("Leave everything on **Auto** unless you know better. Example: set `fitness_level` to *Ordered* and "
                   "write its order from lowest to highest, so the model knows Sedentary < Active < Trained.")
        edited = st.data_editor(
            tbl, hide_index=True, disabled=["Column", "Unique values", "Automatic"], key=f"enc_{key}_{sig}",
            column_config={"Your choice": st.column_config.SelectboxColumn(options=ENC_CHOICES, required=True)},
        )
        for _, r in edited.iterrows():
            if r["Your choice"] != "Auto":
                overrides[r["Column"]] = r["Your choice"]
        for c, ch in overrides.items():
            if ch.startswith("Ordered") and check_choice(df[c], ch)[0]:
                txt = st.text_input(f"Order for `{c}` (lowest → highest, separated by commas)",
                                    ", ".join(sorted_values(df[c])), key=f"ord_{key}_{c}")
                orders[c] = [v.strip() for v in txt.split(",") if v.strip()]
    return overrides, orders


def render_leakage(leak):
    st.subheader("🕵️ Leakage check")
    if leak["formula_msg"]:
        st.warning("⚠️ " + leak["formula_msg"])
    t = leak["table"]
    if len(t):
        st.dataframe(t, hide_index=True)
        st.caption("**High**-risk columns are left out automatically (add them back in the sidebar if you disagree). "
                   "Leakage means a column that would not be known when you make a real prediction.")
    elif not leak["formula_msg"]:
        st.success("No column gives away y on its own.")
    st.caption("The app can only spot suspicious numbers and names. Only you know which columns would really be available at prediction time.")


# ------------------------------------------------------------ train & model
def render_train(df, target, task, features, dropped, plan, file_name, leak):
    if target is None:
        st.info("👈 Choose the target column (y) in the sidebar first.")
        return
    if not features:
        st.warning("No usable feature columns were found.")
        return
    data = df.dropna(subset=[target]).copy()
    if len(data) < 20:
        st.error("Need at least 20 rows with a value in the target column.")
        return
    X = align_features(data[features], plan)
    le = None
    if task == "Regression":
        y = data[target]
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(data[target].astype(str) if not pd.api.types.is_numeric_dtype(data[target]) else data[target]), index=data.index)

    if leak is not None:
        render_leakage(leak)

    st.subheader("Automatic data preparation")
    st.markdown(f"**Target (y):** `{target}`  ·  **Problem type:** {task}  ·  **Features (X):** {len(features)} columns")
    for note in plan["notes"]:
        st.warning(note)
    st.dataframe(plan_table(X, plan))
    if dropped:
        with st.expander(f"Columns left out ({len(dropped)})"):
            st.dataframe(pd.DataFrame({"column": list(dropped), "reason": list(dropped.values())}), hide_index=True)

    st.subheader("Train")
    models = get_models(task)
    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox("Model", list(models))
    test_size = c2.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    cv_label = c3.selectbox("Cross-validation", ["5-fold", "3-fold", "Off"], index=1 if len(data) > 30000 else 0,
                            help="Trains the model several times on different parts of the training rows. "
                                 "Gives a more reliable score than one split.")
    cv_folds = {"5-fold": 5, "3-fold": 3, "Off": 0}[cv_label]
    out_label = st.selectbox(
        "Outlier handling", list(OUTLIER_OPTIONS),
        help="Cap: values beyond Q1-1.5×IQR / Q3+1.5×IQR are pulled back to that limit. "
             "Robust scaling: scales with median and IQR. Tree models are already fairly robust; "
             "linear models and KNN gain the most.",
    )
    outliers = OUTLIER_OPTIONS[out_label]
    if not HAS_XGB:
        st.caption("XGBoost is not installed. Run `pip install xgboost` to add it.")
    b1, b2 = st.columns(2)
    train_one = b1.button("🚀 Train selected model", type="primary")
    train_all = b2.button("🏁 Compare ALL models")
    with st.expander("🔧 Hyperparameter tuning (look for better settings)"):
        if param_space(model_name) is None:
            st.caption(f"{model_name} has no settings worth tuning.")
        n_iter = st.slider("Number of settings to try", 5, 40, 12)
        tune = st.button(f"Tune {model_name}", disabled=param_space(model_name) is None)
        st.caption("Settings are chosen with cross-validation on the training rows. The test rows only judge the final result.")

    data_hash = int(pd.util.hash_pandas_object(data[features + [target]], index=False).sum())
    sig = dict(file=file_name, target=target, task=task, test_size=test_size, features=tuple(features),
               outliers=outliers, plan=str(plan), cv=cv_folds, data=data_hash)
    info = st.session_state.get("info")
    if info is not None and info["sig"] != sig:
        info = None  # settings changed, old results are no longer valid

    if train_one or train_all or tune:
        stratify = None
        if task == "Classification":
            vc = y.value_counts()
            if vc.min() >= 2 and int(np.ceil(test_size * len(y))) >= len(vc):
                stratify = y
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
        if info is None:
            info = dict(sig=sig, file=file_name, target=target, task=task, plan=plan, features=features, X_ref=X,
                        X_te=X_te, y_te=y_te, le=le, n_train=len(X_tr), n_test=len(X_te), fitted={}, active=None, tuning={})

        def run_fit(nm):
            try:
                info["fitted"][nm] = fit_model(models[nm], plan, X_tr, X_te, y_tr, y_te, task, outliers, cv_folds)
            except Exception as e:
                st.warning(f"{nm} failed and was skipped: {e}")

        if train_one or train_all:
            names = list(models) if train_all else [model_name]
            bar = st.progress(0.0, text="Training...")
            for i, nm in enumerate(names):
                bar.progress(i / len(names), text=f"Training {nm} ({i + 1}/{len(names)})" + (" with cross-validation" if cv_folds else ""))
                run_fit(nm)
            bar.empty()
            if info["fitted"]:
                info["active"] = best_model_name(info["fitted"]) if train_all else (model_name if model_name in info["fitted"] else info["active"])
        if tune:
            if model_name not in info["fitted"]:
                run_fit(model_name)
            if model_name in info["fitted"]:
                try:
                    with st.spinner(f"Trying {n_iter} settings for {model_name}..."):
                        res = tune_model(model_name, models[model_name], plan, outliers, X_tr, X_te, y_tr, y_te, task, n_iter, 3)
                    info["fitted"][f"{model_name} (tuned)"] = res
                    info["tuning"][model_name] = f"{model_name} (tuned)"
                    info["active"] = f"{model_name} (tuned)"
                except Exception as e:
                    st.error(f"Tuning failed: {e}")
        st.session_state["info"] = info

    if info is None or not info["fitted"]:
        return

    st.success(f"Split done automatically: {info['n_train']:,} training rows, {info['n_test']:,} test rows. "
               f"Outliers: {[k for k, v in OUTLIER_OPTIONS.items() if v == info['sig']['outliers']][0].lower()}.")
    st.subheader("How accurate is it?")
    score_name = "R²" if task == "Regression" else "Accuracy"
    st.caption(
        f"**Test {score_name}** = how well the model does on rows it has never seen (the real accuracy). "
        f"**Train {score_name}** = how well it does on rows it learned from. "
        f"**CV {score_name}** = the average over several cross-validation rounds; **CV ±** shows how much it moves. "
        "A big gap between Train and Test means the model is memorising (overfitting)."
    )
    board = leaderboard(info["fitted"], task)
    st.dataframe(board)
    if len(board) > 1:
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(board) + 1.5))
        b = board.iloc[::-1]
        ax.barh(b["Model"], b[f"Test {score_name}"], color=BLUE)
        ax.set_xlabel(f"Test {score_name}")
        show(fig)

    for base, tuned in info["tuning"].items():
        if base in info["fitted"] and tuned in info["fitted"]:
            b0, b1_ = info["fitted"][base], info["fitted"][tuned]
            st.markdown(f"**🔧 Tuning {base}:** test {score_name} {b0['test']:.4f} → **{b1_['test']:.4f}** "
                        f"({b1_['test'] - b0['test']:+.4f})  ·  best CV {score_name} {b1_['cv_mean']:.4f}")
            st.caption("Best settings found: " + ", ".join(f"{k} = {v}" for k, v in b1_["params"].items()))

    names = list(info["fitted"])
    chosen = st.selectbox("Model used for details and predictions", names, index=names.index(info["active"]))
    info["active"] = chosen
    r = info["fitted"][chosen]
    st.markdown(f"### {chosen}")
    y_te, preds = info["y_te"], r["preds"]
    if task == "Regression":
        m1, m2, m3 = st.columns(3)
        m1.metric("R² score (test)", f"{r['test']:.3f}")
        m2.metric("MAE (average error)", f"{r['MAE']:.3f}")
        m3.metric("RMSE", f"{r['RMSE']:.3f}")
        st.caption(f"This model explains about {max(r['test'], 0):.0%} of the variation in `{target}`. 1.0 is perfect.")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_te, preds, alpha=0.5)
        lo, hi = min(np.min(y_te), np.min(preds)), max(np.max(y_te), np.max(preds))
        ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        show(fig)
    else:
        kw = dict(average="weighted", zero_division=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_te, preds):.3f}")
        m2.metric("Precision", f"{precision_score(y_te, preds, **kw):.3f}")
        m3.metric("Recall", f"{recall_score(y_te, preds, **kw):.3f}")
        m4.metric("F1", f"{f1_score(y_te, preds, **kw):.3f}")
        if len(le.classes_) <= 15:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(le.inverse_transform(y_te), le.inverse_transform(preds), ax=ax, cmap="Blues", colorbar=False)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            show(fig)

    # ---- explanation
    st.markdown("### 🔍 Why does the model predict this?")
    if r.get("imp") is None:
        if st.button("Explain this model"):
            with st.spinner("Shuffling each column to see how much the model depends on it..."):
                r["imp"] = importance_table(r["pipe"], info["X_te"], y_te, task)
    imp = r.get("imp")
    if imp is not None:
        st.markdown("**Which columns does the model really use?**")
        show(plot_importance(imp, task))
        st.caption("Each bar shows how much the score falls when that column is shuffled. A long bar means the model depends on it. "
                   "Bars near zero mean it barely matters.")
    can_pdp = task == "Regression" or hasattr(r["pipe"], "predict_proba")
    if can_pdp:
        st.markdown("**How does the prediction change with one column?**")
        default = info["features"].index(imp.iloc[0]["Column"]) if imp is not None else 0
        col = st.selectbox("Column", info["features"], index=default, key="pdp_col")
        cls_idx, label = None, f"Average predicted {target}"
        if task == "Classification":
            cls = st.selectbox("Class", list(le.classes_), key="pdp_cls")
            cls_idx = list(r["pipe"].classes_).index(int(le.transform([cls])[0]))
            label = f"Average probability of {cls}"
        vals, ys = partial_dependence_curve(r["pipe"], info["X_ref"], col, info["plan"], task, cls_idx)
        show(plot_pdp(vals, ys, col, col in info["plan"]["num"], label))
        st.caption("Other columns keep their real values; only this one is moved. A flat line means the column has little effect.")

    buf = io.BytesIO()
    joblib.dump({"pipeline": r["pipe"], "label_encoder": info["le"], "features": info["features"]}, buf)
    st.download_button("⬇️ Download trained model (.joblib)", buf.getvalue(), "model.joblib")


# ----------------------------------------------------------------- predict
def render_predict(target, file_name):
    info = st.session_state.get("info")
    if not info or info["file"] != file_name or info["target"] != target or not info["fitted"]:
        st.info("Train a model in the previous tab first.")
        return

    plan, feats, X_ref, le = info["plan"], info["features"], info["X_ref"], info["le"]
    pipe = info["fitted"][info["active"]]["pipe"]
    st.subheader(f"Predict `{info['target']}`")
    st.caption(f"Using model: **{info['active']}** (change it in the previous tab)")
    mode = st.radio("Input method", ["Enter values", "Upload CSV"], horizontal=True)

    def decode(p):
        return le.inverse_transform(np.asarray(p).astype(int)) if le is not None else p

    if mode == "Enter values":
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(feats):
            with cols[i % 3]:
                if f in plan["num"]:
                    med = X_ref[f].median()
                    values[f] = st.number_input(f, value=float(med) if pd.notna(med) else 0.0, key=f"in_{f}")
                else:
                    options = plan["orders"][f] if f in plan["ordered"] else sorted(X_ref[f].dropna().unique().tolist())
                    values[f] = st.selectbox(f, options, key=f"in_{f}")
        if st.button("Predict"):
            row = align_features(pd.DataFrame([values]), plan)
            out = decode(pipe.predict(row))[0]
            st.success(f"Predicted {info['target']}: **{out:.4f}**" if info["task"] == "Regression" else f"Predicted {info['target']}: **{out}**")
            if info["task"] == "Classification" and hasattr(pipe, "predict_proba"):
                proba = pd.Series(pipe.predict_proba(row)[0], index=decode(pipe.classes_), name="probability").sort_values(ascending=False)
                st.dataframe(proba.head(5).to_frame())
            try:
                eff, label = local_effects(pipe, row, X_ref, plan, info["task"])
                st.markdown("**Why this prediction?**")
                show(plot_effects(eff, label))
                st.caption("Each bar shows how much that input pushed this prediction up (green) or down (red) compared with a typical value. "
                           "It is an approximation: one column is changed at a time.")
            except Exception:
                pass
    else:
        up = st.file_uploader("CSV with the feature columns", type=["csv"], key="new_csv")
        if up is not None:
            new = read_csv_bytes(up.getvalue())
            new.columns = [str(c).strip() for c in new.columns]
            missing = [c for c in feats if c not in new.columns]
            if missing:
                st.error(f"These columns are missing from your file: {missing}")
            else:
                result = new.copy()
                result[f"predicted_{info['target']}"] = decode(pipe.predict(align_features(new[feats], plan)))
                st.dataframe(result.head(50))
                st.download_button("⬇️ Download predictions", result.to_csv(index=False).encode(), "predictions.csv", "text/csv")


# --------------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="CSV → EDA → ML", layout="wide")
    st.title("📊 CSV → EDA → Model → Prediction")

    up = st.sidebar.file_uploader("1. Upload a CSV file", type=["csv"])
    if up is None:
        st.info("👈 Upload a CSV file from the sidebar to begin.")
        return
    try:
        df_raw = load_data(up.getvalue())
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        return

    target = st.sidebar.selectbox("2. Select target column (y)", list(df_raw.columns), index=None, placeholder="Choose y")
    df = render_cleaning(df_raw, up.name)
    if len(df) < 20:
        st.error("Fewer than 20 rows are left after cleaning.")
        return

    task, features, dropped, plan, leak = None, [], {}, make_plan(pd.DataFrame()), None
    if target is not None:
        key = f"{up.name}_{target}"
        y_all = df[target].dropna()
        if y_all.empty:
            st.error("The target column is empty.")
            return
        guess = guess_task(y_all)
        task = st.sidebar.radio("Problem type (auto-detected)", ["Regression", "Classification"],
                                index=0 if guess == "Regression" else 1, key=f"task_{key}")
        if task == "Regression" and not pd.api.types.is_numeric_dtype(y_all):
            st.error(f"`{target}` is text, so it can only be used for Classification.")
            return
        if task == "Classification" and y_all.nunique() > 50:
            st.error(f"`{target}` has {y_all.nunique()} different values. That is too many classes. Try Regression or another column.")
            return

        auto_keep, dropped = auto_select_features(df, target)
        candidates = auto_keep + [c for c, why in dropped.items() if why.startswith("duplicates")]
        leak = detect_leakage_cached(df, target, tuple(candidates), task)
        excluded = leak["auto_exclude"]
        features = st.sidebar.multiselect(
            "3. Feature columns (X) — chosen automatically", [c for c in df.columns if c != target],
            default=[c for c in auto_keep if c not in excluded], key=f"feat_{key}",
        )
        st.sidebar.caption("All other columns are used as X. Useless ones are removed for you; you can still edit the list.")
        if excluded:
            st.sidebar.warning("Left out as possible leakage: " + ", ".join(f"`{c}`" for c in excluded))

        overrides, orders = render_encoding_editor(df, features, key)
        for c in [c for c in features if overrides.get(c) == "Drop"]:
            dropped[c] = "dropped by you in the encoding table"
        features = [c for c in features if overrides.get(c) != "Drop"]
        for c in excluded:
            if c not in features and c not in dropped:
                dropped[c] = "possible data leakage (see the leakage check)"
        plan = make_plan(df[features], overrides, orders) if features else make_plan(pd.DataFrame())

    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🛠️ Prepare & Train", "🔮 Predict"])
    with tab1:
        render_eda(df, target, task, features, plan, dropped)
    with tab2:
        render_train(df, target, task, features, dropped, plan, up.name, leak)
    with tab3:
        render_predict(target, up.name)


if __name__ == "__main__":
    main()  