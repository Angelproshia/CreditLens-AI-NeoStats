from __future__ import annotations
import pandas as pd
from .config import DATA_PATH, TARGET, PREFERRED_FEATURES


def load_data(path=DATA_PATH, sample_size: int | None = None) -> pd.DataFrame:
    path = str(path)
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {path}")
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42, stratify=df[TARGET])
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    selected = [c for c in PREFERRED_FEATURES if c in df.columns]
    if len(selected) < 8:
        selected = [c for c in df.columns if c not in {TARGET, "SK_ID_CURR"}][:30]
    return selected


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    }).sort_values("missing_pct", ascending=False)

