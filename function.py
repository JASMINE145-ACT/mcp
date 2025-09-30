import json
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
def eda(df: pd.DataFrame, top_k: int = 5):
    """
    自动化 EDA: 输出数据结构、缺失值分析、描述性统计
    :param df: 输入的 pandas DataFrame
    :param top_k: 类别变量展示 Top-K
    :return: dict(json_summary), str(markdown_summary)
    """
    summary = {}

    # 1. 基本结构
    summary["shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
    
    # 2. 每列统计
    cols = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_rate": float(df[col].isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["stats"] = {
                "mean": float(df[col].mean(skipna=True)),
                "std": float(df[col].std(skipna=True)),
                "min": float(df[col].min(skipna=True)),
                "q25": float(df[col].quantile(0.25)),
                "median": float(df[col].median(skipna=True)),
                "q75": float(df[col].quantile(0.75)),
                "max": float(df[col].max(skipna=True)),
            }
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object":
            vc = df[col].value_counts(dropna=True).head(top_k)
            col_info["top_values"] = vc.to_dict()
        cols.append(col_info)
    summary["columns"] = cols

    # JSON 输出
    json_summary = json.dumps(summary, indent=2, ensure_ascii=False)

    # Markdown output (more readable)
    md_lines = []
    md_lines.append(f"### Data Dimensions\n- Rows: {summary['shape']['rows']}\n- Columns: {summary['shape']['cols']}\n")

    md_lines.append("### Column Overview")
    for col in summary["columns"]:
        md_lines.append(f"- **{col['name']}** (dtype={col['dtype']}, missing_rate={col['missing_rate']:.2%})")
        if "stats" in col:
            s = col["stats"]
            md_lines.append(f"  - Numeric stats: mean={s['mean']:.2f}, std={s['std']:.2f}, min={s['min']}, max={s['max']}")
        if "top_values" in col:
            md_lines.append(f"  - Top{top_k} categories: {col['top_values']}")

    markdown_summary = "\n".join(md_lines)

    return summary, markdown_summary

def profile_dataframe(
    df: pd.DataFrame,
    table_name: Optional[str] = None,
    sample_rows_n: int = 5,
    max_examples: int = 5,
    max_topk: int = 20,
    datetime_parse_threshold: float = 0.98,
) -> Dict[str, Any]:
    """
    Produce a compact JSON-able profile for an agent.
    """
    def to_py(v):
        """Make values JSON-serializable."""
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            try:
                return pd.to_datetime(v).isoformat()
            except Exception:
                return str(v)
        if isinstance(v, (pd.Interval,)):
            return str(v)
        if isinstance(v, (set,)):
            return list(v)
        if isinstance(v, (pd.Timedelta, np.timedelta64)):
            return str(v)
        return v

    def schema_hash(dataframe: pd.DataFrame) -> str:
        sig = ";".join([f"{c}:{str(dataframe[c].dtype)}" for c in dataframe.columns])
        return hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]

    def bucket_cardinality(unique_ratio: float) -> str:
        if unique_ratio <= 0.05:
            return "low"
        if unique_ratio <= 0.5:
            return "med"
        return "high"

    def non_null(s: pd.Series) -> pd.Series:
        return s[~s.isna()]

    def estimate_cardinality(s: pd.Series) -> Tuple[int, float]:
        nn = non_null(s)
        n_unique = nn.nunique(dropna=True)
        denom = max(len(nn), 1)
        unique_ratio = float(n_unique) / float(denom)
        return int(n_unique), unique_ratio

    def sample_examples(s: pd.Series, k: int = 5) -> List[Any]:
        nn = non_null(s).drop_duplicates()
        if len(nn) == 0:
            return []
        if len(nn) <= k:
            vals = nn.iloc[:k].tolist()
        else:
            vals = nn.sample(k, random_state=42).tolist()
        # 轻度脱敏/裁剪（避免长文本）
        out = []
        for v in vals:
            v = to_py(v)
            if isinstance(v, str) and len(v) > 120:
                v = v[:117] + "..."
            out.append(v)
        return out

    def try_parse_datetime(s: pd.Series) -> Tuple[float, Optional[str], Optional[str]]:
        # 粗略检测：仅在 object/string/Int64 等可疑类型上尝试解析
        if s.dtype.kind in ("M",):  # already datetime64
            return 1.0, None, "unknown"
        if s.dtype.kind not in ("O", "U", "S", "i", "u"):
            return 0.0, None, None
        coerced = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        parse_rate = float((~coerced.isna() & ~s.isna()).mean())
        if parse_rate == 0.0:
            return 0.0, None, None
        # granularity（粗略）
        gran = "seconds"
        if all(coerced.dt.floor("D") == coerced):
            gran = "days"
        elif all(coerced.dt.floor("H") == coerced):
            gran = "hours"
        elif all(coerced.dt.floor("min") == coerced):
            gran = "minutes"
        return parse_rate, None, gran  # timezone_guess 暂不可稳妥推断

    def infer_logical_type(s: pd.Series, n_unique: int, unique_ratio: float) -> str:
        # binary
        nn = non_null(s)
        # map common yes/no to binary
        if s.dtype.kind in ("b",):
            return "binary"
        lowered = None
        if nn.dtype == object:
            lowered = nn.astype(str).str.lower()
        # candidate binary sets
        binary_sets = [
            {"0", "1"},
            {"true", "false"},
            {"y", "n"},
            {"yes", "no"},
        ]
        if n_unique == 2:
            return "binary"
        if lowered is not None and set(lowered.unique()).issubset({"0", "1", "true", "false", "y", "n", "yes", "no"}):
            return "binary"

        # datetime
        parse_rate, _, _ = try_parse_datetime(s)
        if parse_rate >= datetime_parse_threshold:
            return "datetime"

        # numeric continuous vs categorical vs identifier vs text
        if s.dtype.kind in ("i", "u", "f"):
            if unique_ratio > 0.95:
                return "identifier"  # high-card numeric id
            # decide categorical if small distinct
            if n_unique <= 50 or unique_ratio <= 0.05:
                return "categorical"
            return "numeric_continuous"

        # strings/objects
        if s.dtype.kind in ("O", "U", "S"):
            # identifier-like: high unique ratio + mix of letters/digits
            if unique_ratio > 0.95:
                return "identifier"
            if n_unique <= 50 or unique_ratio <= 0.05:
                return "categorical"
            # heuristic: long average length => free text
            try:
                avg_len = non_null(s).astype(str).str.len().mean()
            except Exception:
                avg_len = None
            if avg_len is not None and avg_len >= 20:
                return "text_freeform"
            return "categorical"  # default fallback for strings

        return "mixed_type"

    def numeric_stats(s: pd.Series) -> Dict[str, Any]:
        nn = pd.to_numeric(non_null(s), errors="coerce")
        nn = nn[np.isfinite(nn)]
        if len(nn) == 0:
            return {}
        q = nn.quantile([0.0, 0.05, 0.5, 0.95, 1.0])
        return {
            "min": to_py(q.loc[0.0]),
            "p5": to_py(q.loc[0.05]),
            "p50": to_py(q.loc[0.5]),
            "p95": to_py(q.loc[0.95]),
            "max": to_py(q.loc[1.0]),
            "mean": to_py(nn.mean()),
            "std": to_py(nn.std(ddof=1)) if len(nn) > 1 else 0.0,
        }

    def topk_categories(s: pd.Series, k: int = 20) -> List[List[Any]]:
        vc = non_null(s).astype(str).value_counts(normalize=True)
        vc = vc.head(k)
        return [[idx, float(round(val, 6))] for idx, val in vc.items()]

    def suggest_outlier_rule(s: pd.Series) -> Optional[str]:
        if s.dtype.kind not in ("i", "u", "f"):
            return None
        return "p99.5 + 3*IQR"

    def profile_datetime(s: pd.Series) -> Dict[str, Any]:
        parse_rate, tz_guess, gran = try_parse_datetime(s)
        parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        info = {
            "parse_rate": round(parse_rate, 6),
            "timezone_guess": tz_guess,
            "granularity": gran,
        }
        if parse_rate > 0:
            info["min"] = to_py(parsed.min())
            info["max"] = to_py(parsed.max())
        return info

    # ---------- table-level ----------
    table = {
        "name": table_name or "dataframe",
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "primary_key_guess": [],
        "timestamp_col_guess": None,
        "problems": [],
        "profile_version": {"schema_hash": schema_hash(df), "ts": pd.Timestamp.utcnow().isoformat()},
    }

    columns_meta: List[Dict[str, Any]] = []
    id_candidates = []
    datetime_candidates = []

    # ---------- per-column ----------
    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean())
        n_unique, unique_ratio = estimate_cardinality(s)

        dtype_physical = s.dtype.name
        # normalize to a small set
        if s.dtype.kind in ("i", "u"):
            dtype_physical = "int64"
        elif s.dtype.kind == "f":
            dtype_physical = "float64"
        elif s.dtype.kind == "b":
            dtype_physical = "bool"
        elif s.dtype.kind in ("M",):
            dtype_physical = "datetime64"
        else:
            dtype_physical = "string"

        dtype_logical = infer_logical_type(s, n_unique, unique_ratio)
        role = "feature"
        if dtype_logical == "identifier":
            role = "id"
            id_candidates.append(col)
        if dtype_logical == "datetime":
            role = "time"
            datetime_candidates.append(col)

        info: Dict[str, Any] = {
            "name": col,
            "dtype_physical": dtype_physical,
            "dtype_logical": dtype_logical,
            "role": role,
            "missing_rate": round(missing_rate, 6),
            "cardinality": {
                "n_unique": int(n_unique),
                "unique_ratio": float(round(unique_ratio, 6)),
                "bucket": bucket_cardinality(unique_ratio),
            },
            "examples": sample_examples(s, max_examples),
        }

        if dtype_logical in ("numeric_continuous",):
            info["stats"] = numeric_stats(s)
            info["outlier_flag_rule"] = suggest_outlier_rule(s)
        elif dtype_logical in ("categorical", "binary"):
            info["topk"] = topk_categories(s, max_topk)
        elif dtype_logical == "datetime":
            info.update(profile_datetime(s))

        columns_meta.append(info)

    # primary key guess: unique & no missing
    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean())
        n_unique, unique_ratio = estimate_cardinality(s)
        if missing_rate == 0.0 and math.isclose(unique_ratio, 1.0, rel_tol=0, abs_tol=1e-9):
            table["primary_key_guess"].append(col)

    table["timestamp_col_guess"] = datetime_candidates[0] if datetime_candidates else None

    # ---------- relationships (lightweight) ----------
    relationships = {"numeric_corr_top": [], "categorical_assoc_top": [], "leakage_warnings": []}
    numeric_df = df.select_dtypes(include=[np.number])
    try:
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr(method="spearman").abs()
            pairs = []
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((cols[i], cols[j], float(round(corr.iloc[i, j], 6))))
            pairs.sort(key=lambda x: x[2], reverse=True)
            relationships["numeric_corr_top"] = [list(p) for p in pairs[:10]]
    except Exception:
        pass  # keep empty if anything fails

    # ---------- PII warnings (very light regex heuristics) ----------
    pii_cols = []
    for col in df.columns:
        if df[col].dtype.kind not in ("O", "U", "S"):
            continue
        ser = df[col].astype(str)
        # simple patterns
        has_email = ser.str.contains(r"[^@\s]+@[^@\s]+\.[^@\s]+", regex=True, na=False).mean() > 0.2
        has_phone = ser.str.contains(r"\+?\d[\d\-\s()]{6,}", regex=True, na=False).mean() > 0.2
        if has_email:
            pii_cols.append(f"{col}:email")
        if has_phone:
            pii_cols.append(f"{col}:phone")
    pii_warnings = pii_cols

    # ---------- sample rows (representative, small) ----------
    n = min(max(sample_rows_n, 0), len(df))
    if n > 0:
        sample_df = df.sample(n, random_state=42) if len(df) > n else df.copy()
        sample_rows = []
        for _, row in sample_df.iterrows():
            sample_rows.append({k: to_py(v) for k, v in row.items()})
    else:
        sample_rows = []

    # assemble
    meta = {
        "table": table,
        "columns": columns_meta,
        "relationships": relationships,
        "pii_warnings": pii_warnings,
        "sample_rows": sample_rows,
    }
    return meta


def profile_dataframe_json(
    df: pd.DataFrame,
    table_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience wrapper returning a JSON string (UTF-8, pretty)."""
    meta = profile_dataframe(df, table_name=table_name, **kwargs)
    return json.dumps(meta, ensure_ascii=False, indent=2)


def profile_dataframe_simple(
    df: pd.DataFrame,
    table_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a simplified data overview with English output
    """
    def to_py(v):
        """Make values JSON-serializable."""
        if isinstance(v, (np.generic,)):
            return v.item()
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            try:
                return pd.to_datetime(v).isoformat()
            except Exception:
                return str(v)
        if isinstance(v, (pd.Interval,)):
            return str(v)
        if isinstance(v, (set,)):
            return list(v)
        if isinstance(v, (pd.Timedelta, np.timedelta64)):
            return str(v)
        return v

    def show_sample_data(s: pd.Series, n_samples: int = 2) -> List[Any]:
        """Show sample data from column"""
        nn = s[~s.isna()].drop_duplicates()
        if len(nn) == 0:
            return []
        
        if len(nn) <= n_samples:
            samples = nn.iloc[:n_samples].tolist()
        else:
            samples = nn.sample(n_samples, random_state=42).tolist()
        
        # Convert to JSON serializable format
        result = []
        for sample in samples:
            sample = to_py(sample)
            if isinstance(sample, str) and len(sample) > 50:
                sample = sample[:47] + "..."
            result.append(sample)
        return result

    # Table-level information
    table_info = {
        "dataset_name": table_name or "dataframe",
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns))
    }
    
    # Column overview
    columns_overview = []
    
    for col in df.columns:
        s = df[col]
        samples = show_sample_data(s, n_samples=2)
        
        col_info = {
            "column_name": col,
            "sample_values": samples
        }
        columns_overview.append(col_info)
    
    # Simplified output format
    simple_profile = {
        "dataset_info": table_info,
        "column_overview": columns_overview
    }
    
    return simple_profile


def profile_dataframe_simple_json(
    df: pd.DataFrame,
    table_name: Optional[str] = None,
) -> str:
    """简化版profile的JSON输出"""
    simple_meta = profile_dataframe_simple(df, table_name=table_name)
    return json.dumps(simple_meta, ensure_ascii=False, indent=2)
