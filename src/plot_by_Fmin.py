"""
Plot average entanglement distribution rate vs F_min for:
- Adaptive protocol (rows where `adaptive` is True)
- A configurable set of static protocols, where a protocol is defined by
  (`source_fidelity`, `fid_check_freq`).

Data source: `results/results.csv` (relative to this file by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.size"] = 13

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O']



@dataclass(frozen=True)
class StaticProtocol:
    source_fidelity: float
    fid_check_freq: float
    label: str | None = None

    def display_label(self) -> str:
        if self.label:
            return self.label
        return r"Static: $F_\text{sd}$=" + f"{self.source_fidelity:g}"
        return f"Static: F_source={self.source_fidelity:g}, fid_check_freq={self.fid_check_freq:g}s"


def _coerce_adaptive_to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "t", "yes", "y"])


def _value_mask(series: pd.Series, value: object) -> pd.Series:
    """
    Return boolean mask where `series` matches `value`.

    - Numeric series uses `np.isclose` (for robust float CSV comparisons)
    - Non-numeric uses equality
    - If `value` is NaN (float), matches NaNs in the series
    """
    if isinstance(value, float) and np.isnan(value):
        return series.isna()

    if pd.api.types.is_numeric_dtype(series.dtype) and isinstance(value, (int, float, np.number)):
        return np.isclose(series.astype(float), float(value), rtol=0, atol=1e-12)

    return series == value


def _apply_constant_filters(df: pd.DataFrame, constant_filters: dict[str, object]) -> pd.DataFrame:
    """
    Apply constant filters like {"F_trigger": 0.98, "timeout": 55}.

    Any filter with a column not present is an error (helps catch typos).
    """
    out = df
    for col, value in constant_filters.items():
        if col not in out.columns:
            raise ValueError(f"Constant filter column '{col}' not found in results.csv")
        out = out[_value_mask(out[col], value)]
    return out


def _load_results_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    # Robust coercions; adaptive rows may have blank static-parameter columns.
    for col in ["F_min", "source_fidelity", "fid_check_freq", "average_rate"]:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    if "adaptive" not in df.columns:
        raise ValueError(f"Expected an 'adaptive' column in {csv_path}")
    df.loc[:, "adaptive"] = _coerce_adaptive_to_bool(df["adaptive"])

    required = {"F_min", "average_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}")

    return df


def _filter_static_protocol(df: pd.DataFrame, protocol: StaticProtocol) -> pd.DataFrame:
    # Use isclose to avoid float representation surprises in CSV.
    fid_match = np.isclose(df["source_fidelity"], protocol.source_fidelity, rtol=0, atol=1e-12)
    freq_match = np.isclose(df["fid_check_freq"], protocol.fid_check_freq, rtol=0, atol=1e-12)
    return df[(df["adaptive"] == False) & fid_match & freq_match]  # noqa: E712


def _unique_rows_per_fmin(
    df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """
    Enforce uniqueness per F_min.

    Raises if >=2 rows exist for any given F_min in `df`.
    """
    d = df.dropna(subset=["F_min", "average_rate"]).copy()
    if len(d) == 0:
        return d

    counts = d.groupby("F_min").size()
    ambiguous_fmins = counts[counts >= 2].index.to_list()
    if ambiguous_fmins:
        examples = (
            d[d["F_min"].isin(ambiguous_fmins)]
            .sort_values(["F_min"])
            .loc[:, [c for c in ["time", "F_min", "average_rate", "source_fidelity", "fid_check_freq", "adaptive"] if c in d.columns]]
        )
        raise ValueError(
            "Found multiple rows matching the same selection conditions.\n"
            f"Curve: {label}\n"
            f"Ambiguous F_min values: {ambiguous_fmins}\n"
            f"Matching rows (subset of columns):\n{examples.to_string(index=False)}"
        )

    # Exactly one row per F_min -> keep that row.
    return d.sort_values("F_min")


def _resolve_default_csv_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path("results") / "results.csv",  # run from `src/`
        Path("src") / "results" / "results.csv",  # run from repo root
        here / "results" / "results.csv",  # explicit relative-to-this-file
    ]
    for p in candidates:
        if p.exists():
            return p
    # Default to the "most correct" location even if it doesn't exist, so the
    # error message is informative.
    return here / "results" / "results.csv"


def plot_average_rate_vs_fmin(
    df: pd.DataFrame,
    static_protocols: Iterable[StaticProtocol],
    *,
    static_constant_filters: dict[str, object] | None = None,
    adaptive_constant_filters: dict[str, object] | None = None,
    title: str = "Average rate vs $F_{min}$",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    # tableau = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    colors = sns.color_palette("rocket_r", n_colors=13)[2:]
    adaptive_color = colors[-2]

    if static_constant_filters is None:
        static_constant_filters = {}
    # By default, apply the same constant filters to adaptive rows too. If you
    # want an "unconstrained" adaptive curve, pass adaptive_constant_filters={}
    # explicitly.
    if adaptive_constant_filters is None:
        adaptive_constant_filters = dict(static_constant_filters)

    adaptive_df = df[df["adaptive"] == True]  # noqa: E712
    adaptive_df = _apply_constant_filters(adaptive_df, adaptive_constant_filters)
    adaptive_curve = _unique_rows_per_fmin(adaptive_df, label="Adaptive")
    ax.plot(
        adaptive_curve["F_min"] if len(adaptive_curve) else [],
        adaptive_curve["average_rate"] if len(adaptive_curve) else [],
        "-o",
        linewidth=2,
        color=adaptive_color,
        label="Adaptive",
    )

    markers = ["s", "^", "D", "v", "P", "X"]
    static_base_df = _apply_constant_filters(df, static_constant_filters)

    for i, proto in enumerate(static_protocols):
        proto_df = _filter_static_protocol(static_base_df, proto)
        curve = _unique_rows_per_fmin(proto_df, label=proto.display_label())
        if len(curve) == 0:
            # Skip silently; change to `print(...)` if you want warnings.
            continue

        ax.plot(
            curve["F_min"],
            curve["average_rate"],
            linestyle=":",
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=proto.display_label(),
        )

    ax.set_xlabel(r"Minimum Fidelity Requirement ($F_\text{min}$)")
    ax.set_ylabel("Mean Entanglement Distribution Rate (Hz)")
    # plt.title(title)

    
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.25)
    # ax.grid(True, which="minor", alpha=0.10)
    ax.minorticks_on()

    # ... use ax.plot(...) instead of plt.plot(...)
    # ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # Edit this list to choose which static protocols to show.
    STATIC_PROTOCOLS: list[StaticProtocol] = [
        StaticProtocol(source_fidelity=0.85, fid_check_freq=2.0),
        StaticProtocol(source_fidelity=0.87, fid_check_freq=2.0),
        StaticProtocol(source_fidelity=0.89, fid_check_freq=2.0),
        StaticProtocol(source_fidelity=0.91, fid_check_freq=2.0),
        StaticProtocol(source_fidelity=0.93, fid_check_freq=2.0),
    ]

    # Hold other simulation parameters constant by filtering rows before plotting.
    # Add/remove entries here as needed (column names must match results.csv headers).
    STATIC_CONSTANT_FILTERS: dict[str, object] = {
        # Example:
        "timeout": 55,
        "F_trigger": 0.98,
        "F_target": 0.99,
        "resample_ms": 100,
        "label": 'normal',
    }
    # Adaptive rows may want a different set of constant filters than static
    # protocol sweeps. If you want the adaptive curve *not* to be constrained by
    # these, set this to {}.
    ADAPTIVE_CONSTANT_FILTERS: dict[str, object] = {
        "resample_ms": 100,
        "label": 'normal',
    }

    csv_path = _resolve_default_csv_path()
    df = _load_results_csv(csv_path)
    plot_average_rate_vs_fmin(
        df,
        STATIC_PROTOCOLS,
        static_constant_filters=STATIC_CONSTANT_FILTERS,
        adaptive_constant_filters=ADAPTIVE_CONSTANT_FILTERS,
        title="Adaptive vs static protocols (mean rate vs $F_{min}$)",
    )

