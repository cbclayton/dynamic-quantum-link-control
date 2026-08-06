"""
Plot results from `results.csv`.

Default plot:
- x-axis: `fid_check_freq`
- y-axis: `average_rate`
- one curve per `source_fidelity`
- adaptive strategy: the row where `source_fidelity`, `fid_check_freq`, and `F_trigger` are missing
  is plotted as a dotted horizontal reference line (using the chosen y-axis column).

To change what gets plotted, edit the CONFIG block below.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.size"] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O']

@dataclass(frozen=True)
class PlotConfig:
    # ---- What to plot (easy to change) ----
    x_col: str = "fid_check_freq"
    y_col: str = "average_rate"
    curve_col: str = "source_fidelity"

    # Optional: only plot points where x_col is within [x_col_min, x_col_max] (inclusive).
    # Use None for no bound.
    x_col_min: float | None = None
    x_col_max: float | None = None

    # Optional: only plot curves where curve_col is within [curve_col_min, curve_col_max] (inclusive).
    # Use None for no bound.
    curve_col_min: float | None = None
    curve_col_max: float | None = None

    # Dimensions that you want to easily swap between x-axis / curve / held-constant
    dimension_cols: tuple[str, ...] = ("fid_check_freq", "source_fidelity", "F_trigger", "F_target", "F_min", "label", "resample_ms")

    # Holds for any dimension not used as x_col or curve_col (filled in main() from CLI).
    # Values are strings or floats; numeric columns are compared with np.isclose.
    hold_values: dict[str, float | str] = field(default_factory=dict)

    # Holds applied to adaptive row selection (separate from static curve holds).
    # Only these columns are used to pick the unique adaptive row.
    adaptive_hold_values: dict[str, float | str] = field(default_factory=dict)

    # ---- How to detect the adaptive strategy row(s) ----
    adaptive_null_cols: tuple[str, ...] = ("source_fidelity", "fid_check_freq", "F_trigger")
    adaptive_label: str = "Adaptive protocol"

    # ---- Labels & styling ----
    title: str | None = "Entanglement distribution rate vs fidelity check frequency"
    x_label: str | None = "Fidelity check frequency (s)"
    y_label: str | None = "Average entanglement distribution rate (Hz)"
    curve_label_fmt: str = "{curve_col}={curve_val:g}"

    marker: str = "o"
    linewidth: float = 1.75
    grid: bool = True
    legend: bool = True
    xscale: str = "auto"  # "auto" | "linear" | "log"


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. Available: {list(df.columns)}")


def _coerce_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            # Use .loc assignment to avoid pandas Copy-on-Write / chained-assignment warnings.
            df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")


def _detect_adaptive_rows(df: pd.DataFrame, null_cols: Iterable[str]) -> pd.Series:
    # If the CSV includes an explicit adaptive flag, trust it.
    if "adaptive" in df.columns:
        s = df["adaptive"]
        if s.dtype == bool:
            return s == True  # noqa: E712
        s_norm = s.astype(str).str.strip().str.lower()
        return s_norm.isin({"true", "t", "1", "yes", "y"})

    null_cols = tuple(null_cols)
    _require_columns(df, null_cols)
    mask = pd.Series(True, index=df.index)
    for c in null_cols:
        # Treat both NaNs and blank/whitespace strings as "null" for adaptive detection.
        s = df[c]
        s_str = s.astype("string")
        is_empty = s_str.isna() | (s_str.str.strip() == "")
        mask &= is_empty
    return mask


def _is_numeric_series(s: pd.Series) -> bool:
    s_num = pd.to_numeric(s, errors="coerce")
    return bool(s_num.notna().any())


def _is_empty_cell_series(s: pd.Series) -> pd.Series:
    """
    True where the cell is "empty": NaN/None or blank/whitespace string.
    """
    s_str = s.astype("string")
    return s_str.isna() | (s_str.str.strip() == "")


def _filter_on_holds(df: pd.DataFrame, hold_values: dict[str, float | str]) -> pd.DataFrame:
    """
    Filter rows where df[col] == value for each hold.
    - Numeric columns: uses np.isclose
    - Non-numeric columns: string equality
    """
    if not hold_values:
        return df

    out = df
    for col, desired in hold_values.items():
        if col not in out.columns:
            raise KeyError(f"Hold column '{col}' not found in dataframe columns.")

        s = out[col]
        if _is_numeric_series(s):
            s_num = pd.to_numeric(s, errors="coerce")
            try:
                desired_f = float(desired)  # type: ignore[arg-type]
            except (TypeError, ValueError) as e:
                raise ValueError(f"Hold value for numeric column '{col}' must be numeric; got {desired!r}") from e
            mask = np.isclose(s_num.to_numpy(), desired_f, rtol=1e-9, atol=1e-12, equal_nan=False)
            out = out.loc[mask]
        else:
            out = out.loc[s.astype(str) == str(desired)]

    return out


def _filter_x_range(
    df: pd.DataFrame,
    *,
    x_col: str,
    x_min: float | None,
    x_max: float | None,
) -> pd.DataFrame:
    if x_min is None and x_max is None:
        return df
    if x_col not in df.columns:
        raise KeyError(f"x_col '{x_col}' not found in dataframe columns.")

    xs = pd.to_numeric(df[x_col], errors="coerce")
    mask = xs.notna()
    if x_min is not None:
        mask &= xs >= float(x_min)
    if x_max is not None:
        mask &= xs <= float(x_max)
    return df.loc[mask]


def _filter_curve_range(
    df: pd.DataFrame,
    *,
    curve_col: str,
    curve_min: float | None,
    curve_max: float | None,
) -> pd.DataFrame:
    if curve_min is None and curve_max is None:
        return df
    if curve_col not in df.columns:
        raise KeyError(f"curve_col '{curve_col}' not found in dataframe columns.")

    cs = pd.to_numeric(df[curve_col], errors="coerce")
    mask = cs.notna()
    if curve_min is not None:
        mask &= cs >= float(curve_min)
    if curve_max is not None:
        mask &= cs <= float(curve_max)
    return df.loc[mask]


def _infer_or_require_holds(
    df: pd.DataFrame,
    *,
    dimension_cols: Iterable[str],
    x_col: str,
    curve_col: str,
    user_holds: dict[str, str],
) -> dict[str, float | str]:
    """
    Among dimension_cols, enforce that every dim not in {x_col, curve_col} is held constant.
    - If user provided a hold for that column, use it.
    - Else if there is exactly one unique non-null value in data, infer it.
    - Else raise with a helpful message showing available values.
    """
    dim_cols = tuple(dimension_cols)
    dim_set = set(dim_cols)

    # Fail fast on invalid inputs instead of silently ignoring them.
    unknown_hold_cols = sorted(set(user_holds) - dim_set)
    if unknown_hold_cols:
        raise ValueError(
            "Unknown hold column(s): "
            f"{unknown_hold_cols}. Allowed dimension columns: {list(dim_cols)}"
        )
    conflicting_hold_cols = sorted(set(user_holds) & {x_col, curve_col})
    if conflicting_hold_cols:
        raise ValueError(
            f"Hold column(s) {conflicting_hold_cols} conflict with x_col/curve_col. "
            "You cannot hold the dimension you're plotting on an axis or using as a curve."
        )

    used = {x_col, curve_col}
    remaining = [c for c in dim_cols if c not in used]

    holds: dict[str, float | str] = {}
    for col in remaining:
        if col in user_holds:
            holds[col] = user_holds[col]
            continue

        if col not in df.columns:
            raise KeyError(f"Dimension column '{col}' missing from CSV.")

        s = df[col]
        if _is_numeric_series(s):
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            uniq = np.sort(s_num.unique())
            if len(uniq) == 1:
                holds[col] = float(uniq[0])
            else:
                preview = ", ".join(map(lambda x: f"{x:g}", uniq[:12]))
                more = "" if len(uniq) <= 12 else f", ... ({len(uniq)} values)"
                raise ValueError(
                    f"You selected x='{x_col}' and curve='{curve_col}', so '{col}' must be held constant.\n"
                    f"Provide it via --hold {col}=<value>. Available values: {preview}{more}"
                )
        else:
            uniq = sorted(s.dropna().astype(str).unique().tolist())
            if len(uniq) == 1:
                holds[col] = uniq[0]
            else:
                preview = ", ".join(uniq[:12])
                more = "" if len(uniq) <= 12 else f", ... ({len(uniq)} values)"
                raise ValueError(
                    f"You selected x='{x_col}' and curve='{curve_col}', so '{col}' must be held constant.\n"
                    f"Provide it via --hold {col}=<value>. Available values: {preview}{more}"
                )

    return holds


@dataclass(frozen=True)
class RunConfig:
    """
    All user-editable knobs for running this script without CLI arguments.
    """

    # Data / outputs
    csv_path: str | Path | None = None
    save_path: str | Path | None = None
    show: bool = True

    # Plot config + which dimensions to hold constant
    plot: PlotConfig = field(default_factory=PlotConfig)
    holds: dict[str, float | str] = field(default_factory=dict)
    adaptive_holds: dict[str, float | str] = field(default_factory=dict)


def plot_results(df: pd.DataFrame, cfg: PlotConfig) -> tuple[plt.Figure, plt.Axes]:
    # Avoid mutating a view of an external DataFrame (and keep behavior predictable).
    df = df.copy()
    _require_columns(df, (cfg.x_col, cfg.y_col, cfg.curve_col))
    _coerce_numeric_inplace(df, (cfg.x_col, cfg.y_col, cfg.curve_col))

    adaptive_mask = _detect_adaptive_rows(df, cfg.adaptive_null_cols)
    adaptive_df = df.loc[adaptive_mask]
    static_df = df.loc[~adaptive_mask]

    # Apply constant-hold filters (only to non-adaptive rows)
    static_df = _filter_on_holds(static_df, cfg.hold_values)

    # Clean rows needed for curve plotting
    static_df = static_df.dropna(subset=[cfg.x_col, cfg.y_col, cfg.curve_col]).copy()

    # Apply x-axis range filter (only affects point plots)
    static_df = _filter_x_range(
        static_df,
        x_col=cfg.x_col,
        x_min=cfg.x_col_min,
        x_max=cfg.x_col_max,
    )

    # Apply curve range filter (drops curves outside range)
    static_df = _filter_curve_range(
        static_df,
        curve_col=cfg.curve_col,
        curve_min=cfg.curve_col_min,
        curve_max=cfg.curve_col_max,
    )

    fig, ax = plt.subplots(figsize=(4, 3.5))

    default_colors = sns.color_palette("rocket_r", n_colors=12)[2:-1]
    night_colors = sns.color_palette(palette='twilight', n_colors=15)[2:-4]
    day_colors = sns.color_palette(palette='afmhot_r', n_colors=18)[5:15]
    
    colors = default_colors
    adaptive_color = colors[-1]
    curve_markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

    # One curve per curve_col value
    for i, (curve_val, g) in enumerate(static_df.groupby(cfg.curve_col, sort=True)):
        g = g.sort_values(cfg.x_col)
        label = cfg.curve_label_fmt.format(curve_col=cfg.curve_col, curve_val=curve_val)
        ax.plot(
            g[cfg.x_col].to_numpy(),
            g[cfg.y_col].to_numpy(),
            color=colors[i % len(colors)],
            linestyle=":",
            marker=curve_markers[i % len(curve_markers)],
            ms=4.5,
            linewidth=cfg.linewidth,
            label=label,
        )

    # Adaptive horizontal line (if present)
    if len(adaptive_df) > 0:
        # Filter adaptive rows using adaptive_hold_values, with a simple sentinel:
        # "__EMPTY__" means "this cell must be empty/blank/NaN".
        adaptive_filtered = adaptive_df
        for col, desired in cfg.adaptive_hold_values.items():
            if desired == "__EMPTY__":
                if col not in adaptive_filtered.columns:
                    raise KeyError(f"Adaptive hold column '{col}' not found in dataframe columns.")
                adaptive_filtered = adaptive_filtered.loc[_is_empty_cell_series(adaptive_filtered[col])]
            else:
                adaptive_filtered = _filter_on_holds(adaptive_filtered, {col: desired})
        if len(adaptive_filtered) > 1:
            raise ValueError(
                f"Adaptive rows are not unique after applying adaptive holds {cfg.adaptive_hold_values}. "
                f"Expected 0 or 1 matching adaptive row, found {len(adaptive_filtered)}."
            )
        if len(adaptive_filtered) == 1:
            y_val = pd.to_numeric(adaptive_filtered.iloc[0][cfg.y_col], errors="coerce")
            if pd.isna(y_val):
                raise ValueError(
                    f"Matched an adaptive row after applying adaptive holds {cfg.adaptive_hold_values}, "
                    f"but '{cfg.y_col}' is NaN."
                )
            y = float(y_val)
            ax.axhline(y=y, color=adaptive_color, linestyle="-", linewidth=3.0, label=cfg.adaptive_label)

    if cfg.title:
        ax.set_title(cfg.title)
    ax.set_xlabel(cfg.x_label or cfg.x_col)
    ax.set_ylabel(cfg.y_label or cfg.y_col)

    if cfg.grid:
        ax.grid(True, which="major", alpha=0.25)
        # ax.grid(True, which="minor", alpha=0.10)

    # Auto log-scale if x is positive and spans multiple unique values
    if cfg.xscale == "log" or cfg.xscale == "auto":
        xs = pd.to_numeric(static_df[cfg.x_col], errors="coerce").dropna()
        uniq = np.sort(xs.unique())
        if cfg.xscale == "log":
            ax.set_xscale("log")
        elif len(uniq) > 1 and np.all(uniq > 0):
            ax.set_xscale("log")

    if cfg.legend:
        ax.legend()
    
    ax.set_ylim(0, 83)
    ax.minorticks_on()
    plt.tight_layout()

    return fig, ax


def _default_results_path() -> Path:
    # Prefer `src/results.csv` when running from repo root; otherwise fall back to local `results.csv`.
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    candidates = [
        repo_root / "src" / "results" / "results.csv",
        here.parent / "results.csv",
        Path.cwd() / "results.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main(run: RunConfig) -> None:
    cfg = run.plot
    if cfg.x_col == cfg.curve_col:
        raise ValueError("x_col and curve_col must be different.")

    csv_path = Path(run.csv_path) if run.csv_path is not None else _default_results_path()
    df = pd.read_csv(csv_path)

    # Validate user-provided column selections early (clearer errors than downstream failures).
    dim_set = set(cfg.dimension_cols)
    if cfg.x_col not in dim_set:
        raise ValueError(f"x_col='{cfg.x_col}' must be one of {list(cfg.dimension_cols)}")
    if cfg.curve_col not in dim_set:
        raise ValueError(f"curve_col='{cfg.curve_col}' must be one of {list(cfg.dimension_cols)}")
    _require_columns(df, (cfg.x_col, cfg.y_col, cfg.curve_col))

    # Determine which of the remaining 2 dimension columns must be held constant and at what values.
    user_holds_raw = {k: str(v) for k, v in run.holds.items()}
    user_adaptive_holds_raw = {k: str(v) for k, v in run.adaptive_holds.items()}
    static_for_hold_inference = df.loc[~_detect_adaptive_rows(df, cfg.adaptive_null_cols)].copy()
    hold_values = _infer_or_require_holds(
        static_for_hold_inference,
        dimension_cols=cfg.dimension_cols,
        x_col=cfg.x_col,
        curve_col=cfg.curve_col,
        user_holds=user_holds_raw,
    )
    # Validate adaptive holds (no inference; only what the user provides).
    adaptive_hold_values = _infer_or_require_holds(
        static_for_hold_inference,
        dimension_cols=cfg.dimension_cols,
        x_col=cfg.x_col,
        curve_col=cfg.curve_col,
        user_holds=user_adaptive_holds_raw,
    )

    # Build a final PlotConfig with the inferred/validated holds injected.
    cfg = PlotConfig(
        x_col=cfg.x_col,
        y_col=cfg.y_col,
        curve_col=cfg.curve_col,
        x_col_min=cfg.x_col_min,
        x_col_max=cfg.x_col_max,
        curve_col_min=cfg.curve_col_min,
        curve_col_max=cfg.curve_col_max,
        dimension_cols=cfg.dimension_cols,
        hold_values=hold_values,
        adaptive_hold_values=adaptive_hold_values,
        adaptive_null_cols=cfg.adaptive_null_cols,
        adaptive_label=cfg.adaptive_label,
        title=cfg.title,
        x_label=cfg.x_label,
        y_label=cfg.y_label,
        curve_label_fmt=cfg.curve_label_fmt,
        marker=cfg.marker,
        linewidth=cfg.linewidth,
        grid=cfg.grid,
        legend=cfg.legend,
        xscale=cfg.xscale,
    )

    fig, _ax = plot_results(df, cfg)

    if run.save_path:
        fig.savefig(run.save_path, dpi=200, bbox_inches="tight")
    if run.show:
        plt.show()


if __name__ == "__main__":
    for label in ['normal', 'night', 'day']:
        # -------------------- EDIT THESE VALUES --------------------
        # Pick which of the 4 dimensions to use for x-axis and curves:
        X_COL = "source_fidelity"       # one of: fid_check_freq, source_fidelity, F_trigger, F_target
        CURVE_COL = "fid_check_freq"  # one of: fid_check_freq, source_fidelity, F_trigger, F_target

        # Optional: only plot points where X_COL/CURVE_COL is within this inclusive range.
        # Use None for no bound.
        X_COL_MIN: float | None = None
        X_COL_MAX: float | None = .9
        CURVE_COL_MIN: float | None = None
        CURVE_COL_MAX: float | None = 20

        # Choose the metric for y-axis:
        Y_COL = "average_rate"

        # Hold some dimensions constant (only needed if they vary in your CSV).
        # label = 'normal'
        F_min = 0.85
        resample_ms = 1000
        HOLDS: dict[str, float | str] = {
            "F_trigger": 0.98,
            "F_target": 0.99,
            "F_min": F_min,
            "label": label,
            "resample_ms": resample_ms
        }

        # Holds used ONLY to select the unique adaptive row (separate from HOLDS above).
        # This avoids conditioning on dimensions that are intentionally blank in adaptive rows.
        ADAPTIVE_HOLDS: dict[str, float | str] = {
            "F_min": F_min,
            "label": label,
            "resample_ms": resample_ms,
            # Convention: "__EMPTY__" means "this cell must be empty/blank/NaN" for adaptive-row selection.
            "F_trigger": "__EMPTY__",
            "F_target": 1.0
        }

        # Optional: override labels/title, or just keep defaults.
        TITLE = f"{label} {resample_ms}" #"Entanglement distribution rate vs fidelity check frequency"
        X_LABEL = r"$F_\text{sd}$" # f"Fidelity check frequency (s)"
        Y_LABEL = "Mean entanglement distribution rate (Hz)"

        X_SCALE = "linear"

        # Optional: where to load/save.
        CSV_PATH: str | Path | None = 'results/results.csv'  # None => auto-detect (prefers src/results.csv)
        SAVE_PATH: str | Path | None = None  # e.g. \"plots/figure.png\"
        SHOW = True
        # -----------------------------------------------------------

        run_cfg = RunConfig(
            csv_path=CSV_PATH,
            save_path=SAVE_PATH,
            show=SHOW,
            holds=HOLDS,
            adaptive_holds=ADAPTIVE_HOLDS,
            plot=PlotConfig(
                x_col=X_COL,
                y_col=Y_COL,
                curve_col=CURVE_COL,
                x_col_min=X_COL_MIN,
                x_col_max=X_COL_MAX,
                curve_col_min=CURVE_COL_MIN,
                curve_col_max=CURVE_COL_MAX,
                title=TITLE,
                x_label=X_LABEL,
                y_label=Y_LABEL,
                xscale=X_SCALE,
            ),
        )

        main(run_cfg)
