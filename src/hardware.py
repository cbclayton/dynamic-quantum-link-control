import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from pathlib import Path
import pickle as pkl

# Constants
SOURCE_RATE = 76_142_000 # 76.142 MHz
FID_CHECK_DURATION = 0.044  # 44ms
FIBER_LOSS_DB = 18 # 18 dB loss over the fiber


def poly_fit_func(x, y, d):
    coeffs = np.polyfit(x, y, d)
    f = lambda x, coeffs=coeffs: np.polyval(coeffs, x)
    return f

def poly_fit_func_inverse(x, y, d, lo, hi):
    coeffs = np.polyfit(x, y, d)
    p = np.poly1d(coeffs)
    def p_inv(y):
        solns = (p-y).roots
        solns = solns[solns >= lo]
        solns = solns[solns <= hi]
        if len(solns) == 0:
            print(f"p_inv({y:.3f}) has {len(solns)} solutions in range [{lo}, {hi}]: {(p-y).roots}...", end=" ")
            solns = (p-y).roots
            if max(solns) < lo:
                print(f"returning {lo}")
                return lo
            elif min(solns) > hi:
                print(f"returning {hi}")
                return hi
            else:
                raise NotImplementedError(f"p_inv({y:.3f}) has {len(solns)} solutions in range [{lo}, {hi}]: {(p-y).roots}.")
        assert len(solns) == 1, f"p_inv({y:.3f}) has {len(solns)} solutions in range [{lo}, {hi}]: {(p-y).roots}."
        return solns[0]
    return p_inv

# --- Linear (least squares) helpers ---
def _linear_ls_fit(x, y):
    """
    Unweighted linear least-squares fit y ≈ m*x + b.
    Returns (m, b) as floats.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 2:
        raise ValueError("Need at least 2 finite points for linear fit.")
    A = np.vstack([x[good], np.ones_like(x[good])]).T
    m, b = np.linalg.lstsq(A, y[good], rcond=None)[0]
    return float(m), float(b)

def _line_func(m: float, b: float):
    return lambda x, m=m, b=b: m * np.asarray(x, dtype=float) + b

def _line_inverse_func(m: float, b: float, lo: float, hi: float):
    if not np.isfinite(m) or abs(m) < 1e-15:
        raise ValueError("Cannot invert line with zero/invalid slope.")
    def inv(y):
        y = np.asarray(y, dtype=float)
        x = (y - b) / m
        return np.clip(x, lo, hi)
    return inv

# --- Source/detector fidelity model from tomography results ---
#
# Data source: produced by `data/MIRA_source_characterization/sourceAnalysis/analysis_script_tomo.py`
# which writes `tomo_results_th=*.pkl` as a list of dicts:
#   [{'pump_powers': [...], 'fids': [...], 'fid_errors': [...], 'counts': [...], 'count_errors': [...], 'int_times': [...]}, ...]
# _TOMO_RESULTS_PATH = (
#     Path(__file__).resolve().parent.parent
#     / "data"
#     / "Qunnect_data"
#     / "qu-src_characterization_results.pkl"
# )
_TOMO_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "MIRA_source_characterization"
    / "sourceAnalysis"
    / "tomo_results_th=12.pkl"
)

def _get_first_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return k
    raise KeyError(f"None of the candidate keys {candidates} found in dict keys {list(d.keys())}")

REMOVED_POINTS = []
def _load_tomo_results_raw(path: Path = _TOMO_RESULTS_PATH):
    results = pkl.loads(path.read_bytes())
    if not isinstance(results, list) or len(results) == 0:
        raise ValueError(f"Unexpected tomo results format in {path}.")
    for i, d in enumerate(results):
    #     if 300 in d['pump_powers']:
    #         for k, v in d.items():
    #             d[k] = v[:-1]
        REMOVED_POINTS.append({})
        for k, v in d.items():
            REMOVED_POINTS[i][k] = v[:2]
            d[k] = v[2:]
    return results


def _combine_tomo_results(results, value_key: str, error_key: str):
    """
    Combine datasets at the same pump power using inverse-variance weighting.

    Args:
        results: list[dict] with keys including:
            - 'pump_powers'
            - value_key
            - error_key

    Returns:
        pump_powers_mw, values, errors (all np.ndarray shape (N,))
    """
    triples = []
    for res in results:
        for pp, v, e in zip(res["pump_powers"], res[value_key], res[error_key]):
            triples.append((float(pp), float(v), float(e)))

    if not triples:
        raise ValueError("No data found in tomo results.")

    pump_powers = np.array(sorted({pp for pp, _, _ in triples}), dtype=float)
    vals_out = np.empty_like(pump_powers, dtype=float)
    errs = np.empty_like(pump_powers, dtype=float)

    for i, pp in enumerate(pump_powers):
        vals = [(v, e) for p, v, e in triples if p == pp]
        vs = np.array([v[0] for v in vals], dtype=float)
        es = np.array([v[1] for v in vals], dtype=float)

        good = np.isfinite(es) & (es > 0)
        if np.any(good):
            w = 1.0 / (es[good] ** 2)
            vals_out[i] = float(np.sum(w * vs[good]) / np.sum(w))
            errs[i] = float(np.sqrt(1.0 / np.sum(w)))
        else:
            vals_out[i] = float(np.mean(vs))
            errs[i] = float(np.std(vs) if len(vs) > 1 else np.nan)

    return pump_powers, vals_out, errs


def _normalize_tomo_results_to_reference(
    raw_results,
    ref_dataset_idxs=(0, 1),
    target_dataset_idxs=(2, 3),
    value_key: str = "fids",
    error_key: str = "fid_errors",
):
    """
    Normalize target datasets to match reference datasets using a purely multiplicative transform:
        f_norm = a * f
    where a is fit (weighted least squares) using overlapping pump powers.

    Returns:
        normalized_results: list[dict] same shape as input, with normalized target datasets
        params: dict[idx] -> {'a': a, 'b': 0.0}
    """
    raw = raw_results
    if max(ref_dataset_idxs + target_dataset_idxs) >= len(raw):
        raise ValueError("Requested dataset index out of range for tomo results.")

    # Build reference curve: mean of the reference datasets at each pump power (weighted by their errors)
    ref_triples = []
    for i in ref_dataset_idxs:
        r = raw[i]
        for pp, f, e in zip(r["pump_powers"], r[value_key], r[error_key]):
            ref_triples.append((float(pp), float(f), float(e)))

    ref_pps = sorted({pp for pp, _, _ in ref_triples})
    ref_map = {}
    for pp in ref_pps:
        vals = [(f, e) for p, f, e in ref_triples if p == pp]
        fs = np.array([v[0] for v in vals], dtype=float)
        es = np.array([v[1] for v in vals], dtype=float)
        good = np.isfinite(es) & (es > 0)
        if np.any(good):
            w = 1.0 / (es[good] ** 2)
            f_ref = float(np.sum(w * fs[good]) / np.sum(w))
            e_ref = float(np.sqrt(1.0 / np.sum(w)))
        else:
            f_ref = float(np.mean(fs))
            e_ref = float(np.std(fs) if len(fs) > 1 else np.nan)
        ref_map[pp] = (f_ref, e_ref)

    normalized = []
    params = {}
    for idx, res in enumerate(raw):
        pp = np.asarray(res["pump_powers"], dtype=float)
        f = np.asarray(res[value_key], dtype=float)
        e = np.asarray(res[error_key], dtype=float)

        if idx not in target_dataset_idxs:
            # Keep all fields, unchanged.
            normalized.append({k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in res.items()})
            continue

        # Overlap points where reference exists
        overlap = np.array([p in ref_map for p in pp], dtype=bool)
        if not np.any(overlap):
            # Nothing to fit; leave unchanged
            params[idx] = {"a": 1.0, "b": 0.0}
            normalized.append({k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in res.items()})
            continue

        pp_o = pp[overlap]
        f_o = f[overlap]
        e_o = e[overlap]
        f_ref = np.array([ref_map[p][0] for p in pp_o], dtype=float)
        e_ref = np.array([ref_map[p][1] for p in pp_o], dtype=float)

        # Weight by combined uncertainty (target + reference)
        sigma2 = np.where(np.isfinite(e_o) & np.isfinite(e_ref), e_o**2 + e_ref**2, np.nan)
        good = np.isfinite(sigma2) & (sigma2 > 0)
        if np.any(good):
            w = 1.0 / sigma2[good]
        else:
            w = np.ones_like(f_o)
            good = np.ones_like(f_o, dtype=bool)

        # Weighted least squares for a with b=0: minimize sum w*(a*f - f_ref)^2
        fg = f_o[good]
        yg = f_ref[good]
        wg = w
        denom = float(np.sum(wg * (fg ** 2)))
        if denom <= 0 or not np.isfinite(denom):
            a = 1.0
        else:
            a = float(np.sum(wg * fg * yg) / denom)
        params[idx] = {"a": a, "b": 0.0}

        f_n = a * f
        e_n = np.abs(a) * e
        out = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in res.items()}
        out[value_key] = list(f_n)
        out[error_key] = list(e_n)
        normalized.append(out)

    return normalized, params


def _apply_multiplicative_scale(results, idx_to_params, value_key: str, error_key: str, inverse: bool = False):
    """
    Apply per-dataset scaling to a value + its error:
        v_scaled = a * v
        e_scaled = |a| * e

    If inverse=True, apply:
        v_scaled = v / a
        e_scaled = e / |a|
    """
    out = []
    for i, res in enumerate(results):
        r = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in res.items()}
        if i in idx_to_params:
            a = float(idx_to_params[i].get("a", 1.0))
            if inverse:
                r[value_key] = list(np.asarray(res[value_key], dtype=float) / a)
                r[error_key] = list(np.asarray(res[error_key], dtype=float) / np.abs(a))
            else:
                r[value_key] = list(a * np.asarray(res[value_key], dtype=float))
                r[error_key] = list(np.abs(a) * np.asarray(res[error_key], dtype=float))
        out.append(r)
    return out


def _load_tomo_fidelity_curve(path: Path = _TOMO_RESULTS_PATH):
    """
    Returns:
        pump_powers_mw: np.ndarray shape (N,)
        fidelities: np.ndarray shape (N,)
        errors: np.ndarray shape (N,)  (1-sigma; weighted-averaged if multiple datasets per power)
    """
    raw = _load_tomo_results_raw(path)
    fid_error_key = _get_first_key(raw[0], ("fid_errors", "errors"))
    # Fidelity curves are left unmodified (no per-dataset normalization).
    # We keep these return values for backward compatibility with older callers.
    normalized = [{k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in res.items()} for res in raw]
    norm_params = {}
    pump_powers, fids, errs = _combine_tomo_results(raw, value_key="fids", error_key=fid_error_key)
    return pump_powers, fids, errs, raw, normalized, norm_params, fid_error_key


def _derive_rate_results(raw_results):
    """
    Add derived rate fields to each dataset:
        rate = counts / (INT_TIME / 1e12)
    where INT_TIME is taken per-point from `int_times` in the tomo results.
    """
    out = []
    for res in raw_results:
        r = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in res.items()}
        if "counts" not in r or "count_errors" not in r or "int_times" not in r:
            raise KeyError("Missing one of required keys: 'counts', 'count_errors', 'int_times'")

        counts = np.asarray(r["counts"], dtype=float)
        count_err = np.asarray(r["count_errors"], dtype=float)
        int_times = np.asarray(r["int_times"], dtype=float)  # stored as strings in the pickle

        rate = counts / (int_times / 1e12)
        rate_err = count_err / (int_times / 1e12)

        r["rates"] = list(rate)
        r["rate_errors"] = list(rate_err)
        out.append(r)
    return out


def _load_tomo_rate_curve(
    path: Path = _TOMO_RESULTS_PATH,
    normalize_to_reference: bool = False,
    ref_dataset_idxs=(0, 1),
    target_dataset_idxs=(2, 3),
):
    """
    Returns:
        pump_powers_mw: np.ndarray shape (N,)
        rates: np.ndarray shape (N,)  (counts / second)
        rate_errors: np.ndarray shape (N,)  (1-sigma; weighted-averaged if multiple datasets per power)
    """
    raw = _load_tomo_results_raw(path)
    rate_results = _derive_rate_results(raw)
    rate_norm_params = {}
    if normalize_to_reference:
        # Normalize rate curves directly (datasets 2/3) to match datasets 0/1 via a multiplicative factor:
        #   rate_norm = a * rate
        rate_results, rate_norm_params = _normalize_tomo_results_to_reference(
            rate_results,
            ref_dataset_idxs=ref_dataset_idxs,
            target_dataset_idxs=target_dataset_idxs,
            value_key="rates",
            error_key="rate_errors",
        )
    pump_powers, rates, errs = _combine_tomo_results(rate_results, value_key="rates", error_key="rate_errors")
    return pump_powers, rates, errs, raw, rate_results, rate_norm_params


(
    _PUMP_POWERS_MW,
    _FIDS_TOMO,
    _FIDS_TOMO_ERR,
    _RAW_RESULTS,
    _NORM_RESULTS,
    _NORM_PARAMS,
    _FID_ERROR_KEY,
) = _load_tomo_fidelity_curve()

try:
    # Rate is computed from counts + int_times (no per-dataset normalization in the model).
    # Any normalization is only applied for plotting in __main__.
    (
        _RATE_PUMP_POWERS_MW,
        _RATES_TOMO,
        _RATES_TOMO_ERR,
        _RAW_RESULTS_FOR_RATE,
        _RATE_RESULTS,
        _RATE_NORM_PARAMS,
    ) = _load_tomo_rate_curve(normalize_to_reference=False)
except KeyError:
    # Backward compatibility: older tomo_results files may not contain required fields for rate.
    _RATE_PUMP_POWERS_MW, _RATES_TOMO, _RATES_TOMO_ERR, _RAW_RESULTS_FOR_RATE, _RATE_RESULTS, _RATE_NORM_PARAMS = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

# Set min/max pump powers from measured data
MIN_PUMP_POWER_MW = float(np.min(_PUMP_POWERS_MW))
MAX_PUMP_POWER_MW = float(np.max(_PUMP_POWERS_MW))

# Linear fits (one per dataset), and an "overall" fitted curve defined as the average slope/intercept.
# Fidelity: fit each of the four raw datasets independently.
_FID_LINES = []
_FID_LINE_PARAMS = []  # list[(m, b)]
for i, res in enumerate(_RAW_RESULTS):
    pp = np.asarray(res["pump_powers"], dtype=float)
    f = np.asarray(res["fids"], dtype=float)
    m, b = _linear_ls_fit(pp, f)
    _FID_LINE_PARAMS.append((m, b))
    _FID_LINES.append(_line_func(m, b))

_FID_M_AVG = float(np.mean([m for m, _ in _FID_LINE_PARAMS]))
_FID_B_AVG = float(np.mean([b for _, b in _FID_LINE_PARAMS]))
_fidelity_sd_poly = _line_func(_FID_M_AVG, _FID_B_AVG)  # keep name for backwards compatibility
_fidelity_sd_poly_inverse = _line_inverse_func(_FID_M_AVG, _FID_B_AVG, MIN_PUMP_POWER_MW, MAX_PUMP_POWER_MW)

# Rate: fit each dataset's derived rates independently (if available), then average slopes/intercepts.
_RATE_LINES = []
_RATE_LINE_PARAMS = []  # list[(dataset_idx, m, b)]
if _RATE_PUMP_POWERS_MW is not None:
    for i, res in enumerate(_RATE_RESULTS):
        pp = np.asarray(res["pump_powers"], dtype=float)
        rate = np.asarray(res["rates"], dtype=float)
        m, b = _linear_ls_fit(pp, rate)
        _RATE_LINE_PARAMS.append((i, m, b))
        _RATE_LINES.append(_line_func(m, b))

    if len(_RATE_LINE_PARAMS) > 0:
        _RATE_M_AVG = float(np.mean([m for _, m, _ in _RATE_LINE_PARAMS]))
        _RATE_B_AVG = float(np.mean([b for _, _, b in _RATE_LINE_PARAMS]))
        _rate_sd_poly = _line_func(_RATE_M_AVG, _RATE_B_AVG)  # keep name for backwards compatibility
        _rate_sd_poly_inverse = _line_inverse_func(_RATE_M_AVG, _RATE_B_AVG, MIN_PUMP_POWER_MW, MAX_PUMP_POWER_MW)

        # Clip range for inversions
        _MIN_RATE = float(np.min([np.min(np.asarray(res["rates"], dtype=float)) for res in _RATE_RESULTS if "rates" in res]))
        _MAX_RATE = float(np.max([np.max(np.asarray(res["rates"], dtype=float)) for res in _RATE_RESULTS if "rates" in res]))
    else:
        _rate_sd_poly = None
        _rate_sd_poly_inverse = None
        _MIN_RATE = None
        _MAX_RATE = None
else:
    _rate_sd_poly = None
    _rate_sd_poly_inverse = None
    _MIN_RATE = None
    _MAX_RATE = None


def power_from_fid(fid):
    power = _fidelity_sd_poly_inverse(fid)
    return float(np.clip(power, MIN_PUMP_POWER_MW, MAX_PUMP_POWER_MW))

def power_from_rate(rate):
    if _rate_sd_poly_inverse is None or _MIN_RATE is None or _MAX_RATE is None:
        raise RuntimeError("Rate model is not available (missing rate fields in tomo_results).")
    rate = np.clip(rate, _MIN_RATE, _MAX_RATE)
    power = _rate_sd_poly_inverse(rate)
    return float(np.clip(power, MIN_PUMP_POWER_MW, MAX_PUMP_POWER_MW))

def rate_to_fid(rate):
    power = power_from_rate(rate)
    return fidelity_sd([power])

def fid_to_rate(fid):
    power = power_from_fid(fid)
    return _rate_sd_poly(power)


# Source/detector fidelity
def fidelity_sd(link_params):
    """
    Predict source/detector fidelity as a function of pump power (mW),
    using an average of per-dataset linear least-squares fits to tomography results.
    """
    assert len(link_params) == 1
    pump_power_mw = link_params[0]
    assert MIN_PUMP_POWER_MW <= pump_power_mw <= MAX_PUMP_POWER_MW
    F = float(_fidelity_sd_poly(pump_power_mw))
    assert 0.0 <= F <= 1.0
    return F

# Source/detector p_success
def p_success_sd(link_params):
    assert len(link_params) == 1
    pump_power_mw = link_params[0]
    assert MIN_PUMP_POWER_MW <= pump_power_mw <= MAX_PUMP_POWER_MW, f"{MIN_PUMP_POWER_MW} <= {pump_power_mw} <= {MAX_PUMP_POWER_MW}"
    assert _rate_sd_poly is not None
    rate = float(_rate_sd_poly(pump_power_mw))
    return rate / SOURCE_RATE

# Transmission p_success
def p_success_transmission():
    total_loss_db = FIBER_LOSS_DB
    p_success = 10**(-total_loss_db/10)
    return p_success


if __name__ == "__main__":
    # Plot:
    # - unmodified dataset curves (dotted)
    # Notes:
    # - Fidelity curves are left unmodified (no per-dataset normalization).
    # - Rate normalization (datasets 2/3 only) is applied ONLY for plotting.
    
    normalize = True

    plt.rcParams["font.size"] = 11
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Linux Libertine O']

    fig, (ax_fid, ax_cnt) = plt.subplots(2, 1, sharex=True, figsize=(5, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    colors = list(mcolors.TABLEAU_COLORS.values())[1:]

    # Unmodified curves (dotted)
    for i, res in enumerate(_RAW_RESULTS):
        pp = np.asarray(res["pump_powers"], dtype=float)
        f = np.asarray(res["fids"], dtype=float)
        e = np.asarray(res[_FID_ERROR_KEY], dtype=float)
        order = np.argsort(pp)
        ax_fid.errorbar(
            pp[order],
            f[order],
            yerr=e[order],
            fmt=markers[i % len(markers)],
            alpha=.85,
            linewidth=1.4,
            markersize=4.0,
            color=colors[i % len(colors)],
            label="_nolegend_",
        )

    if not normalize:
        for i, res in enumerate(REMOVED_POINTS[:4]):
            pp = np.asarray(res["pump_powers"], dtype=float)
            f = np.asarray(res["fids"], dtype=float)
            e = np.asarray(res[_FID_ERROR_KEY], dtype=float)

            order = np.argsort(pp)
            ax_fid.errorbar(
            pp[order],
            f[order],
            yerr=e[order],
            fmt=markers[i % len(markers)],
            alpha=0.85,
            linewidth=1.2,
            markersize=4.0,
            color=colors[i % len(colors)],
            label="_nolegend_",
        )


    xfit = np.linspace(MIN_PUMP_POWER_MW, MAX_PUMP_POWER_MW, 400)
    # Per-dataset linear fits + average fit
    # for i, (m, b) in enumerate(_FID_LINE_PARAMS):
    #     ax_fid.plot(
    #         xfit,
    #         m * xfit + b,
    #         color=colors[i % len(colors)],
    #         linewidth=1.6,
    #         alpha=1,
    #         linestyle="--",
    #         label="_nolegend_",
    #     )
    fid_avg_line, = ax_fid.plot(
        xfit,
        _fidelity_sd_poly(xfit),
        color="black",
        linewidth=3.0,
        label="_nolegend_",
    )

    ax_fid.set_ylabel("Fidelity")
    ax_fid.set_ylim(None, 1.0)
    ax_fid.grid(True, alpha=0.25)
    # ax_fid.minorticks_on()
    fid_proxy_dataset = Line2D([0], [0], color="black", linestyle="--", linewidth=2.0)
    fid_proxy_avg = Line2D([0], [0], color="black", linestyle="-", linewidth=3.0)
    # ax_fid.legend(
    #     handles=[fid_proxy_dataset, fid_proxy_avg],
    #     labels=["Per-Dataset Fit", "Average Fit"],
    # )

    # --- Rate plot (apply normalization only for plotting) ---
    has_rate = _RATE_PUMP_POWERS_MW is not None and _RATE_RESULTS is not None
    if has_rate:
        # Apply multiplicative normalization to the POINTS first (datasets 2/3 only),
        # then fit to those scaled points for the plotted curve.
        rate_plot_results, _rate_plot_params = _normalize_tomo_results_to_reference(
            _RATE_RESULTS,
            ref_dataset_idxs=(0, 1),
            target_dataset_idxs=(2, 3) if normalize else (),
            value_key="rates",
            error_key="rate_errors",
        )

        if normalize:
            norm = max([max(res['rates']) for res in rate_plot_results])
            for res in rate_plot_results:
                for key in ["rates", "rate_errors"]:
                    res[key] /= norm

        for i, res in enumerate(rate_plot_results):
            pp = np.asarray(res["pump_powers"], dtype=float)
            rate = np.asarray(res["rates"], dtype=float)
            rate_err = np.asarray(res["rate_errors"], dtype=float)

            order = np.argsort(pp)
            ax_cnt.errorbar(
                pp[order],
                rate[order],
                yerr=rate_err[order],
                fmt=markers[i % len(markers)],
                alpha=0.85,
                linewidth=1.2,
                markersize=4.0,
                color=colors[i % len(colors)],
                label="_nolegend_",
            )

        # Fit per-dataset lines to the (possibly scaled) plotted points, then average.
        _rate_plot_line_params = []
        for i, res in enumerate(rate_plot_results):
            pp = np.asarray(res["pump_powers"], dtype=float)
            rate = np.asarray(res["rates"], dtype=float)
            m, b = _linear_ls_fit(pp, rate)
            _rate_plot_line_params.append((i, m, b))

        if len(_rate_plot_line_params) > 0:
            m_avg = float(np.mean([m for _, m, _ in _rate_plot_line_params]))
            b_avg = float(np.mean([b for _, _, b in _rate_plot_line_params]))
            rate_avg_line, = ax_cnt.plot(
                xfit,
                m_avg * xfit + b_avg,
                color="black",
                linewidth=3.0,
                label="_nolegend_",
            )
        if not normalize:
            for i, res in enumerate(REMOVED_POINTS[:4]):
                pp = np.asarray(res["pump_powers"], dtype=float)
                rate = np.asarray(res["counts"], dtype=float) / np.asarray(res["int_times"], dtype=float) * 1e12
                rate_err = np.asarray(res["count_errors"], dtype=float) / np.asarray(res["int_times"], dtype=float) * 1e12

                order = np.argsort(pp)
                ax_cnt.errorbar(
                pp[order],
                rate[order],
                yerr=rate_err[order],
                fmt=markers[i % len(markers)],
                alpha=0.85,
                linewidth=1.2,
                markersize=4.0,
                color=colors[i % len(colors)],
                label="_nolegend_",
            )
    else:
        ax_cnt.text(
            0.02,
            0.95,
            "Rate not available in tomo_results file (expected keys: 'counts', 'count_errors', 'int_times').",
            transform=ax_cnt.transAxes,
            va="top",
        )
    
    # ax_cnt.plot(xfit, [p_success_sd([x])*SOURCE_RATE for x in xfit], color="r", linewidth=2.0, label="rate poly fit")

    ax_cnt.set_xlabel("Pump Power (mW)")
    ax_cnt.set_xlim(0, None)
    ax_cnt.set_ylabel("Normalized Entanglement\nGeneration Rate" if normalize else "Entanglement\nGeneration Rate (pairs/s)")
    ax_cnt.grid(True, alpha=0.25)
    # ax_cnt.minorticks_on()
    if has_rate:
        rate_proxy_dataset = Line2D([0], [0], color="black", linestyle="--", linewidth=2.0)
        rate_proxy_avg = Line2D([0], [0], color="black", linestyle="-", linewidth=3.0)
        # ax_cnt.legend(
        #     handles=[rate_proxy_dataset, rate_proxy_avg],
        #     labels=["Per-Dataset Fit", "Average Fit"],
        # )

    fig.tight_layout()
    plt.show()