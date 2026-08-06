import numpy as np
import pandas as pd
from utils import progressBar


class DriftPredictor:
    def __init__(
        self,
        training_data_path=None,
        downsample="200ms",
        time_bin_count=30,
        time_bin_base=None,
        time_bin_max=120,
        points_per_square=10000,
        points_per_angle_bin=200,
        percentiles=(95,),
        max_attempts_multiplier=10,
        random_state=None,
        debug=False,
        debug_max_points=20000,
        load_path=None,
    ):
        self.downsample = downsample
        self.time_bin_count = int(time_bin_count)
        self.time_bin_base = time_bin_base
        self.time_bin_max = time_bin_max
        self.points_per_square = int(points_per_square)
        self.points_per_angle_bin = int(points_per_angle_bin)
        self.percentiles = self._normalize_percentiles(percentiles)
        self.max_attempts_multiplier = int(max_attempts_multiplier)
        self.random_state = random_state
        self.debug = bool(debug)
        self.debug_max_points = int(debug_max_points)
        self._debug_points = [] if self.debug else None

        if load_path is not None:
            self._load_state(load_path)
            return
        if training_data_path is None:
            raise ValueError("training_data_path is required when load_path is not set")

        self.data = pd.read_csv(training_data_path, engine="pyarrow")
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data = self.data.set_index("timestamp").sort_index().dropna()
        self.data = self.data.resample(downsample).first()

        # truncate data before 2023-09-30 10:00:00
        print("WARNING: Truncating data before 2023-10-01 06:00:00")
        self.data = self.data.loc[self.data.index >= '2023-10-01 06:00:00']

        num_cols = self.data.select_dtypes(include=["float64", "int64"]).columns
        self.data[num_cols] = self.data[num_cols].astype(np.float32)

        self.times_s, self.stokes = self._extract_times_stokes()
        self.time_edges = self._build_time_edges()
        self.bin_map = self._build_bins()
        if self.debug:
            self._plot_debug_bins()

    @staticmethod
    def _angle_between(stokes_a, stokes_b):
        norm_a = np.linalg.norm(stokes_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(stokes_b, axis=1, keepdims=True)
        safe_a = stokes_a / np.where(norm_a == 0, 1.0, norm_a)
        safe_b = stokes_b / np.where(norm_b == 0, 1.0, norm_b)
        dot = np.clip(np.sum(safe_a * safe_b, axis=1), -1.0, 1.0)
        return np.arccos(dot)

    @staticmethod
    def _normalize_percentiles(percentiles):
        perc = tuple(sorted(set(np.atleast_1d(percentiles).astype(float))))
        if not perc:
            raise ValueError("percentiles must include at least one value")
        if any((p <= 0.0 or p >= 100.0) for p in perc):
            raise ValueError("percentiles must be between 0 and 100")
        return perc

    def _extract_times_stokes(self):
        times = self.data.index.to_numpy()
        times_s = times.astype("datetime64[ns]").astype(np.int64) * 1e-9
        stokes = self.data[["S1", "S2", "S3"]].to_numpy(copy=False)
        return times_s.astype(np.float64), stokes.astype(np.float32)

    def _build_time_edges(self):
        print("WARNING: hardcoding bin edges...")
        edges = np.array([0] + list(np.arange(0.15, 2.05, .1)) + list(range(3,20)) + list(range(21, 101, 3)))
        print(edges)
        return edges

        if self.time_bin_count < 2:
            raise ValueError("time_bin_count must be >= 2")
        if self.time_bin_base is None:
            base = pd.to_timedelta(self.downsample).total_seconds()
        else:
            base = float(self.time_bin_base)
        power = np.log(self.time_bin_max / base) / np.log(self.time_bin_count-1)
        edges = base * (np.arange(self.time_bin_count, dtype=np.float64) ** power)
        print(edges)
        return edges

    def _sample_square(self, lo_prev, hi_prev, lo_curr, hi_curr):
        rng = np.random.default_rng(self.random_state)
        n = len(self.times_s)
        if n < 3:
            raise ValueError(f"Not enough data to sample from... {n} < 3")
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        max_attempts = self.points_per_square * self.max_attempts_multiplier
        prev_angles = []
        curr_angles = []
        collected = 0
        attempts = 0

        while collected < self.points_per_square and attempts < max_attempts:
            batch = min(self.points_per_square - collected, 2048)
            j_idx = rng.integers(1, n - 1, size=batch)
            dt_prev = rng.uniform(lo_prev, hi_prev, size=batch)
            dt_curr = rng.uniform(lo_curr, hi_curr, size=batch)

            t_j = self.times_s[j_idx]
            t_prev_target = t_j - dt_prev
            t_next_target = t_j + dt_curr

            i_idx = np.searchsorted(self.times_s, t_prev_target, side="right") - 1
            k_idx = np.searchsorted(self.times_s, t_next_target, side="left")

            valid = (i_idx >= 0) & (k_idx < n) & (i_idx < j_idx) & (j_idx < k_idx)
            if not np.any(valid):
                attempts += batch
                continue

            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            k_idx = k_idx[valid]

            stokes_i = self.stokes[i_idx]
            stokes_j = self.stokes[j_idx]
            stokes_k = self.stokes[k_idx]

            prev = self._angle_between(stokes_i, stokes_j)
            curr = self._angle_between(stokes_j, stokes_k)

            prev_angles.append(prev)
            curr_angles.append(curr)
            if self._debug_points is not None and len(self._debug_points) < self.debug_max_points:
                remaining = self.debug_max_points - len(self._debug_points)
                take = min(remaining, len(prev))
                if take > 0:
                    dt_prev_valid = dt_prev[valid][:take]
                    dt_curr_valid = dt_curr[valid][:take]
                    self._debug_points.extend(zip(dt_prev_valid, dt_curr_valid))
            collected += len(prev)
            attempts += batch

        if not prev_angles:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        assert len(np.concatenate(prev_angles).astype(np.float32, copy=False)) == self.points_per_square, f"len(np.concatenate(prev_angles).astype(np.float32, copy=False)) != self.points_per_square... {len(np.concatenate(prev_angles).astype(np.float32, copy=False))} != {self.points_per_square}"
        prev_all = np.concatenate(prev_angles).astype(np.float32, copy=False)[: self.points_per_square]
        curr_all = np.concatenate(curr_angles).astype(np.float32, copy=False)[: self.points_per_square]
        return prev_all, curr_all

    def _reduce_curr_angles(self, curr_angles):
        return [
            float(np.nanpercentile(curr_angles, percentile))
            for percentile in self.percentiles
        ]

    def _build_bins(self):
        bin_map = {}
        edges = self.time_edges
        for i in range(len(edges) - 1):
            lo_prev, hi_prev = edges[i], edges[i + 1]
            for j in range(len(edges) - 1):
                lo_curr, hi_curr = edges[j], edges[j + 1]
                prev_angles, curr_angles = self._sample_square(
                    lo_prev, hi_prev, lo_curr, hi_curr
                )
                if len(prev_angles) < self.points_per_angle_bin:
                    raise ValueError(f"Not enough points to build bin... {len(prev_angles)} < {self.points_per_angle_bin}")

                order = np.argsort(prev_angles)
                prev_sorted = prev_angles[order]
                curr_sorted = curr_angles[order]

                bin_count = int(np.ceil(len(prev_sorted) / self.points_per_angle_bin))
                if bin_count == 0:
                    raise ValueError("Bin count is 0")

                upper_edges = []
                values = [[] for _ in self.percentiles]
                for b in range(bin_count):
                    start = b * self.points_per_angle_bin
                    end = min(start + self.points_per_angle_bin, len(prev_sorted))
                    prev_chunk = prev_sorted[start:end]
                    curr_chunk = curr_sorted[start:end]
                    upper_edges.append(prev_chunk[-1])
                    reduced = self._reduce_curr_angles(curr_chunk)
                    for idx, value in enumerate(reduced):
                        values[idx].append(value)

                bin_map[(i, j)] = {
                    "upper_edges": np.asarray(upper_edges, dtype=np.float32),
                    "values": {
                        percentile: np.asarray(values[idx], dtype=np.float32)
                        for idx, percentile in enumerate(self.percentiles)
                    },
                }
                if self.debug and False:
                    print(
                        "bin",
                        (i, j),
                        "prev_time",
                        (lo_prev, hi_prev),
                        "curr_time",
                        (lo_curr, hi_curr),
                    )
                    for b_idx, edge in enumerate(bin_map[(i, j)]["upper_edges"]):
                        percs = {
                            p: float(bin_map[(i, j)]["values"][p][b_idx])
                            for p in self.percentiles
                        }
                        print("  angle_bin", b_idx, "upper_edge", float(edge), "percentiles", percs)
            progressBar(i*(len(edges) - 1) + j, (len(edges) - 1)**2)

        progressBar(1, 1, done=True)
        return bin_map

    def _plot_debug_bins(self):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except Exception:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        edges = self.time_edges
        max_bins = 0
        for (i, j), entry in self.bin_map.items():
            count = len(entry["upper_edges"])
            max_bins = max(max_bins, count)
            x0, x1 = edges[i], edges[i + 1]
            y0, y1 = edges[j], edges[j + 1]
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="C0",
                    alpha=0.15 + 0.55 * (count / max(1, max_bins)),
                    edgecolor="none",
                )
            )

        for e in edges:
            ax.axvline(e, color="k", lw=0.4, alpha=0.3)
            ax.axhline(e, color="k", lw=0.4, alpha=0.3)

        if self._debug_points:
            pts = np.array(self._debug_points, dtype=np.float32)
            ax.scatter(pts[:, 0], pts[:, 1], s=6, c="C1", alpha=0.5)

        ax.set_xlabel("prev_drift_time")
        ax.set_ylabel("curr_drift_time")
        ax.set_title("Time bins with sampled points")
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()

    def save(self, filepath):
        np.savez_compressed(
            filepath,
            time_edges=self.time_edges,
            percentiles=np.asarray(self.percentiles, dtype=np.float32),
            bin_map=self.bin_map,
        )

    def _load_state(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.time_edges = data["time_edges"]
        self.percentiles = tuple(data["percentiles"].astype(float).tolist())
        self.bin_map = data["bin_map"].item()

    def _time_bin_index(self, t):
        idx = int(np.searchsorted(self.time_edges, t, side="right") - 1)
        if idx < 0:
            return None
        if idx >= len(self.time_edges) - 1:
            print(f"Time bin index {idx} is out of bounds for time {t}")
            return idx
        return idx

    def predict(self, prev_angle_diff, prev_drift_time, curr_drift_time, percentile):
        i = self._time_bin_index(prev_drift_time)
        j = self._time_bin_index(curr_drift_time)
        if i is None or j is None:
            print(f"No bin index found for prev_drift_time {prev_drift_time} or curr_drift_time {curr_drift_time}")
            return np.nan
        
        if i >= len(self.time_edges) - 1 or j >= len(self.time_edges) - 1:
            print(f"Time bin index {i} or {j} is out of bounds for time {prev_drift_time} or {curr_drift_time}")
            return 1.0 # Predict high drift

        entry = self.bin_map.get((i, j))
        if not entry:
            print(f"No entry found for (i, j) = ({i}, {j})")
            return np.nan

        edges = entry["upper_edges"]
        if not percentile in entry["values"]:
            raise ValueError(f"Predictor called with percentile {percentile} not found in bin map. Valid percentiles: {entry['values'].keys()}")
        values = entry["values"].get(float(percentile))
        if edges.size == 0:
            return np.nan
        if values is None or values.size == 0:
            return np.nan

        idx = int(np.searchsorted(edges, prev_angle_diff, side="right"))
        idx = int(np.clip(idx, 0, len(values) - 1))
        ret = float(values[idx])
        assert np.isfinite(ret), f"Predictor returned {ret} for prev_angle_diff {prev_angle_diff}, prev_drift_time {prev_drift_time}, curr_drift_time {curr_drift_time}, percentile {percentile}"
        return ret

    def __call__(self, prev_angle_diff, prev_drift_time, curr_drift_time, percentile):
        return self.predict(prev_angle_diff, prev_drift_time, curr_drift_time, percentile)

if __name__ == "__main__":
    model_path = "drift_prediction_model_no_overfit.npz"
    training_path = "data.csv"

    drift_predictor = DriftPredictor(
        training_data_path=training_path,
        downsample='200ms',
        time_bin_count=100,
        time_bin_max=180,
        points_per_square=1_000_000,
        points_per_angle_bin=10000,
        percentiles=[50, 90, 95, 97, 98, 99],
        random_state=0,
        debug=True
    )
    drift_predictor.save(model_path)