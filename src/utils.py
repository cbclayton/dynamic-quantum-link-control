from sortedcontainers import SortedDict
import pandas as pd
import numpy as np
import pickle
import sys


def load_data(window_size_s, resample_ms=1000, start_time=None, sim_time_seconds=None, filename='data.pkl'):
    """
    Load and process data from data.csv.
        
    Args:
        window_size_s: Window size for rolling average in seconds
        resample_ms: Resample interval in milliseconds
    Returns:
        pandas.DataFrame: Processed dataframe with timestamp index
    """
    # Load data.pkl as a pandas dataframe
    data = pd.read_pickle(filename)
    data = data.sort_index()
    data = data.dropna()

    # Resample and apply rolling average
    sample_rate_s = (data.index[1] - data.index[0]).total_seconds()
    n_samples = round(window_size_s / sample_rate_s)
    data.loc[:, 'S1-smoothed'] = data['S1'].rolling(window=n_samples, center=True).mean()
    data.loc[:, 'S2-smoothed'] = data['S2'].rolling(window=n_samples, center=True).mean()
    data.loc[:, 'S3-smoothed'] = data['S3'].rolling(window=n_samples, center=True).mean()
    data = data.resample('%dms' % resample_ms).mean()
    data = data.dropna()

    # Compute polarization drift rate
    def get_polarization_drift_rate(df, cols=['S1', 'S2', 'S3']):
        """Compute angle change (radians) and polarization drift rate (radians/sec)."""
        S = df[cols]
        S_next = S.shift(-1)
        
        dot = (S * S_next).sum(axis=1)
        norm = np.linalg.norm(S.to_numpy(), axis=1)
        norm_next = np.linalg.norm(S_next.to_numpy(), axis=1)
        den = norm * norm_next
        
        cos_theta = (dot / den).clip(-1, 1)
        angle_step_rad = np.arccos(cos_theta)
        
        dt = (df.index.to_series().shift(-1) - df.index.to_series()).dt.total_seconds()
        drift_rate = angle_step_rad / dt
        
        mask_invalid = (den == 0) | (~np.isfinite(den)) | (~np.isfinite(dt)) | (dt == 0)
        angle_step_rad[mask_invalid] = np.nan
        drift_rate[mask_invalid] = np.nan
        
        return drift_rate
    
    data['drift_rate'] = get_polarization_drift_rate(
        data, cols=['S1-smoothed', 'S2-smoothed', 'S3-smoothed']
    )
    data = data.dropna()

    # Filter data within desired time range
    if start_time:
        data = data.loc[data.index >= start_time]
    if sim_time_seconds:
        end_time = data.index[0] + pd.Timedelta(seconds=sim_time_seconds)
        data = data.loc[data.index <= end_time]

    return data

# Progress bar
def progressBar(count_value, total, suffix='', done=False):
    bar_length = 50
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()
    if done:
        print()

# Class to memoize the optimization results
class MemoOpt():
    def __init__(self):
        self.memo = SortedDict()
        self.precision = 6

    def __len__(self):
        return len(self.memo)

    def memoize(self, fid, params):
        fid = round(fid, self.precision)
        if fid in self.memo:
            print(f"Warning: Fidelity {fid} already memoized. Overwriting with new params.")
        self.memo[fid] = params

    def retrieve(self, fid, tol=0.01):
        # Retrieve the closest fidelity in the memoization dictionary
        # Returns True if the fidelity is found, False otherwise
        # Returns the parameters if the fidelity is within tolerance, None otherwise
        fid = round(fid, self.precision)
        if fid in self.memo:
            return True, self.memo[fid]

        idx = self.memo.bisect_left(fid)
        if idx == len(self.memo):
            fidr, paramsr = 1e6, None
        else:
            fidr, paramsr = self.memo.peekitem(idx)

        if idx == 0:
            if abs(fidr - fid) <= tol:
                return False, paramsr
            else:
                return False, None

        # Determine which of fidr or fidl is closer to fid and within tolerance
        fidl, paramsl = self.memo.peekitem(idx-1)
        diff_r = abs(fidr - fid)
        diff_l = abs(fidl - fid)
        if diff_r <= diff_l and diff_r <= tol:
            return False, paramsr
        elif diff_l < diff_r and diff_l <= tol:
            return False, paramsl
        else:
            return False, None
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memo, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memo = pickle.load(f)

if __name__ == "__main__":
    opt = MemoOpt()
    opt.memoize(0.75, (2,3,4))
    opt.memoize(0.80, (5,6,7))
    print(opt.memo)
    print(opt.retrieve(0.75))
    print(opt.retrieve(0.75))
    print(opt.retrieve(0.8))
    print(opt.retrieve(0.759))
    print(opt.retrieve(0.791))
    print(opt.retrieve(.741))
    print(opt.retrieve(0.809))
    print(opt.retrieve(0.775))
    print(opt.retrieve(0.73))
    print(opt.retrieve(0.82))