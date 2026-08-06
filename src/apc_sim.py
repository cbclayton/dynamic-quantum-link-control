from re import I
import pandas as pd
import numpy as np
from polarization_drift import fid2angle, angle2fid
import matplotlib.pyplot as plt
from utils import load_data, progressBar

DELTA = 0.0277 # time for one gradient descent iteration
ETA = 0.062 # step size for gradient descent


def drift(data, t, step_duration):
    # polarization angle drift at time t
    idx = data.index.get_indexer([t], method='nearest')[0]
    drift_rate = data.iloc[idx]['drift_rate']
    angle = drift_rate * step_duration
    assert angle >= 0, "Invalid angle: %f at time %s" % (angle, t)
    return angle




def _simulate_apc(data, F_0, F_target, t_start, timeout_s=55, F_timeout=0.0, constant_drift_rate=None):
    """
    Simulate the execution of a polarization compensation routine.

    Args:
        data: pandas DataFrame with Stokes parameters
        F_0: initial fidelity
        F_target: target fidelity
        t_start: start timestamp
        timeout_s: timeout duration in seconds
        F_timeout: timeout fidelity
    Returns:
        tuple: (time_elapsed, F_final)
    """
    theta_target = fid2angle(F_target)
    theta = fid2angle(F_0)
    t = t_start
    while theta > theta_target:
        step_duration = DELTA
        if constant_drift_rate == None:
            d = drift(data, t, step_duration)
        else:
            d = constant_drift_rate * step_duration
        theta = theta - np.sin(theta)*ETA/2 + d
        t = t + pd.Timedelta(seconds=step_duration)
        if t > t_start + pd.Timedelta(seconds=timeout_s):
            theta_target = fid2angle(F_timeout)
            if theta > np.pi:
                break
    
    time_elapsed = (t - t_start).total_seconds()
    F_final = angle2fid(theta)
    return time_elapsed, F_final

def simulate_apc(data, F_0, F_target, t_start : pd.Timestamp, timeout_s=55, F_timeout=0.0):
    constant_drift_rate = None
    return _simulate_apc(data, F_0, F_target, t_start, timeout_s, F_timeout, constant_drift_rate=constant_drift_rate)

def predict_compensation_time(drift_rate, F_0, F_target, timeout_s=55, F_timeout=0.0):
    data = None # not used
    t_start = pd.Timestamp.now() # arbitrary
    time_elapsed, _ = _simulate_apc(data, F_0, F_target, t_start, timeout_s, F_timeout, constant_drift_rate=drift_rate)
    return time_elapsed


def timeout_drift_rate(F_target):
    # Drift rate above which all compensations will time out
    return ETA / (2*DELTA) * np.sin(fid2angle(F_target))

def upper_bound(drift_rate, F_0, F_target):
    # Upper bound on compensation times
    theta_0 = fid2angle(F_0)
    theta_target = fid2angle(F_target)
    timeout = timeout_drift_rate(F_target)
    if drift_rate > timeout:
        return np.inf
    return (theta_0 - theta_target) / (timeout - drift_rate)

# def better_upper_bound(drift_rate, F_0, F_target):
#     # Better upper bound on compensation times
#     theta_0 = fid2angle(F_0)
#     theta_target = fid2angle(F_target)
#     timeout = timeout_drift_rate(F_target)
#     a = 2*drift_rate*DELTA/ETA
#     denom = np.sqrt(1 - a**2)
#     atan = lambda x : np.arctanh((np.tan(x/2) - a) / denom)
#     k = (4/(ETA*denom)) * (atan(theta_0) - atan(theta_target))
#     return k * DELTA

def explore_compensation_thresholds():
    # make plot of drift rate vs compensation time for various threshold settings
    def plot_drift():
        drift_rates = np.linspace(0, 0.25, 100)
        F0s = [.90, .95, .98]#np.linspace(0.90, 0.99, 10)
        F_targets = [0.99, 0.995, 0.999]
        colors = plt.cm.tab10(np.linspace(0, 1, len(F_targets) * len(F0s)))
        for i, F_target in enumerate(F_targets):
            plt.axvline(x=timeout_drift_rate(F_target), color='k', linestyle='--')  # vertical line at timeout_drift_rate(F_target)
            for j, F_0 in enumerate(F0s):
                compensation_times = [predict_compensation_time(drift_rate, F_0=F_0, F_target=F_target, timeout_s=55, F_timeout=0.0) for drift_rate in drift_rates]
                plt.plot(drift_rates, compensation_times, label=f'{F_0:.2f} -> {F_target:.3f}', color=colors[i*len(F0s) + j])
                # upper_bound_times = [better_upper_bound(drift_rate, F_0=F_0, F_target=F_target) for drift_rate in drift_rates]
                # plt.plot(drift_rates, upper_bound_times, color=colors[i], linestyle='--')
        plt.legend()
        plt.xlabel('Drift Rate (rad/s)')
        plt.ylabel('Compensation Time (s)')
        plt.title('Compensation Time vs Drift Rate')
        plt.ylim(0, None)
        plt.show()

    def plot_F_target():
        # Plot compensation time as a function of F_target for various F_0s and drift rates
        F0s = [.90, .95, .98]
        drift_rates = np.linspace(0, 0.2, 3)
        F_targets = np.logspace(np.log10(.9), np.log10(.999), 100)
        colors = plt.cm.tab10(np.linspace(0, 1, len(drift_rates) * len(F0s)))
        for i, F_0 in enumerate(F0s):
            for j, drift_rate in enumerate(drift_rates):
                compensation_times = [predict_compensation_time(drift_rate, F_0=F_0, F_target=F_target) for F_target in F_targets]
                plt.plot(F_targets, compensation_times, label=f'F_0 = {F_0:.2f}, drift rate = {drift_rate:.2f}', color=colors[i*len(drift_rates) + j])
        plt.legend()
        plt.xlabel('F_target')
        plt.ylabel('Compensation Time (s)')
        plt.title('Compensation Time vs F_target')
        plt.ylim(0, None)
        plt.show()
    
    plot_drift()
    plot_F_target()



if __name__ == "__main__":
    explore_compensation_thresholds()


    # n = 3600
    # window_size_s = 1
    # skip = 3
    # bin_width = 27.7

    # data = load_data(window_size_s=window_size_s, resample_ms=200, sim_time_seconds=n)
    # ts = []
    # durs = []
    # Fs = []
    # for i, t_start in enumerate(data.index[::skip]):
    #     time_elapsed, F_final = simulate_apc(data, F_0=0.98, F_target=0.99, t_start=t_start)
    #     ts.append(t_start)
    #     durs.append(time_elapsed*1000 + 44)
    #     Fs.append(F_final)
    #     if i % 10 == 0:
    #         progressBar(i+1, len(data.index)/skip)

    # fig, ax = plt.subplots(1, 1)
    # bins = np.arange(min(durs), max(durs) + bin_width, bin_width)
    # ax.hist(durs, bins=bins, edgecolor='k', density=True)
    # # xx = np.linspace(0, max(durs), 10000)
    # ax.set_xlabel('Compensation Time (ms)')
    # ax.set_ylabel('Density')
    # # ax.set_title('Histogram of Compensation Times')
    # plt.show()