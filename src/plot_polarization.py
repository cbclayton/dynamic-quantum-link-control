import pickle
import matplotlib.pyplot as plt
from utils import load_data


plt.rcParams["font.size"] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O']

# Load data.pkl
filename = 'data.pkl'
dt = 48*3600
resample_ms = 1000
window1 = 1
window2 = 15*60 # rolling average window for drift rate
data = load_data(window_size_s=window1, resample_ms=resample_ms, sim_time_seconds=dt, filename=filename)
# data2 = load_data(window_size_s=window1, resample_ms=resample_ms, sim_time_seconds=dt)



# Plot S1, S2, S3
fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
# ax2 = ax1.twinx()

# Plot background color depending on data['daytime']
# Assume data['daytime'] is 1 for day, 0 for night, and values align with data.index.
import numpy as np

def plot_daynight_background(ax, data, color_day='#f9f3a9', color_night='#bbd0ec', alpha=0.2):
    """Plot background bands on ax indicating daytime/nighttime."""
    # Find transitions in daytime value
    daytime = data['daytime'].values
    times = data.index.to_numpy()
    changes = np.where(np.diff(daytime) != 0)[0]
    # Add start and end for complete coverage
    segments = [0] + (changes+1).tolist() + [len(data)]
    labeled = [False, False]
    for start, end in zip(segments[:-1], segments[1:]):
        val = daytime[start]
        label = ('Daytime' if val else 'Nighttime') if not labeled[1 if val else 0] else None
        ax.axvspan(times[start], times[end-1] if end < len(times) else times[-1],
                   color=color_day if val else color_night, alpha=alpha, zorder=0, label=label)
        labeled[1 if val else 0] = True

plot_daynight_background(ax1, data)
plot_daynight_background(ax2, data)

# First subplot: S1, S2, S3
alpha = .8
ax2.plot(data.index, data['S1-smoothed'], label=r'$S_1$', color='tab:red', alpha=alpha, linewidth=0.5, linestyle='-')
ax2.plot(data.index, data['S2-smoothed'], label=r'$S_2$', color='tab:blue', alpha=alpha, linewidth=0.5, linestyle='-')
ax2.plot(data.index, data['S3-smoothed'], label=r'$S_3$', color='tab:green', alpha=alpha, linewidth=0.5, linestyle='-')
ax2.set_ylabel('Stokes Parameter Value')


# Second subplot: drift_rate
ax1.plot(data.index, data['drift_rate'].rolling(window=window2, center=True).mean(), label='Drift rate', color='purple', linewidth=2)
ax1.set_ylabel('Mean Polarization Drift Rate (rad/s)')
ax1.set_xlabel('Time (Date / Hour)')
ax1.set_ylim(0, None)
ax1.set_xlim(data.index[0], data.index[-1])
ax2.legend()

plt.tight_layout()
plt.show()