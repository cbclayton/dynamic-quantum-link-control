import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import pickle
from datetime import datetime


from hardware import FID_CHECK_DURATION, power_from_rate, p_success_transmission

plt.rcParams["font.size"] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O']


def print_statistics(stats):
    print("\n=== Simulation Statistics ===")
    print(f"Average entanglement distribution rate: {stats['average_rate']:.2f} Hz")
    print(f"Number of fidelity check events: {stats['n_fid_checks']}")
    print(f"Number of recalibration events: {stats['n_compensations']}")
    print(f"Median, mean compensation time: {stats['median_compensation_time']:.1f} ms, {stats['mean_compensation_time']:.1f} ms")
    print(f"Time spent compensating: {stats['time_compensating']:.2f} seconds ({stats['time_compensating']/stats['total_time']:.1%})")
    print(f"Time below fidelity threshold: {stats['time_below_threshold']:.2f} seconds ({stats['time_below_threshold']/stats['total_time']:.1%})")

def plot_simulation(stats):
    """
    Create a plot of the simulation showing fidelity and entanglement distribution rate.
    
    Args:
        stats: Dictionary of statistics from run_simulation
        min_fidelity: Minimum fidelity threshold
    """
    # Print statistics
    print_statistics(stats)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot fidelity over time
    time_history = np.array(stats['time_history'])
    fidelity_history = np.array(stats['fidelity_history'])
    expected_rate_history = np.array(stats['expected_rate_history'])
    rate_history = np.array(stats['rate_history'])

    # ax1.plot(time_history, fidelity_history, 'b-', alpha=0.5, linewidth=0.5, label='Fidelity')
    # ax2.plot(time_history, expected_rate_history, 'k-', alpha=0.5, linewidth=0.5, label='Expected Instantaneous Rate')
    # ax2.plot(time_history, rate_history, 'g-', linewidth=1, label='Entanglement Rate')
    
    # Plot in contiguous segments where fidelity > 0
    mask = fidelity_history > 0 # Find segments where fidelity > 0
    for i, (k, g) in enumerate(groupby(enumerate(mask), key=itemgetter(1))):
        group = list(g)
        seg_indices = [i for i, m in group]
        if k:  # k == True: fidelity > 0
            labels = ['Fidelity', 'Expected Instantaneous Rate', 'Entanglement Distribution Rate']
            ax1.plot(time_history[seg_indices], fidelity_history[seg_indices], 'b-', alpha=0.5, linewidth=0.5, label=labels[0] if i == 0 else '')
            # ax2.plot(time_history[seg_indices], expected_rate_history[seg_indices], 'k-', alpha=0.5, linewidth=0.5, label=labels[1] if i == 0 else '')
            ax2.plot(time_history[seg_indices], rate_history[seg_indices], 'g-', linewidth=1, label=labels[2] if i == 0 else '')
    
    ax2.plot(stats['succ_agg_xx'], stats['succ_agg_yy'], 'ro', linewidth=1, label='Average Rate')

    ax1.set_ylabel('Fidelity')
    ax1.set_title('Simulation Results: Fidelity and Entanglement Distribution Rate')
    ax1.grid(True, alpha=0.3)    
    ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
    ax2.set_ylabel('Entanglement Distribution Rate (Hz)')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1])
    
    # Mark recalibration events with vertical bars
    mask = np.zeros_like(time_history, dtype=bool)
    for i in range(len(stats['time_history'])-1):
        mask[i] = fidelity_history[i] <= 0 or (i > 0 and fidelity_history[i-1] <= 0)
    ax1.fill_between(time_history, ax1.get_ylim()[0], ax1.get_ylim()[1], where=mask, color="#BEA82B", alpha=0.3, ec=None, label='Polarization Compensation')
    ax2.fill_between(time_history, ax2.get_ylim()[0], ax2.get_ylim()[1], where=mask, color="#BEA82B", alpha=0.3, ec=None, label='Polarization Compensation')
    # vertical line
    # for t in stats['compensation_event_times']:
    #     ax1.axvline(x=t, color='r', linestyle='--', alpha=0.7, linewidth=1)
    #     ax2.axvline(x=t, color='r', linestyle='--', alpha=0.7, linewidth=1)
      
    # Mark minimum fidelity with a horizontal line
    ax1.plot(stats['time_history'], stats['F_mins'], color='r', linestyle='--', alpha=0.7, linewidth=1, label='Minimum Fidelity')

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    
    # plt.tight_layout()
    
    
    
    plt.show()

def plot_calibration_durs(stats):
    plt.rcParams["font.size"] = 13
    print_statistics(stats)
    recalibration_durs = [1000*x for x in stats['compensation_event_durations'] if x > FID_CHECK_DURATION]

    # Save recalibration durs to pickle file
    with open(f'results/recalibration_durs_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl', 'wb') as f:
        pickle.dump(recalibration_durs, f)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bin_width = 27.7
    bins = np.arange(min(recalibration_durs), max(recalibration_durs) + bin_width, bin_width)
    ax.hist(np.array(recalibration_durs), bins=bins, edgecolor='k', density=True)
    ax.set_xlabel('Compensation Time (ms)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, .0066)
    # ax.set_title('Histogram of Compensation Times')
    plt.tight_layout()
    plt.show()
    plt.rcParams["font.size"] = 11

def plot_rates(freqs, results):

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # First subplot: Average rate vs recalibration frequency
    if None in freqs:
        adaptive_idx = freqs.index(None)
        adaptive_rate = results[adaptive_idx]['average_rate']
        adaptive_result = results[adaptive_idx]
        ax1.axhline(y=adaptive_rate, color='r', linestyle='--', label='Adaptive Recalibration')
    else:
        adaptive_idx = None
        adaptive_result = None
    
    # Get static recalibration data (excluding adaptive if present)
    static_freqs = [f for i, f in enumerate(freqs) if i != adaptive_idx]
    static_results = [r for i, r in enumerate(results) if i != adaptive_idx]
    
    if static_freqs:
        ax1.plot(static_freqs, [result['average_rate'] for result in static_results], 'o-', label='Static Recalibration')
    ax1.set_xlabel('Recalibration Frequency (seconds)')
    ax1.set_ylabel('Average Entanglement Distribution Rate (Hz)')
    ax1.set_title('Average Rate vs. Recalibration Frequency')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xscale('log')
    
    # Second subplot: Binned rates vs time bins for each frequency
    colors = plt.cm.tab10(np.linspace(0, 1, len(freqs)))
    color_idx = 0
    
    for i, (freq, result) in enumerate(zip(freqs, results)):
        if freq is None:
            label = 'Adaptive Recalibration'
            color = 'r'
            linestyle = '--'
        else:
            label = f'Frequency: {freq}s'
            color = colors[color_idx]
            linestyle = '-'
            color_idx += 1
        
        ax2.plot(result['time_bins'], result['rate_bins'], color=color, linestyle=linestyle, 
                label=label, alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Entanglement Distribution Rate (Hz)')
    ax2.set_title('Rates vs. Time for Each Recalibration Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_simulation_detailed(stats, start: pd.Timestamp=None, end: pd.Timestamp=None, F_min_varying=False):
    # Print statistics
    print_statistics(stats)
    
    if F_min_varying:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        (ax4, ax1, ax2) = axes
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        (ax1, ax2, ax3, ax4) = axes
    ax2_twin = ax2.twinx()

    # Plot fidelity over time
    time_history = np.array(stats['time_history'])
    fidelity_history = np.array(stats['fidelity_history'])
    expected_rate_history = np.array(stats['expected_rate_history'])
    rate_history = np.array(stats['rate_history'])
    F_source_history = np.array(stats['F_source_history'])
    F_trans_history = np.array(stats['F_trans_history'])
    F_trans_predicted_history = np.array(stats['F_trans_predicted_history'])
    r_avg_history = np.array(stats['r_avg_history'])
    if start:
        i = time_history.searchsorted(start)
        time_history = time_history[i:]
        fidelity_history = fidelity_history[i:]
        expected_rate_history = expected_rate_history[i:]
        rate_history = rate_history[i:]
        F_source_history = F_source_history[i:]
        F_trans_history = F_trans_history[i:]
        F_trans_predicted_history = F_trans_predicted_history[i:]
        r_avg_history = r_avg_history[i:]
        i = np.array(stats['succ_agg_xx']).searchsorted(start)
        succ_agg_xx = stats['succ_agg_xx'][i:]
        succ_agg_yy = stats['succ_agg_yy'][i:]
    if end:
        j = time_history.searchsorted(end)
        time_history = time_history[:j]
        fidelity_history = fidelity_history[:j]
        expected_rate_history = expected_rate_history[:j]
        rate_history = rate_history[:j]
        F_source_history = F_source_history[:j]
        F_trans_history = F_trans_history[:j]
        F_trans_predicted_history = F_trans_predicted_history[:j]
        r_avg_history = r_avg_history[:j]
        j = np.array(stats['succ_agg_xx']).searchsorted(end)
        succ_agg_xx = succ_agg_xx[:j]
        succ_agg_yy = succ_agg_yy[:j]
    pump_powers = np.array([power_from_rate(r/p_success_transmission()) for r in expected_rate_history])



    # ax1.plot(time_history, fidelity_history, 'b-', alpha=0.5, linewidth=0.5, label='Fidelity')
    # ax2.plot(time_history, expected_rate_history, 'k-', alpha=0.5, linewidth=0.5, label='Expected Instantaneous Rate')
    # ax2.plot(time_history, rate_history, 'g-', linewidth=1, label='Entanglement Rate')
    
    # Plot in contiguous segments where fidelity > 0
    mask = fidelity_history > 0 # Find segments where fidelity > 0
    for i, (k, g) in enumerate(groupby(enumerate(mask), key=itemgetter(1))):
        group = list(g)
        seg_indices = [i for i, m in group]
        if k:  # k == True: fidelity > 0
            ax1.plot(time_history[seg_indices], rate_history[seg_indices], '-', c='tab:green', linewidth=1.5, label='r(t)' if i == 0 else '')
            ax2_twin.plot(time_history[seg_indices], pump_powers[seg_indices], '--', c='m', alpha=0.6, linewidth=1.5, label='Pump Power' if i == 0 else '')
            ax2.plot(time_history[seg_indices], F_source_history[seg_indices], '-', c='tab:purple', linewidth=1.5, label=r'F$_{\mathrm{sd}}$' if i == 0 else '')
            if not F_min_varying:
                ax3.plot(time_history[seg_indices], F_trans_history[seg_indices], '-', c='tab:orange', linewidth=1.5, label=r'F$_{\mathrm{pol}}$' if i == 0 else '')
                ax3.plot(time_history[seg_indices], F_trans_predicted_history[seg_indices], '--', c='tab:orange', linewidth=1.5, label=r'F$_{\mathrm{pol}}^{\delta=0.1}$' if i == 0 else '')
                ax1.plot(time_history[seg_indices], r_avg_history[seg_indices], ':', c='tab:red', linewidth=2, label=r'$\bar{r}(t)$' if i == 0 else '')
            ax4.plot(time_history[seg_indices], fidelity_history[seg_indices], '-', c='tab:blue', linewidth=1.5, label='F(t)' if i == 0 else '')
            
    
    
    # Set lims, labels
    ax4.set_xlim(start, end)

    ax1.set_ylabel('Entanglement\nDistribution Rate (Hz)')
    if F_min_varying:
        ax1.set_ylim(0, max(rate_history)+3)
    if not F_min_varying:
        ax1.set_ylim(54, 84)

    ax2_twin.set_ylabel('Pump Power (mW)')
    ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1])
    ax2.set_ylabel('Source-Detector\nFidelity')
    ax2.plot([], [], '--', c='m', alpha=0.6, linewidth=1.5, label='Pump Power')

    if not F_min_varying:
        ax3.set_ylabel('Polarization Fidelity')
        ax3.set_ylim(ax3.get_ylim()[0], 1)
        axes[-1].set_xlabel('Time (hh:mm:ss)')
    else:
        axes[-1].set_xlabel('Time (day hh:mm)')

    ax4.set_ylabel('End-to-End Fidelity')
    ax4.set_ylim(min(ax4.get_ylim()[0], min(stats['F_mins'])-0.005), ax4.get_ylim()[1])
    


    # Mark minimum fidelity
    ax4.plot(stats['time_history'], stats['F_mins'], color='r', linestyle='--', alpha=0.7, linewidth=1.5, label=r'F$_{\mathrm{min}}$')

    
    # Mark recalibration events with vertical bars
    mask = np.zeros_like(time_history, dtype=bool)
    mask_fid_check = np.zeros_like(time_history, dtype=bool)
    mask_compensation = np.zeros_like(time_history, dtype=bool)
    for i in range(1, len(time_history)-1):
        fi = fidelity_history[i]
        fi1 = fidelity_history[i-1]
        ti1 = time_history[i-1]
        ti = time_history[i]
        mask[i] = fi <= 0 or fi1 <= 0
        mask_fid_check[i] = mask[i] and (fi <= 0 or (ti - ti1).total_seconds() == FID_CHECK_DURATION)
        mask_compensation[i] = mask[i] and mask_fid_check[i-1] and mask_fid_check[i-2]
    for ax in axes:
        # labels = ['Fidelity Check', 'Polarization Compensation']
        labels = [None, None]
        #BEA82B
        #A63D40
        ax.fill_between(time_history, ax.get_ylim()[0], ax.get_ylim()[1], where=mask_fid_check, color="tab:gray", alpha=0.5, ec=None, label=labels[0])
        ax.fill_between(time_history, ax.get_ylim()[0], ax.get_ylim()[1], where=mask_compensation, color='w', ec='k', label=labels[1], hatch='xx')
      
        ax.legend(loc='lower right', framealpha=1.0)
        ax.minorticks_on()
        
    # plt.tight_layout()
    
    
    
    plt.show()
