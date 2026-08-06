#%% IMPORTS
import glob
import numpy as np
from matplotlib import pyplot as plt

import helper_functions as hf
import file_readers as fr

#%% FILE I/O

# data directory
datadir = 'data/MIRA/g2'

#####  CHOOSE DATASET (uncomment below) ############
basename = 'MIRAsourcedirect'
# basename = 'MIRAsourceBS'
# basename = 'MIRAsourceSPDC1'
# basename = 'MIRAsourceSPDC2'
###################################################

pump_powers = np.array([5,10,25,50,75,100,150,200,250])

# initialize output data arrays
g2_from_hist = np.zeros((len(pump_powers),2))
g2_from_counts = np.zeros((len(pump_powers),2))

'''
TWO arrays are defined above:
    
(1) "from hist" is meant for calculations from the raw data histogram.
(2) "from counts" uses summed count values, which were computed at the time of measurement.

Option (1) allows you more freedom, like choosing the coincidence window.
Option (2) uses data generated from a pre-chosen coincidence window; there are multiple trials.

'''


#%% DATA ANALYSIS

#Loop through pump powers defined above

j=0 #pump power index
for power in pump_powers: 
    
    # filename pattern, based on chosen pump power
    pattern = datadir + '/' + basename + '_' + \
                ''.join(['0' for i in range(3-len(str(power)))]) + \
                str(power)
    
    ##############################################
    #### READ DATA from MAIN data file ###########
    ##############################################
    flist = glob.glob(pattern + '*_CAR.dat')
    fname = flist[0]
    count_data, _, header_info = fr.read_count_data(fname,meas_type='car')
    
    # header contains key info about measurement, like the coincidence window
    #print(header_info)
    
    # find weighted average of CAR from all trials
    N = len(count_data['car']) # number of trials
    if N > 1:
        CAR_mean = np.average(count_data['car'],weights=count_data['err']**(-2))
        CAR_mean_err = np.std(count_data['car'])/np.sqrt(len(count_data['car']))
    else:
        CAR_mean = count_data['car'].flatten()[0]
        CAR_mean_err = count_data['err'].flatten()[0]
    
    # store the CAR and uncertainty in the "from counts" array
    g2_from_counts[j][0] = CAR_mean
    g2_from_counts[j][1] = CAR_mean_err
    
    #############################################
    #### READ DATA from RAW HISTOGRAM  ##########
    #############################################
    flist = glob.glob(pattern + '*_hist.dat')
    fname = flist[0]
    times, coinc_data = fr.read_histogram(fname)
    
    # Choose coincidence window (ps)
    window = int(header_info['COIN_WDW']) # this is the window that was used in the main data file
    # window = 1000 #ps
    
    # Set reprate (Hz), either with an automated function or a fixed value 
    reprate = hf.find_reprate(coinc_data,times,window,plot_peaks=False)
    # reprate = 76142000 #Hz
    
    # ABOVE: "find_reprate" function infers the reprate from the histogram.
    # To view the original histogram, set PLOT_PEAKS above to TRUE
    # This may fail if count rates are low; in that case, you can just set a fixed number
    
    dutyfrac = reprate*window*1e-12 #Express coincidence window as a fraction of the space between pulses
    
    # Calculate CAR from raw histogram
    CAR, err, coins, acc = hf.car_pulsed(coinc_data, times, reprate, dutyfrac=dutyfrac)
    print(power, CAR, err, coins, acc)
    
    # store the CAR and uncertainty in the "from hist" array
    g2_from_hist[j][0] = CAR
    g2_from_hist[j][1] = err
    
    j += 1
    

#%% MAKE PLOTS

'''
NOTE: in this context, "CAR" and "g2" are essentially the same measurement.
It's just that their interpretation is slightly different.

When measuring both signal and idler photons, "CAR" is an appropriate term,
because the largest peak corresponds (primarily) to coincident detections of
two members of the same photon pair. The side peaks correspond to background
noise and correlations between SPDC photons emitted at different times.
We expect this same noise level to contribute to the size of the "main" peak
in the form of "accidental" coincidences. Hence CAR measures the signal-to-noise.

When measuring just ONE of the signal/idler modes, input to a 50:50 BS as in
the SPDC1(2) datasets, it is called a "g(2) measurement". It characterizes
correlations and photon statistics of just ONE mode at a time. The side peaks
have almost the same meaning here: background noise + correlations to SPDC photons
emitted at different times. Because photons from different pulses should be
uncorrelated, we use the side peaks to normalize the "main" peak and determine
the strength of correlations at zero time delay. Exact same calculation as
CAR, but different physical interpretation.
'''

plt.figure()
# plt.axes().set_aspect('equal')
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim(1,2)
plt.ylabel('g2 or CAR')
plt.xlabel('Pump power (mW)')
plt.title(basename)
plt.errorbar(pump_powers,g2_from_hist[:,0],yerr=g2_from_hist[:,1])
plt.errorbar(pump_powers,g2_from_counts[:,0],yerr=g2_from_counts[:,1])
plt.show()

#%% RESULTS, DISCUSSION

'''

For most data sets, CAR vs pump power seems to be roughly an inverse relationship.

For SPDC1 and SPDC2, "CAR" is more appropriately called g(2), see discussion above.
Ideally, g(2)(dt = 0) would be 2, reflecting thermal statistics. This means it
is twice as likely to get coincident detections from the same pump pulse than it is
from two separate pump pulses. This is often described as "bunching", where it
is slightly more likely to see photons emitted at the same time, than at different times.
This is closely related to the rate of multipair emission.

In practice, g(2)(dt=0) is less than 2. This often indicates the incoming light is not truly
single mode, and is actually a combination of slightly distinguishable SPDC processes.
This "multimode" light shouldn't negatively effect fidelity for simple entanglement
distribution (I think), but it could negatively effect more complicated protocols
like swapping, 

You can see g2 < 2 in the "SPDC2" data set, with g2 ~ 1.7. This is H-polarized light,
coupled from the transmitted port of the PBS. We'll call this the "signal".
g2 is fairly constant with pump power, which makes sense because it is more
related to the spatial & spectral properties of the pump and phase matching
conditions in the crystal. (These are mostly fixed as pump power increases).
It is possible that other changes could appear at much higher pump powers,
like the "depleted pump" regime, which is beyond our current setup.

Meanwhile, the "SPDC1" ("idler") data set looks very different, with very high g2 and
inverse relationship to pump power. This can be explained by an imperfect PBS,
which reflects some proportion of H-polarized light into the V-polarized output.
This means the g2 measurement of the idler mode is "polluted" by the signal,
resulting in an artificially high number of coincident counts. As pump power
increases, the small leak of signal photons is overtaken by the statistics of
the individual idler mode, and g2 decreases toward 2 (or less).

The faulty PBS could also slightly affect HOM visibility; if the signal and idler exit
the same port of the PBS, then they will not interfere and hence reduce visibility.
As HOM visibility decreases, so does the achievable fidelity.


'''

