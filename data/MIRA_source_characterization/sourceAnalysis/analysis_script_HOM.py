#%% IMPORTS
import glob
import numpy as np
from matplotlib import pyplot as plt

import helper_functions as hf
import file_readers as fr

#%% FILE I/O

# data directory
datadir = 'data/MIRA/HOM'

### CHOOSE DATASET (uncomment below) ########
# basename = 'MIRAsourceHOM_TEST'
basename = 'MIRAsourceHOM_thruPAs'
############################################


#%% DATA ANALYSIS 

# filename pattern, based on selection above
pattern = datadir + '/' + basename 

#########################################
#### READ DATA from MAIN data file ######
#########################################
flist = glob.glob(pattern + '*_HOM.dat')
fname = flist[0]
count_data, delays, header_info = fr.read_count_data(fname,meas_type='hom')

# normalize time delays, so ZERO is in the middle
delays = delays - np.mean(delays)

# coincidence counts for the first (and only) HOM scan
coinc = count_data['coinc'][0]

# fit "dip" to a gaussian, get fit parameters
[xmid, x0, x0_err, fwhm, fwhm_err, y0, y0_err, Vis, Vis_err] = hf.dip_char(delays,coinc)

#%% PLOT

plt.figure()
plt.ylabel(f'counts ({float(header_info["INT_TIME"])*1e-12} s)')
plt.ylim(0,np.max(coinc)*1.1)
plt.xlabel('relative path delay (ps)')
plt.title(basename + f': Vis = {Vis:.3f}+/-{Vis_err:.3f}')
plt.plot(delays,coinc)
plt.show()