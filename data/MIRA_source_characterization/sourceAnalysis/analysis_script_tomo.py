#%% IMPORTS
import glob
# import helper_functions as hf
import numpy as np
import file_readers as fr
from matplotlib import pyplot as plt
import QST_bayes as btom
from plot_functions import makeRhoImages

#%% FILE I/O

datadir = 'data/MIRA/TOMO'
basename = 'MIRAsourceTOMO'

pump_powers = np.array([5,10,25,50,75,100,150,200,250,300])

results = []

# THINNING PARAMETER (effectively increases number of Bayesian samples by 2**th)
th = 12

# BACKGROUND SUBTRACTION (set to true to subtract accidentals)
bkgd_subtraction = False

#%% DATA ANALYSIS

# LOOP through pump powers defined above
for power in pump_powers:
    pattern = datadir + '/' + basename + '_' + \
                ''.join(['0' for i in range(3-len(str(power)))]) + \
                str(power)
                
    print(power)

    flist = list(glob.glob(pattern + '*_hvdarl.dat'))
    
    # There are two datasets for every pump power setting, one taken before the other.
    # To separate them, we sort file names so that data is analyzed in the order it was taken.
    flist.sort()
    
    # LOOP through the two data sets
    for i, fname in enumerate(flist):
        
        # READ in count data
        count_data, proj_order, header_info = fr.read_count_data(fname,meas_type='tomo')

        if bkgd_subtraction:
            counts_arr = abs(count_data['coinc'] - count_data['accs'])
        else:
            counts_arr = count_data['coinc']
        
        # take counts from the first (and only) trial
        counts = counts_arr[0]
        
        # Run bayesian tomography, store results in output list
        result = btom.run_tomography(counts,proj_order,th=th)
        
        if power > 250:
            i += 2 # Temporary hack to get around different number of powers for each curve
        if len(results) <= i:
            results.append({'pump_powers': [], 'fids': [], 'fid_errors': [], 'counts': [], 'count_errors': [], 'int_times': []}) 
        
        # Extract results
        matrix = result['rhoC']
        results[i]['pump_powers'].append(power)
        results[i]['fids'].append(result['F0c'])
        results[i]['fid_errors'].append(result['dF0c'])
        results[i]['counts'].append(result['K'])
        results[i]['count_errors'].append(result['dK'])
        results[i]['int_times'].append(header_info['INT_TIME'])
        # print(result)
        # print(result.keys())
        # print("\n\n")
        # print(results[i])
        # print(counts)
        # print(count_data)
        # print(header_info)
        
        # Plot density matrices, real and imaginary parts
        # makeRhoImages(matrix,plt,'')
    
#%% PLOT FIDELITIES

# Save results to pkl file
import pickle as pkl
with open(f'tomo_results_th={th}.pkl', 'wb') as f:
    pkl.dump(results, f)

plt.figure()
for i, res in enumerate(results):
    plt.errorbar(res['pump_powers'],res['fids'],yerr=res['fid_errors'], label=f'dataset {i}')
plt.legend()
plt.show()
