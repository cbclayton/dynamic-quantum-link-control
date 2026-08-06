# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 11:43:50 2025

@author: cmn
"""

import numpy as np

def read_histogram(fname):
    data = np.transpose(np.loadtxt(fname))
    [xvals,yvals] = data
    return xvals, yvals

def read_count_data(fname,meas_type='car'):    
    
    xvals = []
    count_data = {}
    output_data = {}
    header_info = {}
    
    keys = ['coinc','accs','s1','s2']
    
    if meas_type == 'car':
        keys = ['car','err'] + keys
    
    for key in keys: count_data[key] = []
    
    with open(fname) as f:
        
        for line in f:
            
            dat = line.split()
            
            if dat[0] == '#': #if in header
                
                header_info[dat[1]] = ' '.join(dat[3:])
                
            else: # actual data
                xval = None
                if meas_type == 'tomo':
                    xval = dat[0].capitalize()+dat[1].capitalize() #pair of projections               
                elif meas_type == 'hom':
                    xval = float(dat[0])
                xvals.append(xval)
                
                for i in range(len(keys)):
                    if meas_type == 'tomo':
                        count_data[keys[i]].append(float((dat[i+2])))
                    elif meas_type == 'hom':
                        count_data[keys[i]].append(float((dat[i+1])))
                    elif meas_type == 'car':
                        count_data[keys[i]].append(float((dat[i])))

    L2 = int(header_info['LINES_PER_TRIAL'])
    N = len(xvals)//L2 # N number of full trials
    
    xvals = xvals[:L2]
    
    for key in keys: output_data[key] = np.reshape(count_data[key][:N*L2],(N,L2))
    
    return output_data, xvals, header_info

def read_peak_data(fname):
    
    output = {}
    
    keys = ['times','xmid','x0','x0err','fwhm1','fwhm2','fwhm2err','y0','A']
    
    data = np.transpose(np.loadtxt(fname))
    
    for i in range(len(keys)): output[keys[i]] = data[i]
    
    return output
