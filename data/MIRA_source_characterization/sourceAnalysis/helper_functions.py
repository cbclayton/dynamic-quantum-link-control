# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:58:27 2025

@author: cmn
"""

import numpy as np
from matplotlib import pyplot as plt
# from scipy.stats import poisson
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from heapq import nsmallest

#%%%%%%%%%%%% PEAK CHARACTERIZATION %%%%%%%%%%%%%%%%

def peak_char(delays,yval):
    
    sig2fwhm = np.sqrt(8*np.log(2)) #converts sigma to FWHM
    
    midpt = np.argmax(yval)
    
    # initial parameter guesses
    _y0 = np.min(yval)
    _A = np.max(yval)-_y0
    _x0 = delays[midpt]
    xmid, fwhm = real_fwhm(yval, delays, _y0)
    _sig = fwhm/sig2fwhm
    
    print(_y0,_A,_x0,_sig)
    
    # perform gaussian fit
    try:
        popt, pcov = curve_fit(gaussian, delays, yval,
                               p0=[_y0, _A, _x0, _sig])
        
        y0, A, x0, sigma = popt
        y0_err, A_err, x0_err, sig_err = np.sqrt(np.diag(pcov))
    
        print(y0,A,x0,sigma)
    
    
    except:
        y0, A, x0, sigma = 0,0,0,0
        y0_err, A_err, x0_err, sig_err = 0,0,0,0 
    
    return [xmid, x0, x0_err, fwhm, sigma*sig2fwhm, sig_err*sig2fwhm, y0, y0_err, A, A_err]

def dip_char(delays,yval):
    
    [xmid, x0, x0_err, _, fwhm, fwhm_err, y0, y0_err, A, A_err] = peak_char(delays,-yval)
    
    Vis = -A/y0
    Vis_err = Vis*np.sqrt((y0_err/y0)**2 + (A_err/A)**2)
    
    return [xmid, x0, x0_err, fwhm, fwhm_err, -y0, y0_err, Vis, Vis_err]

def gaussian(x, y0, A, x0, sigma):
    return y0 + A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def real_fwhm(y_values_temp, x_values, bkgd):
    y_values, temp_l, temp_r = [], [], []

    # To make 'y_values_temp', a numpy array, into a python list
    for x in range(0,len(y_values_temp)):
        y_values.append(y_values_temp[x])
    peak_height = max(y_values)
    half_peak_height = (peak_height+bkgd)/2
    
    # Splitting the y_values data into before and after x_value at peak height
    y_l_temp = y_values[0:y_values.index(peak_height)]
    y_r_temp = y_values[y_values.index(peak_height):len(y_values)]
    
    # Finds 1st closest value to half_peak_height in y_l and y_r
    y_l = nsmallest(1, y_l_temp, key=lambda x: abs(x-half_peak_height))
    y_r = nsmallest(1, y_r_temp, key=lambda x: abs(x-half_peak_height))
    
    # Gets x_value pairs for y_l and y_r
    temp_l.append(x_values[y_l_temp.index(y_l[0])])
    temp_r.append(x_values[y_r_temp.index(y_r[0]) + len(y_l_temp) -1])
    fwhm_n = temp_l[0] - temp_r[0]
    xmid = (temp_l[0] + temp_r[0])/2
    
    return xmid, abs(fwhm_n)  

#%%%%%%%%%%% CAR CALCULATIONS %%%%%%%%%%%%%%%

def car_cw(counts,binarr,window):
    
    tbin = binarr[1]-binarr[0] #get histogram binsize (ps)
    
    midpt = np.argmax(counts) # find largest peak
    cbin = int(window//(2*tbin)) # number of bins around the central peak bin
    win_arr = np.arange(max(0,midpt-cbin),min(midpt+cbin+1,len(binarr))) #define coinc. window
    coins = np.sum(counts[win_arr[0]:win_arr[-1]+1]) # count coincidences
    
    acc = np.sum(counts) - coins
    
    nc = len(win_arr)
    na = len(binarr) - nc
    
    if coins > 0:
        CAR = coins*na/acc/nc
        err = CAR*np.sqrt(1/coins + 1/acc) #error estimate
    else:
        CAR, err = 0, 0
        
    #TODO refactor so this can be used in "find_reprate",
    # but "acc" is still defined as bkgd counts expected within coinc window
    # i.e. acc*nc/na
    
    return CAR, err, coins, acc/na/tbin


def find_reprate(counts,binarr,window,noise_level=1,ivl=10e3,eps=500,plot_peaks=False):
    '''
    Given an input time-difference histogram, find separation between pulses
    and calculate the repetition rate (Hz)

    Parameters
    ----------
    hist : NUMPY ARRAY (dtype = int64)
        Array of counts (INT).
    
    binarr : NUMPY ARRAY (dtype = int64)
        Array of time differences, corresponding to 'hist' data
        
    noise_level : Float
        Relative noise level, used to calculate minimum peak height
        

    Returns
    -------
    jump : INT
        Index in 'counts' array that corresponds to jump.
    '''
    
    tbin = binarr[1]-binarr[0] #g2 histogram binsize (ps)
    
    #get coincidence counts (total) and accidentals (per ps-bin) (CW)
    _, _, coins, acc = car_cw(counts, binarr, window)
    
    min_height = noise_level*acc*tbin #minimum peak height (relative to accidentals)

    peaks,_ = find_peaks(counts,height=min_height,distance=ivl/tbin)
    median_peak_sep = np.median(peaks[1:]-peaks[:-1]) #median inter-peak interval (bins)
    
    # print(1e12/median_peak_sep/tbin)
    
    #Attempt to improve median estimate of interpulse interval,
    # using full distance between peaks
    
    main_peak = np.argmax(counts)
    peak0_idx = peaks[0]
    peakf_idx = peaks[-1]
    
    # print( 1e12 * ((peakf_idx-peak0_idx)//median_peak_sep) / (binarr[peakf_idx] - binarr[peak0_idx]))
    
    # eps = 500 #allowed peak position error (ps)
    
    #TODO add error handling for no peaks
    
    #Check if first and last peaks are reliable-- i.e., a regular interval from the main peak
    i=1
    while (main_peak-peak0_idx)%median_peak_sep > eps/tbin:
        peak0_idx = peaks[i]
        i+=1
    i=1
    while (peakf_idx-main_peak)%median_peak_sep > eps/tbin:
        peakf_idx = peaks[-1-i]
        i+=1

    total_peak_dist = binarr[peakf_idx] - binarr[peak0_idx] #time between first & last peak (ps)
    
    # number of peaks that should fit 
    peak_number = (peakf_idx-peak0_idx)//median_peak_sep
    
    reprate = peak_number * 1e12 / total_peak_dist
    
    print(reprate)
    
    if plot_peaks:
        
        xlim = 10.5*1e9/reprate #shows 10 peaks on either side
        xmid = binarr[len(binarr)//2]*1e-3
        
        plt.figure()
        plt.plot(binarr*1e-3,counts)
        plt.plot(binarr[peaks]*1e-3,counts[peaks],'x')
        # plt.ylim(0)
        plt.yscale('log')
        plt.xlabel('$\Delta$t (ns)')
        plt.ylabel('Counts')
        plt.xlim(xmid-xlim,xmid+xlim)

    return reprate

def car_pulsed(counts,binarr,reprate,dutyfrac=0.1):
    
    ivl = 1e12/reprate # convert reprate (Hz) to pulse interval (ps)
    
    tbin = binarr[1]-binarr[0] # find histogram bin size
    
    duty = int(ivl // tbin) # convert reprate to number of bins
    
    cbin = int(duty*dutyfrac//2) # find number of bins corresponding to duty cycle
    coin_wdw = (2*cbin+1)*tbin # coincidence window in ps
    # print(f'Coincidence window: {coin_wdw} ps')
    
    if duty < 1:
        print(f'Time bins too large to resolve pulses!\nMax binsize: {ivl/2}ps')
        return car_cw(counts,binarr,coin_wdw)
    
    elif dutyfrac > 1:
        return car_cw(counts,binarr,coin_wdw)
    
    # print(f'Coincidence window (2): {coin_wdw} ps')
    
    # main_peak = np.argmax(counts) if findmax else len(counts)//2 # find main peak
    
    main_peak = np.argmax(counts)
    
    # peak_idx_alt,_ = find_peaks(counts,distance=duty*0.95)
    # find side peaks according to manually entered reprate (ivl)
    peak_idx = np.where((binarr-binarr[main_peak]) % ivl < tbin)[0]
    # if windows will be cut off at the edges of the histogram, delete them
    if peak_idx[0] - cbin < 0:
        peak_idx = np.delete(peak_idx,0)
    if peak_idx[-1] + cbin >= len(binarr):
        peak_idx = np.delete(peak_idx,-1)
    
    # PRINT the number of "manually" indexed peaks alongside results of "find_peaks" f'n
    # print('Number of peaks: ',len(peak_idx),' Auto-detected: ',len(peak_idx_alt))
    
    #define integration windows
    # all_wdows = [np.arange(pk-cbin, pk+cbin+1) for pk in peak_idx]
    # win_arr = np.reshape(all_wdows,-1)
    # TODO convert to mask for histogram array
    
    #sum the counts in each window
    all_peaks = [np.sum(counts[pk-cbin:pk+cbin+1]) for pk in peak_idx]
    
    # print(all_peaks)
    
    # if main_peak >= 0:
    #     coins = main_peak
    # else:
    max_idx = np.argmax(all_peaks)
    coins = all_peaks[max_idx]
    side_peaks = np.delete(all_peaks,max_idx)
    
    n = len(side_peaks)
    accs = np.mean(side_peaks)
    CAR = coins/accs
    err = CAR*np.sqrt(1/coins + 1/accs/n)
    
    return CAR, err, coins, accs
