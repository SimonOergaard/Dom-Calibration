# ---------------------------------------------------------------------------------------------- #
#                                Cherenkov Photon Calculation
# ---------------------------------------------------------------------------------------------- #
#
#   Analysis of Cherenkov photon timing using high statistics
#   
#   Author: Troels Petersen (NBI)
#   Date: 7th of July 2023 (first version)
#   
# ---------------------------------------------------------------------------------------------- #
"""

Ideas:
- Test on time differences on single string.
- Test as a function of muon path.

"""   
# ---------------------------------------------------------------------------------------------- #

import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import simplejson
from iminuit import Minuit
from scipy import stats

sys.path.append('External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax

Verbose = False
SaveFig = True
Nevents_PlotDist_tA = 20

nIce = 1.34         # Index of refraction of ice
c = 0.299792458     # Speed of light in m/ns

# Ranges used for defining mean and std. in "simple" calculation:
rangeMedian_used = 400.0
rangeMean_used = 400.0

Ndirect_min = 15   # Minimum number of "direct" (i.e. not very scattered) cherenkov photon signals in DOMs (reference: 15)

NeventsUsed = 2000  # Number of events considered in analysis
Nupdate = 1000     # Rate of giving progress update

dist_bins = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 9999.0]
# dist_bins = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 9999.0]

Nmuonzenith_bins = 6
# muonzenith_bins = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]   # Unused!

# Range and binning (for tA and fitting tA distribution)
Nbins, xmin, xmax = 200, 5000, 15000
binning = np.linspace(xmin, xmax, Nbins)

Nbins_tA, xmin_tA, xmax_tA = 125, -1000, 1500
binning_tA = np.linspace(xmin_tA, xmax_tA, Nbins_tA)



# ---------------------------------------------------------------------------------------------- #
# 
# ---------------------------------------------------------------------------------------------- #

#data_mc = pd.read_parquet("Data/10k_mc_photon_distance.parquet")
#data_tr = pd.read_parquet("Data/10k_mc_truth_photon_distance.parquet")
#data_rd = pd.read_parquet("Data/10k_rd_photon_distance.parquet")
data_mc = pd.read_parquet("~/GNN/workspace/data/Parguet_files/1k_mc_photon_distance_selected.parquet")
data_tr = pd.read_parquet("~/GNN/workspace/data/Parguet_files/1k_mc_true_photon_distance_selected.parquet")
data_rd = pd.read_parquet("~/GNN/workspace/data/Parguet_files/1k_rd_photon_distance_selected.parquet")
data_mc = data_mc.sort_values(by=['event_no', 'dom_time'])
data_tr = data_tr.sort_values(by=['event_no', 'dom_time'])
data_rd = data_rd.sort_values(by=['event_no', 'dom_time'])

print(data_mc.keys())
print(data_mc.head())

print(data_tr.keys())
print(data_tr.head())

print(data_rd.keys())
print(data_rd.head())

# Old structure:
#Index(['dom_time', 'dom_x', 'dom_y', 'dom_z', 'event_no', 'azimuth_pred',
#       'azimuth', 'position_x_pred', 'position_y_pred', 'position_z_pred',
#       'position_x', 'position_y', 'position_z', 'zenith_pred', 'zenith',
#       'weights', 'dist_dom', 'photon_distance', 'photon_z', 'photon_azimuth',
#       'photon_zenith', 'cherenkov_x', 'cherenkov_y', 'cherenkov_z'],
#        dtype='object')

# New structure (10k):
#Index(['event_no', 'dom_time', 'dom_x', 'dom_y', 'dom_z', 'width',
#       'azimuth_pred', 'zenith_pred', 'position_x_pred', 'position_y_pred',
#       'position_z_pred', 'dom_id', 'activated', 'doca', 'photon_distance',
#       'photon_z', 'photon_azimuth', 'photon_zenith', 'cherenkov_x',
#       'cherenkov_y', 'cherenkov_z', 't_A'],
#      dtype='object')

# Newest structure (100k):
#Index(['charge', 'dom_time', 'dom_x', 'dom_y', 'dom_z', 'event_no', 'width',
#       'azimuth_pred', 'zenith_pred', 'position_x_pred', 'position_y_pred',
#       'position_z_pred', 'doca', 'photon_distance', 'photon_z',
#       'photon_azimuth', 'photon_zenith', 'cherenkov_x', 'cherenkov_y',
#       'cherenkov_z', 'dist_cher_A', 't_A'],
#      dtype='object')

print(f"  Number of DOM hits in MC:    {len(data_mc.event_no.values):7d}")
print(f"  Number of DOM hits in TRUTH: {len(data_tr.event_no.values):7d}")
print(f"  Number of DOM hits in DATA:  {len(data_rd.event_no.values):7d}")


# ---------------------------------------------------------------------------------------------- #
# Functions:
# ---------------------------------------------------------------------------------------------- #

# Function that takes dataframe and two ranges (in ns) as input, and returns an estimate of the peak value and width in t_A:
def dist(x1, y1, z1, x2, y2, z2) :
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

# Function that finds median tA time, and then mean of hits within a range (400ns) of the median. Returns median, mean, std, and N hits in range.
def FindMeanTimeAtA(tA_values, rangeMedian, rangeMean, verbose=False) :
    median_tA = np.median(tA_values)
    if (len(tA_values[np.abs(tA_values-median_tA) < rangeMedian]) > 1) :
        mean_tA = np.mean(tA_values[np.abs(tA_values-median_tA) < rangeMedian])
        std_tA = np.std(tA_values[np.abs(tA_values-mean_tA) < rangeMean], ddof=1)
        Ndirect_tA = len(tA_values[np.abs(tA_values-mean_tA) < 2*std_tA])
    else :      # if (np.isnan(median_tA) or np.isnan(mean_tA) or np.isnan(std_tA)) :
        mean_tA = 0.0
        std_tA = 0.0
        Ndirect_tA = 0    
        # print("\n  WARNING: No pulses found within time window!", median_tA, mean_tA, std_tA)
        # print(tA_values)
    return median_tA, mean_tA, std_tA, Ndirect_tA


# Function for Chi2 fitting tA distributions with CrystalBall function (with exponent parameter "m" fixed):
def func_CBpol0(x, N, mu, sigma, beta, m, cst) :
    binwidth_gauss = (xmax-xmin)/Nbins
    return N * binwidth_gauss * stats.crystalball.pdf(-x, beta, m, loc=mu, scale=sigma) + cst

def tAfit_SingleEvent(data, N_init, mu_init, sigma_init, beta_init, m_init, cst_init, verbose) :

    # Put data into histogram:
    counts, bin_edges = np.histogram(data, bins=binning)
    X = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0
    unc_counts = np.sqrt(counts)

    # Prepare and perform the Chi2 fit:
    chi2reg = Chi2Regression(func_CBpol0, X[counts>0], counts[counts>0], unc_counts[counts>0])
    minuit_chi2 = Minuit(chi2reg, N=N_init, mu=mu_init, sigma=sigma_init, beta=beta_init, m=m_init, cst=cst_init)
    minuit_chi2.errordef = 1.0     # ChiSquare fit
    # minuit_chi2.fix(5)             # Fix the "m" parameter
    minuit_chi2.migrad()           # Unleash the minimizing (do the fit)!
    if (not minuit_chi2.fmin.is_valid) :                                   # Check if the fit converged
        print("  WARNING: The ChiSquare fit DID NOT converge!!!")

    # Print the fit result (if verbose):
    if (verbose) :
        for name in minuit_chi2.parameters :
            value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
            print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")
        chi2_value = minuit_chi2.fval                         # The Chi2 value at the minimum
        Ndof_value = np.sum(counts > 0) - minuit_chi2.nfit    # Number of DOF (Numbor of non-empty bins minus fit parameters) 
        Prob_value = stats.chi2.sf(chi2_value, Ndof_value)    # The chi2 probability given Ndof degrees of freedom
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")

    # Make a string containing fit result:
    d = {'Fit parameters:' : "",
         'Entries'   : "{:d}".format(sum(counts)),
         'Chi2/d.o.f': "{:.1f} / {:d}".format(chi2_value, Ndof_value),
         'Prob'      : "{:.4f}".format(Prob_value),
         'N'         : "{:.1f} +/- {:.1f}".format(minuit_chi2.values['N'], minuit_chi2.errors['N']),
         'mu'        : "{:.1f} +/- {:.1f}".format(minuit_chi2.values['mu'], minuit_chi2.errors['mu']),
         'sigma'     : "{:.1f} +/- {:.1f}".format(minuit_chi2.values['sigma'], minuit_chi2.errors['sigma']),
         'beta'      : "{:.3f} +/- {:.3f}".format(minuit_chi2.values['beta'], minuit_chi2.errors['beta']),
         'm'         : "{:.2f} +/- {:.2f}".format(minuit_chi2.values['m'], minuit_chi2.errors['m']),
         'cst'       : "{:.1f} +/- {:.1f}".format(minuit_chi2.values['cst'], minuit_chi2.errors['cst'])}

    # Return histogram values, fit parameters and errors, Chi2 value, Ndof, and Chi2 probability, fit validity, and fit result string:
    return X, counts, unc_counts, minuit_chi2.values[:], minuit_chi2.errors[:], chi2_value, Ndof_value, Prob_value, minuit_chi2.fmin.is_valid, d



# ---------------------------------------------------------------------------------------------- #
# Loop over MC to define single events:
# ---------------------------------------------------------------------------------------------- #

Nevents = [0, 0, 0]                             # Event counter

means_tA = [[], [], []]         # Means
stds_tA  = [[], [], []]         # Standard deviations
N_tA     = [[], [], []]         # Number of DOM hits
Ngood_tA = [[], [], []]         # Number of DOM hits in a range of 400ns around the mean of the expected times
muonzenith = [[], [], []]       # Muon zenith distributions

# Overall and differential distributions in PHOTON TRAVEL DISTANCE and MUON ANGLE:
data_tAcorr = [[], [], []]
data_tAcorr_100m = [[], [], []]
data_tAcorr_dist = [[[] for i in range(3)] for i in range(len(dist_bins)-1)]
data_tAcorr_muonzenith = [[[] for i in range(3)] for i in range(Nmuonzenith_bins)]
Nbins, xmin, xmax = 200, 5000, 15000

datatypes = [0, 1, 2]       # Which datatypes (mc, truth, data) to use
datanames = ["Monte Carlo - RECO", "Monte Carlo - TRUTH", "Real Data - RECO"]
datashortnames = ["MC", "TRUTH", "DATA"]

for i_datatype, datatype in enumerate(datatypes) :
    if (datatypes[i_datatype] == 1) :
        print(f"\n  {datanames[i_datatype]:s} analysis:")
        print("  Looping over hits, dividing between events:")
        if (i_datatype == 0) : data = data_mc
        if (i_datatype == 1) : data = data_tr
        if (i_datatype == 2) : data = data_rd

        startN  = 0                                      # Index of first hit in each event
        EventN_current = data['event_no'].iloc[0]        # Set to first event number (to get natural start)

        # for i in range(NhitsUsed) :
        while (Nevents[i_datatype] < NeventsUsed) :
            if (i%Nupdate==0) : print(i, Nevents[i_datatype], data_mc['event_no'].iloc[i])

            # If a new event number is encoutered, then initiate calculations on the current events:
            if (data['event_no'].iloc[i] != EventN_current) :

                # First, calculate the times at point A (vectorized for hits in event):
                dist_CherDOM = data['photon_distance'].iloc[startN:i]
                dist_CherA   = dist(data['cherenkov_x'].iloc[startN:i],     data['cherenkov_y'].iloc[startN:i],     data['cherenkov_z'].iloc[startN:i],
                                    data['position_x_pred'].iloc[startN:i], data['position_y_pred'].iloc[startN:i], data['position_z_pred'].iloc[startN:i])
                # tA = data.iloc[startN:i,0].values - dist_CherDOM * nIce / c + dist_CherA / c

                # ------------------------------------------------------ KEY LINE -----------------------------------------------
                tA = np.array(data['dom_time'].iloc[startN:i] - data['photon_distance'].iloc[startN:i] * nIce/c + dist_CherA / c)
                # ------------------------------------------------------ KEY LINE -----------------------------------------------

                if (Nevents[i_datatype] < 0) :
                    for j in range(len(tA)) :
                        print(f"  {j:3d}:  {tA[j]:7.1f}  vs.  {data['t_A'].iloc[startN+j]:7.1f}")
        
                # Secondly, get the distribution features (mean and std) of tA:
                median, mean, std, Ndirect = FindMeanTimeAtA(tA, rangeMedian_used, rangeMean_used, verbose=False)
                if (std == 0 and i<100000) :
                    print(f"WARNING: No pulses within time window found!  Median time: {median:7.1f}  Event number:", EventN_current)
                if (i<1000) :
                    print(f"  {i:6}: {Ndirect:2d} / {i-startN:3d}   Median, Mean, and Std of time at A: {median:7.1f}   {mean:7.1f}   {std:7.1f} ns")

                N_tA[i_datatype].append(i-startN)
                Ngood_tA[i_datatype].append(Ndirect)
                if (Ndirect >= Ndirect_min) :
                    # Also, consider which muon zenith angle bin the event is in:
                    muonzenith[i_datatype].append(data['zenith_pred'].iloc[startN])
                    muonzenith_bin = min(max(int(data['zenith_pred'].iloc[startN] / (np.pi/2) * Nmuonzenith_bins), 0), Nmuonzenith_bins-1)
                    data_tAcorr_muonzenith[muonzenith_bin][i_datatype].append(tA[dist_CherDOM<100.0]-mean)
                
                    means_tA[i_datatype].append(mean)
                    stds_tA[i_datatype].append(std)
                    data_tAcorr[i_datatype].append(tA-mean)
                    data_tAcorr_100m[i_datatype].append(tA[dist_CherDOM<100.0]-mean)
                    for j in range(len(dist_bins)-1) :
                        data_tAcorr_dist[j][i_datatype].append(tA[(dist_bins[j] <= dist_CherDOM) & (dist_CherDOM < dist_bins[j+1])] - mean)


                # Plot the few first examples:
                if (Nevents[i_datatype] < Nevents_PlotDist_tA or EventN_current == 450 or EventN_current == 568) :

                    # Should be likelihood fit!!!
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # TAKEN OUT: x, y, sy, fit_values, fit_errors, chi2, Ndof, Prob, isvalid, fittext = tAfit_SingleEvent(tA, Ndirect, mean, std/2.0, 0.5, 3.0, 1.0, True)
                    hist_tA = ax.hist(tA, bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label=f'{datashortnames[i_datatype]:s}: tA (ns)', color='red')
                    # hist_tA = ax.hist(tA, bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='Event', color='blue')
                    # ax.errorbar(x, y, yerr=sy, marker = '.', fmt='o', drawstyle = 'steps-mid', color="blue", label='MC tA')
                    # ax.plot(x, func_CBpol0(x, *fit_values), 'red', linewidth=2.0, label='CrystalBall + Const fit')
                    # ax.text(0.05, 0.95, nice_string_output(fittext, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='blue', verticalalignment='top')

                    ax.set_xlabel('Muon time at Point A from hits (tA in ns)', fontsize=18)
                    ax.set_ylabel('Frequency / 10 ns', fontsize=18)
                    ax.set_ylim(ymin=0.0)
                    ax.legend(loc='upper right', fontsize=18)
                    ax.vlines(x=median, ymin=0, ymax=20, colors='black', linestyle="dashed")
                    ax.vlines(x=mean,   ymin=0, ymax=20, colors='blue', linestyle="dashed")
                    ax.vlines(x=mean-2*std, ymin=0, ymax=20, colors='grey', linestyle="dashed")
                    ax.vlines(x=mean+2*std, ymin=0, ymax=20, colors='grey', linestyle="dashed")
                    ax.vlines(x=median-400.0, ymin=0, ymax=20, colors='green', linestyle="dashed")
                    ax.vlines(x=median+400.0, ymin=0, ymax=20, colors='green', linestyle="dashed")
                    fig.tight_layout()
                    fig.show()
                    if (SaveFig) :
                        fig.savefig(f"Figures_tA/Hist{datashortnames[i_datatype]:s}_timeA_event{Nevents[i_datatype]:1d}.pdf", dpi=600)
                    plt.close(fig)
    
                # Finally, update the event count/search:
                EventN_current = data['event_no'].iloc[i]
                Nevents[i_datatype] += 1
                startN = i


print(f"\n  The number of MC events used is:    {Nevents[0]:5d}")
print(f"\n  The number of Truth events used is: {Nevents[1]:5d}")
print(f"\n  The number of Data events used is:  {Nevents[2]:5d}")


# ---------------------------------------------------------------------------------------------- #
# Write "simple" objects to files:
# ---------------------------------------------------------------------------------------------- #

#with open('outfile', 'wb') as fp:
#    pickle.dump(itemlist, fp)

outputfile = open('Output/output_means_tA.json', 'w')
simplejson.dump(means_tA, outputfile)
outputfile.close()

outputfile = open('Output/output_stds_tA.json', 'w')
simplejson.dump(stds_tA, outputfile)
outputfile.close()

outputfile = open('Output/output_Npulses_tA.json', 'w')
simplejson.dump(N_tA, outputfile)
outputfile.close()

outputfile = open('Output/output_Ngood_tA.json', 'w')
simplejson.dump(Ngood_tA, outputfile)
outputfile.close()

outputfile = open('Output/output_MuonZenith.json', 'w')
simplejson.dump(muonzenith, outputfile)
outputfile.close()






# ---------------------------------------------------------------------------------------------- #
# Concatenate tAcorr arrays to become "pure" lists and hence JSON compliant:
# ---------------------------------------------------------------------------------------------- #

# List of lists:
for k in range(3) :
    data_tAcorr[k] = np.concatenate(data_tAcorr[k]).tolist()
    data_tAcorr_100m[k] = np.concatenate(data_tAcorr_100m[k]).tolist()
    
outputfile = open('Output/output_tAcorr.json', 'w')
simplejson.dump(data_tAcorr, outputfile)
outputfile.close()

outputfile = open('Output/output_tAcorr_100m.json', 'w')
simplejson.dump(data_tAcorr_100m, outputfile)
outputfile.close()


# List of lists of lists:
for k in range(3) :
    for l in range (len(dist_bins)-1) :
        data_tAcorr_dist[l][k] = np.concatenate(data_tAcorr_dist[l][k]).tolist()
    for l in range (Nmuonzenith_bins) :
        if (len(data_tAcorr_muonzenith[l][k]) > 0) :
            data_tAcorr_muonzenith[l][k] = np.concatenate(data_tAcorr_muonzenith[l][k]).tolist()

outputfile = open('Output/output_tAcorr_dist.json', 'w')
simplejson.dump(data_tAcorr_dist, outputfile)
outputfile.close()

outputfile = open('Output/output_tAcorr_muonzenith.json', 'w')
simplejson.dump(data_tAcorr_muonzenith, outputfile)
outputfile.close()


        
        
# ---------------------------------------------------------------------------------------------- #
# Plot tA distributions:
# ---------------------------------------------------------------------------------------------- #

# Consider the tA distribution of all hits and a subselection of those within 100m of the track:
#   Hypothesis: The background from noise should largely disappear with the 100m cut.

# Plot tA for all and with photon travel distance < 100m:
Nbins, xmin, xmax = 250, -1000, 1500
fig, ax = plt.subplots(figsize=(16, 8))
hist_tA = ax.hist(data_tAcorr[0],      bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='MC tA', color='blue')
hist_tA = ax.hist(data_tAcorr_100m[0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='MC tA (<100m)', color='blue')
hist_tA = ax.hist(data_tAcorr[1],      bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Truth tA', color='blue')
hist_tA = ax.hist(data_tAcorr_100m[1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='Truth tA (<100m)', color='blue')
hist_tA = ax.hist(data_tAcorr[2],      bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Data tA', color='red')
hist_tA = ax.hist(data_tAcorr_100m[2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='Data tA (<100m)', color='red')
ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18);
ax.legend(loc='upper right', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Hist_MCTRRD_Dists_tA_All.pdf', dpi=600)
# plt.close(fig)
