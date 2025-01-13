# ---------------------------------------------------------------------------------------------- #
#                                Cherenkov Photon Calculation
# ---------------------------------------------------------------------------------------------- #
#
#   Analysis of Cherenkov photon timing using high statistics - Plotting script
#   
#   Author: Troels Petersen (NBI)
#   Date: 18th of July 2023 (first version)
#   
# ---------------------------------------------------------------------------------------------- #
"""
"""   
# ---------------------------------------------------------------------------------------------- #

import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
from iminuit import Minuit
from scipy import stats

sys.path.append('External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax

plt.rcParams.update({'figure.max_open_warning': 50})

Verbose = False
SaveFig = True

dist_bins = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 9999.0]
Nmuonzenith_bins = 6
# muonzenith_bins = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]   # Unused!

# Range and binning (for fitting tA distribution)
Nbins_tA, xmin_tA, xmax_tA = 125, -1000, 1500
binning_tA = np.linspace(xmin_tA, xmax_tA, Nbins_tA)


# ---------------------------------------------------------------------------------------------- #
# Plot distributions of means, stds, and pulse counts:
# ---------------------------------------------------------------------------------------------- #

# Load means:
with open('Output/output_means_tA.json') as inputfile:
    means_tA = json.load(inputfile)

# Plot means:
Nbins, xmin, xmax = 200, 9000, 13000
fig, ax = plt.subplots(figsize=(16, 8))
hist_means = ax.hist(means_tA[0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Mean tA (MC)', color='blue') #, linestyle="dashed")
hist_means = ax.hist(means_tA[1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='Mean tA (Truth)', color='blue') #, linestyle="dashed")
hist_means = ax.hist(means_tA[2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Mean tA (Data)', color='red')
ax.set_xlabel('Mean time of signal (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18);
ax.legend(loc='upper right', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCRD_Mean_tA.pdf', dpi=600)


# ---------------------------------------------------------------------------------------------- #

# Load stds:
with open('Output_Ref/output_stds_tA.json') as inputfile:
    stds_tA = json.load(inputfile)

# Plot stds:
Nbins, xmin, xmax = 200, 0, 400
fig, ax = plt.subplots(figsize=(16, 8))
hist_stds = ax.hist(stds_tA[0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Std tA (MC)', color='blue') #, linestyle="dashed")
hist_stds = ax.hist(stds_tA[1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='Std tA (MC Truth)', color='blue') #, linestyle="dashed")
hist_stds = ax.hist(stds_tA[2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Std tA (Data)', color='red')
ax.set_xlabel('Std. of time of signal (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18);
ax.legend(loc='upper right', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCRD_Std_tA.pdf', dpi=600)


# ---------------------------------------------------------------------------------------------- #

# Load Npulses and Ngood:
with open('Output_Ref/output_Npulses_tA.json') as inputfile:
    Npulses_tA = json.load(inputfile)
with open('Output_Ref/output_Ngood_tA.json') as inputfile:
    Ngood_tA = json.load(inputfile)

# Plot Npulses:
Nbins, xmin, xmax = 201, -0.5, 200.5
fig, ax = plt.subplots(figsize=(16, 8))
hist_Npulses = ax.hist(Npulses_tA[0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='N pulses (MC)', color='blue') #, linestyle="dashed")
hist_Npulses = ax.hist(Npulses_tA[1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='N pulses (Truth)', color='blue') #, linestyle="dashed")
hist_Npulses = ax.hist(Npulses_tA[2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='N pulses (Data)', color='red')
hist_Npulses = ax.hist(Ngood_tA[0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='N in-time pulses (MC)', color='blue', linestyle="dashed")
hist_Npulses = ax.hist(Ngood_tA[1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, label='N in-time pulses (Truth)', color='blue', linestyle="dashed")
hist_Npulses = ax.hist(Ngood_tA[2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='N in-time pulses (Data)', color='red', linestyle="dashed")
ax.set_xlabel('Number of pulses in event', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18);
ax.legend(loc='upper right', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCRD_NpulsesAndNgood_tA.pdf', dpi=600)


# ---------------------------------------------------------------------------------------------- #
# Plot muon zenith angle distribution:
# ---------------------------------------------------------------------------------------------- #

with open('Output_Ref/output_MuonZenith.json') as inputfile:
    muonzenith = json.load(inputfile)

Nbins, xmin, xmax = 100, 0, np.pi/2
fig, ax = plt.subplots(figsize=(16, 8))
hist_muonzenith = ax.hist(muonzenith[0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Muon RECO zenith angle (MC)', color='blue')
hist_muonzenith = ax.hist(muonzenith[2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Muon RECOT zenith angle (Data)', color='red')
ax.set_xlabel('Muon zenith angle (rad)', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18);
ax.legend(loc='upper left', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCRD_MuonZenithAngle.pdf', dpi=600)


# ---------------------------------------------------------------------------------------------- #
# Plot tA distributions vs. muon zenith angle:
# ---------------------------------------------------------------------------------------------- #

# Try to divide in theta of muon, to consider effect of inter-string timing calibration.
#   Hypothesis: The more straight down the muon (and hence fewer strings), the better the resolution.

with open('Output_Ref/output_tAcorr_muonzenith.json') as inputfile:
    data_tAcorr_muonzenith = json.load(inputfile)


# Plot tA differentially with muon zenith angle:
Nbins, xmin, xmax = 90, -600, 1200
fig, ax = plt.subplots(figsize=(16, 8))
# ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0,1,Nmuonzenith_bins)))

#from cycler import cycler
#color_cyc = cycler('color', ['red', 'cyan', 'blue'])
#plt.gca().set_prop_cycle(color_cyc)

colors = ['red', 'black', 'blue', 'green', 'brown', 'orange']
for j in range(Nmuonzenith_bins) :
    text=f"({j*15:2.0f}-{(j+1)*15:2.0f})"
    histMC_tA = ax.hist(data_tAcorr_muonzenith[j][0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, density=True, label='MC tA '+text)
    histMC_tA = ax.hist(data_tAcorr_muonzenith[j][1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, density=True, label='Truth tA '+text)
    histMC_tA = ax.hist(data_tAcorr_muonzenith[j][2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, density=True, label='Data tA '+text)
ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 20 ns', fontsize=18);
ax.legend(loc='upper right', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCTRRD_tA_muonzeniths.pdf', dpi=600)
# plt.close(fig)



# ---------------------------------------------------------------------------------------------- #
# Fitting functions (for tA distributions):
# ---------------------------------------------------------------------------------------------- #

# Fitting functions:
def func_gpol0(x, N, mu, sigma, cst) :
    binwidth_gauss = (xmax-xmin)/Nbins
    norm = N * binwidth_gauss / np.sqrt(2.0*np.pi) / sigma
    z = (x-mu)/sigma
    return norm * np.exp(-0.5*z*z) + cst

def func_gexppol0(x, N, mu, sigma, tau, cst) :
    binwidth_gauss = (xmax-xmin)/Nbins
    return N * binwidth_gauss * stats.exponnorm.pdf(x, tau, mu, sigma) + cst

def func_CBpol0(x, N, mu, sigma, beta, m, cst) :
    binwidth_gauss = (xmax-xmin)/Nbins
    return N * binwidth_gauss * stats.crystalball.pdf(-x, beta, m, loc=mu, scale=sigma) + cst

def func_CBGpol0(x, N, mu, sigma, beta, m, fCB, sigma2, cst) :
    binwidth_gauss = (xmax-xmin)/Nbins
    z = (-x-mu)/sigma2
    return N * binwidth_gauss * (fCB * stats.crystalball.pdf(-x, beta, m, loc=mu, scale=sigma) + (1.0-fCB) * np.sqrt(2.0*np.pi) / sigma2 * np.exp(-0.5*z*z)) + cst

def func_GEpol0(x, N, mu, sigma, k, cst) :
    binwidth_gauss = (xmax-xmin)/Nbins
    norm = N * binwidth_gauss / np.sqrt(2.0*np.pi) / sigma
    z = (x-mu)/sigma
    if (z > -k) :
        return norm * np.exp(-0.5*z*z) + cst
    else :
        return norm * np.exp(0.5*k**2 + k*z) + cst
func_GEpol0_vec = np.vectorize(func_GEpol0)


# ---------------------------------------------------------------------------------------------- #
# Function for Chi2 fitting tA distributions with CrystalBall function:
# ---------------------------------------------------------------------------------------------- #
def tAfit(data, N_init, mu_init, sigma_init, beta_init, m_init, cst_init, verbose) :

    # Put data into histogram:
    counts, bin_edges = np.histogram(data, bins=binning_tA)
    X = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0
    unc_counts = np.sqrt(counts)

    # Prepare and perform the Chi2 fit:
    chi2reg = Chi2Regression(func_CBpol0, X[counts>0], counts[counts>0], unc_counts[counts>0])
    minuit_chi2 = Minuit(chi2reg, N=N_init, mu=mu_init, sigma=sigma_init, beta=beta_init, m=m_init, cst=cst_init)
    minuit_chi2.errordef = 1.0     # ChiSquare fit
    minuit_chi2.migrad()
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
# Plot tA distributions:
# ---------------------------------------------------------------------------------------------- #

# Consider the tA distribution of all hits and a subselection of those within 100m of the track:
#   Hypothesis: The background from noise should largely disappear with the 100m cut.

with open('Output_Ref/output_tAcorr.json') as inputfile:
    data_tAcorr = json.load(inputfile)
with open('Output_Ref/output_tAcorr_100m.json') as inputfile:
    data_tAcorr_100m = json.load(inputfile)


# Plot tA for all and with photon travel distance < 100m for MC, Truth, and Data:
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

hist_tA = ax.hist(data_tAcorr[0],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='MC tA', color='blue')
hist_tA = ax.hist(data_tAcorr_100m[0], bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=1, label='MC tA ($d_\gamma$ <100m)', color='blue')
hist_tA = ax.hist(data_tAcorr[1],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='Truth tA', color='black')
hist_tA = ax.hist(data_tAcorr_100m[1], bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=1, label='Truth tA ($d_\gamma$ <100m)', color='black')
hist_tA = ax.hist(data_tAcorr[2],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='Data tA', color='red')
hist_tA = ax.hist(data_tAcorr_100m[2], bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=1, label='Data tA ($d_\gamma$ <100m)', color='red')

ax.legend(loc='upper right', fontsize=18)
ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCTRRD_tA_dist.pdf', dpi=600)
# plt.close(fig)


# Plot tA for photons with travel distance < 100m for MC ONLY with fits:
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# MC:
x_MC0, y_MC0, sy_MC0, fit_values_MC0, fit_errors_MC0, chi2_MC0, Ndof_MC0, Prob_MC0, isvalid_MC0, fittext_MC0 = tAfit(data_tAcorr_100m[0], 250000.0, 90.0, 70.0, 0.45, 2.6, 20.0, True)
hist_tA = ax.hist(data_tAcorr_100m[0],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='MC tA ($d_\gamma$ < 100m)', color='blue')
# ax.errorbar(x_MC0, y_MC0, yerr=sy_MC0, marker = '.', fmt='o', drawstyle = 'steps-mid', color="blue", label='MC tA')
ax.plot(x_MC0, func_CBpol0(x_MC0, *fit_values_MC0), 'red', linewidth=2.0, label='CrystalBall + Const fit ()')
ax.text(0.05, 0.95, nice_string_output(fittext_MC0, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='blue', verticalalignment='top')

ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18)
ax.set_ylim(ymin=0.0)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MC_tAfit.pdf', dpi=600)



# Plot tA for photons with travel distance < 100m for MC and Truth with fits:
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# MC:
x_MC0, y_MC0, sy_MC0, fit_values_MC0, fit_errors_MC0, chi2_MC0, Ndof_MC0, Prob_MC0, isvalid_MC0, fittext_MC0 = tAfit(data_tAcorr_100m[0], 250000.0, 90.0, 70.0, 0.45, 2.6, 20.0, True)
hist_tA = ax.hist(data_tAcorr_100m[0],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='MC tA ($d_\gamma$ < 100m)', color='blue')
# ax.errorbar(x_MC0, y_MC0, yerr=sy_MC0, marker = '.', fmt='o', drawstyle = 'steps-mid', color="blue", label='MC tA')
ax.plot(x_MC0, func_CBpol0(x_MC0, *fit_values_MC0), 'blue', linewidth=2.0, label='CrystalBall + Const fit ()')
ax.text(0.05, 0.95, nice_string_output(fittext_MC0, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='blue', verticalalignment='top')

# Truth:
x_TR0, y_TR0, sy_TR0, fit_values_TR0, fit_errors_TR0, chi2_TR0, Ndof_TR0, Prob_TR0, isvalid_TR0, fittext_TR0 = tAfit(data_tAcorr_100m[1], 250000.0, 90.0, 70.0, 0.45, 2.6, 20.0, True)
hist_tA = ax.hist(data_tAcorr_100m[1],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='TR tA ($d_\gamma$ < 100m)', color='black')
# ax.errorbar(x_TR0, y_TR0, yerr=sy_TR0, marker = '.', fmt='o', drawstyle = 'steps-mid', color="black", label='TR tA')
ax.plot(x_TR0, func_CBpol0(x_TR0, *fit_values_TR0), 'black', linewidth=2.0, label='CrystalBall + Const fit ()')
ax.text(0.05, 0.65, nice_string_output(fittext_TR0, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='black', verticalalignment='top')

ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18)
ax.set_ylim(ymin=0.0)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,3,1]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCTR_tAfit.pdf', dpi=600)

print(f"\n  The width difference between MC and Truth is: {np.sqrt(fit_values_MC0[2]**2-fit_values_TR0[2]**2):4.2f} +- {np.sqrt(fit_errors_MC0[2]**2+fit_errors_TR0[2]**2):4.2f} ns")



# Plot tA for photons with travel distance < 100m for MC and Truth with fits:
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# MC:
x_MC0, y_MC0, sy_MC0, fit_values_MC0, fit_errors_MC0, chi2_MC0, Ndof_MC0, Prob_MC0, isvalid_MC0, fittext_MC0 = tAfit(data_tAcorr_100m[0], 250000.0, 90.0, 70.0, 0.45, 2.6, 20.0, True)
hist_tA = ax.hist(data_tAcorr_100m[0],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='MC tA ($d_\gamma$ < 100m)', color='blue')
# ax.errorbar(x_MC0, y_MC0, yerr=sy_MC0, marker = '.', fmt='o', drawstyle = 'steps-mid', color="blue", label='MC tA')
ax.plot(x_MC0, func_CBpol0(x_MC0, *fit_values_MC0), 'blue', linewidth=2.0, label='CrystalBall + Const fit ()')
ax.text(0.05, 0.95, nice_string_output(fittext_MC0, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='blue', verticalalignment='top')

# Data:
x_RD0, y_RD0, sy_RD0, fit_values_RD0, fit_errors_RD0, chi2_RD0, Ndof_RD0, Prob_RD0, isvalid_RD0, fittext_RD0 = tAfit(data_tAcorr_100m[2], 250000.0, 90.0, 70.0, 0.45, 2.6, 20.0, True)
hist_tA = ax.hist(data_tAcorr_100m[2],      bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label='Data tA ($d_\gamma$ < 100m)', color='red')
# ax.errorbar(x_RD0, y_RD0, yerr=sy_RD0, marker = '.', fmt='o', drawstyle = 'steps-mid', color="red", label='RD tA')
ax.plot(x_RD0, func_CBpol0(x_RD0, *fit_values_RD0), 'red', linewidth=2.0, label='CrystalBall + Const fit ()')
ax.text(0.05, 0.65, nice_string_output(fittext_RD0, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='red', verticalalignment='top')

ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 10 ns', fontsize=18)
ax.set_ylim(ymin=0.0)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,3,1]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCRD_tAfit.pdf', dpi=600)




# ---------------------------------------------------------------------------------------------- #
# Plot tA distributions vs. distance:
# ---------------------------------------------------------------------------------------------- #

# Divide in distance to DOM, to consider the effect of cherenkov light's travel in ice.
#   Hypothesis: The longer the distance, the larger the spread in tA (growing with sqrt(dist)?).

with open('Output_Ref/output_tAcorr_dist.json') as inputfile:
#with open('Output_12HitsMinimum/output_tAcorr_dist.json') as inputfile:
#with open('Output_18HitsMinimum/output_tAcorr_dist.json') as inputfile:
    data_tAcorr_dist = json.load(inputfile)

# Plot tA differentially:
Nbins, xmin, xmax = 90, -600, 1200
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0,1,int(len(dist_bins)/4))))

#from cycler import cycler
#color_cyc = cycler('color', ['red', 'cyan', 'blue'])
#plt.gca().set_prop_cycle(color_cyc)

for j, color in enumerate(['red', 'black', 'blue', 'green']):
    k=j*3
    text=f"({dist_bins[k]:3.0f}-{dist_bins[k+1]:3.0f}m)"
    histMC_tA = ax.hist(data_tAcorr_dist[k][0], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, color=color, label='MC tA '+text, linestyle="dashed")
    histTR_tA = ax.hist(data_tAcorr_dist[k][1], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=1, color=color, label='MC tA '+text)
    histRD_tA = ax.hist(data_tAcorr_dist[k][2], bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, color=color, label='MC tA '+text)
ax.set_xlabel('Time at A of hits (ns)', fontsize=18)
ax.set_ylabel('Frequency / 20 ns', fontsize=18);
ax.legend(loc='upper right', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Hist_MCTRRD_Dists_tA_Differentially.pdf', dpi=600)
# plt.close(fig)










# ---------------------------------------------------------------------------------------------- #
# Plot and fit distributions of (corrected) data values:
# ---------------------------------------------------------------------------------------------- #

N_init     = [13500,23500,19500,22000,20000,19000,18000,17000,15000,13000,11000]
mu_init    = [110.0,110.0,105.0,105.0,100.0, 85.0, 70.0, 50.0, 50.0, 40.0, 30.0]
sigma_init = [ 50.0, 55.0, 65.0, 75.0, 80.0, 85.0, 90.0,100.0,100.0,105.0,110.0]
beta_init  = [ 0.95, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.60, 0.55, 0.55, 0.50]
m_init     = [ 2.20, 2.40, 2.60, 2.80, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.60]

# Resulting values of the tA resolution:
tA_sigma_fitMC = []
tA_sigmaunc_fitMC = []
tA_sigma_fitTR = []
tA_sigmaunc_fitTR = []
tA_sigma_fitRD = []
tA_sigmaunc_fitRD = []

# General fit parameters (for improving initial values):
tA_N     = np.zeros((11,3))
tA_mu    = np.zeros((11,3))
tA_sigma = np.zeros((11,3))
tA_beta  = np.zeros((11,3))
tA_m     = np.zeros((11,3))


# Loop over each distance bin and fit both MC and data:
# -----------------------------------------------------
# for j in range(len(dist_bins)-3) :
for j in range(11) :
    print(f"\n  Fitting MC, TRUTH, and DATA: Distance bin {j:d} ({dist_bins[j]:3.0f}-{dist_bins[j+1]:3.0f}m)")
    print("-------------------------------------------------------------------------------------------------")
    
    # Fit the distribution of tA (corrected to zero mean):
    x_MC, y_MC, sy_MC, fit_values_MC, fit_errors_MC, chi2_MC, Ndof_MC, Prob_MC, isvalid_MC, fittext_MC = \
        tAfit(data_tAcorr_dist[j][0], N_init[j], mu_init[j], sigma_init[j], beta_init[j], m_init[j], 2.0, True)
    tA_sigma_fitMC.append(fit_values_MC[2])
    tA_sigmaunc_fitMC.append(fit_errors_MC[2])
    tA_N[j][0], tA_mu[j][0], tA_sigma[j][0], tA_beta[j][0], tA_m[j][0] = fit_values_MC[0:5]

    x_TR, y_TR, sy_TR, fit_values_TR, fit_errors_TR, chi2_TR, Ndof_TR, Prob_TR, isvalid_TR, fittext_TR = \
        tAfit(data_tAcorr_dist[j][1], N_init[j], mu_init[j], sigma_init[j], beta_init[j], m_init[j], 2.0, True)
    tA_sigma_fitTR.append(fit_values_TR[2])
    tA_sigmaunc_fitTR.append(fit_errors_TR[2])
    tA_N[j][0], tA_mu[j][0], tA_sigma[j][0], tA_beta[j][0], tA_m[j][0] = fit_values_TR[0:5]

    x_RD, y_RD, sy_RD, fit_values_RD, fit_errors_RD, chi2_RD, Ndof_RD, Prob_RD, isvalid_RD, fittext_RD = \
        tAfit(data_tAcorr_dist[j][2], N_init[j], mu_init[j], sigma_init[j], beta_init[j], m_init[j], 2.0, True)
    tA_sigma_fitRD.append(fit_values_RD[2])
    tA_sigmaunc_fitRD.append(fit_errors_RD[2])
    tA_N[j][0], tA_mu[j][0], tA_sigma[j][0], tA_beta[j][0], tA_m[j][0] = fit_values_RD[0:5]


    # Plot the distributions:
    fig, ax = plt.subplots(figsize=(16, 8))
    hist_tA = ax.hist(data_tAcorr_dist[j][0], bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label=f"MC tA ({dist_bins[j]:3.0f} < $d_\gamma$ < {dist_bins[j+1]:3.0f}m)", color='blue')
    ax.plot(x_MC, func_CBpol0(x_MC, *fit_values_MC), 'blue', linewidth=2.0, label='CrystalBall + Const fit')
    ax.text(0.05, 0.95, nice_string_output(fittext_MC, 0), family='monospace', transform=ax.transAxes, fontsize=12, color='blue', verticalalignment='top')

    hist_tA = ax.hist(data_tAcorr_dist[j][1], bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label=f"Truth tA ({dist_bins[j]:3.0f} < $d_\gamma$ < {dist_bins[j+1]:3.0f}m)", color='black')
    ax.plot(x_TR, func_CBpol0(x_TR, *fit_values_TR), 'black', linewidth=2.0, label='CrystalBall + Const fit')
    ax.text(0.05, 0.65, nice_string_output(fittext_TR, 0), family='monospace', transform=ax.transAxes, fontsize=12, color='black', verticalalignment='top')

    hist_tA = ax.hist(data_tAcorr_dist[j][2], bins=Nbins_tA, range=(xmin_tA, xmax_tA), histtype='step', linewidth=2, label=f"Data tA ({dist_bins[j]:3.0f} < $d_\gamma$ < {dist_bins[j+1]:3.0f}m)", color='red')
    ax.plot(x_RD, func_CBpol0(x_RD, *fit_values_RD), 'red', linewidth=2.0, label='CrystalBall + Const fit')
    ax.text(0.05, 0.35, nice_string_output(fittext_RD, 0), family='monospace', transform=ax.transAxes, fontsize=12, color='red', verticalalignment='top')

    ax.set_xlabel('Time from Cherenkov emission to DOM in ns (corrected to have mean zero)', fontsize=18)
    ax.set_ylabel('Frequency / 10 ns', fontsize=18)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,0,4,1,5,2]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', fontsize=18)
    # ax.legend(loc='upper right', fontsize=18)
    ax.set_ylim(ymin=0.0)
    fig.tight_layout()
    fig.show()

    if (SaveFig) :
        fig.savefig(f"Figures_tAfits/Hist_MCTRRD_tAcorr_{dist_bins[j]:3.0f}-{dist_bins[j+1]:3.0f}m_Fitted.pdf", dpi=600)


# To get good initial values:
#print("\n-------------------------------------------------------------------------------------------------")
#print(tA_N.T)
#print(tA_mu.T)
#print(tA_sigma.T)
#print(tA_beta.T)
#print(tA_m.T)


# ---------------------------------------------------------------------------------------------- #
# Plot tA resolution as a function of distance:
# ---------------------------------------------------------------------------------------------- #

print("\n-------------------------------------------------------------------------------------------------")

# Determine the impact of reconstruction with uncertainty:
tA_sigma_FromReco = np.sqrt(np.square(tA_sigma_fitMC) - np.square(tA_sigma_fitTR))
tA_sigmaunc_FromReco = np.sqrt(np.square(tA_sigmaunc_fitMC) + np.square(tA_sigmaunc_fitTR))

# Subtract effect of reconstruction from MC and Data:
tA_sigma_NoReco_MC = np.sqrt(np.square(tA_sigma_fitMC) - np.square(tA_sigma_FromReco))
tA_sigmaunc_NoReco_MC = np.sqrt(np.square(tA_sigmaunc_fitMC) + np.square(tA_sigmaunc_FromReco))
tA_sigma_NoReco_RD = np.sqrt(np.square(tA_sigma_fitRD) - np.square(tA_sigma_FromReco))
tA_sigmaunc_NoReco_RD = np.sqrt(np.square(tA_sigmaunc_fitRD) + np.square(tA_sigmaunc_FromReco))

# Also, determine the size of the uncertainty from propagation in ice:
tA_sigma_Ice_MC = np.sqrt(np.square(tA_sigma_NoReco_MC) - np.square(tA_sigma_NoReco_MC[0]))
tA_sigmaunc_Ice_MC = np.sqrt(np.square(tA_sigmaunc_NoReco_MC) + np.square(tA_sigmaunc_NoReco_MC[0]))
tA_sigma_Ice_RD = np.sqrt(np.square(tA_sigma_NoReco_RD) - np.square(tA_sigma_NoReco_RD[0]))
tA_sigmaunc_Ice_RD = np.sqrt(np.square(tA_sigmaunc_NoReco_RD) + np.square(tA_sigmaunc_NoReco_RD[0]))

# Weighted mean of the six first measurements of the impact of reconstruction:
x = tA_sigma_FromReco[0:6]
w = 1.0 / np.power(tA_sigmaunc_FromReco[0:6],2)
wmean = (x * w).sum() / w.sum()
ewmean = np.sqrt(1.0 / w.sum())

# Weighted mean of all the measurements of the impact of reconstruction:
x2 = tA_sigma_FromReco[:]
w2 = 1.0 / np.power(tA_sigmaunc_FromReco[:],2)
wmean2 = (x2 * w2).sum() / w2.sum()
ewmean2 = np.sqrt(1.0 / w2.sum())

# The x-values should be centered in the dist_bins:
# x_dist_bins = dist_bins[0:11]... half way!!!

# Draw the different "conclusions" step-wise in a single plot:
fig, ax = plt.subplots(figsize=(16, 8))
plotMC_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitMC, yerr=tA_sigmaunc_fitMC, linewidth=2, label='MC tA resolution', color='blue')
plotTR_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitTR, yerr=tA_sigmaunc_fitTR, linewidth=2, label='Truth tA resolution', color='black')
plotRD_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitRD, yerr=tA_sigmaunc_fitRD, linewidth=2, label='Data tA resolution', color='red')
ax.set_xlabel('Distance travelled by cherenkov photon to DOM (m)', fontsize=18)
ax.set_ylabel('Time resolution of tA for these hits (ns)', fontsize=18);
ax.legend(loc='upper left', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Plot_MCTRRD_tAresolution_vs_Distance.pdf', dpi=600)


# Draw the different "conclusions" step-wise in a single plot:
fig, ax = plt.subplots(figsize=(16, 8))
plotMC_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitMC, yerr=tA_sigmaunc_fitMC, linewidth=2, label='MC tA resolution', color='blue')
plotTR_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitTR, yerr=tA_sigmaunc_fitTR, linewidth=2, label='Truth tA resolution (MC no reco effect)', color='black')
plotRD_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitRD, yerr=tA_sigmaunc_fitRD, linewidth=2, label='Data tA resolution', color='red')
plot_tA_FromReco = ax.errorbar(dist_bins[0:11], tA_sigma_FromReco, yerr=tA_sigmaunc_FromReco, linewidth=2, label='tA resolution from reco effect (MC)', color='blue', linestyle="dashed")
#plot_tA_NoReco_MC = ax.errorbar(dist_bins[0:11], tA_sigma_NoReco_MC, yerr=tA_sigmaunc_NoReco_MC, linewidth=2, label='MC tA resolution (no reco effect)', color='blue', linestyle="dashed")
plot_tA_NoReco_RD = ax.errorbar(dist_bins[0:11], tA_sigma_NoReco_RD, yerr=tA_sigmaunc_NoReco_RD, linewidth=2, label='Data tA resolution (no reco effect)', color='red', linestyle="dashed")
#plot_tA_Ice_MC = ax.errorbar(dist_bins[0:11], tA_sigma_Ice_MC, yerr=tA_sigmaunc_Ice_MC, linewidth=2, label='MC tA resolution from the ice', color='blue', linestyle="dotted")
#plot_tA_Ice_RD = ax.errorbar(dist_bins[0:11], tA_sigma_Ice_RD, yerr=tA_sigmaunc_Ice_RD, linewidth=2, label='Data tA resolution from the ice', color='red', linestyle="dotted")
ax.set_xlabel('Distance travelled by cherenkov photon to DOM (m)', fontsize=18)
ax.set_ylabel('Time resolution of tA for these hits (ns)', fontsize=18);
ax.legend(loc='upper left', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Plot_MCTRRD_tAresolution_vs_Distance_RecoEffect.pdf', dpi=600)


# Draw the different "conclusions" step-wise in a single plot:
fig, ax = plt.subplots(figsize=(16, 8))
plotMC_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitMC, yerr=tA_sigmaunc_fitMC, linewidth=2, label='MC tA resolution', color='blue')
plotTR_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitTR, yerr=tA_sigmaunc_fitTR, linewidth=2, label='Truth tA resolution (no reco effect)', color='black')
plotRD_tA = ax.errorbar(dist_bins[0:11], tA_sigma_fitRD, yerr=tA_sigmaunc_fitRD, linewidth=2, label='Data tA resolution', color='red')
plot_tA_FromReco = ax.errorbar(dist_bins[0:11], tA_sigma_FromReco, yerr=tA_sigmaunc_FromReco, linewidth=2, label='tA resolution from reco effect (MC)', color='blue', linestyle="dashed")
#plot_tA_NoReco_MC = ax.errorbar(dist_bins[0:11], tA_sigma_NoReco_MC, yerr=tA_sigmaunc_NoReco_MC, linewidth=2, label='MC tA resolution (no reco effect)', color='blue', linestyle="dashed")
plot_tA_NoReco_RD = ax.errorbar(dist_bins[0:11], tA_sigma_NoReco_RD, yerr=tA_sigmaunc_NoReco_RD, linewidth=2, label='Data tA resolution (no reco effect)', color='red', linestyle="dashed")
plot_tA_Ice_MC = ax.errorbar(dist_bins[0:11], tA_sigma_Ice_MC, yerr=tA_sigmaunc_Ice_MC, linewidth=2, label='MC tA resolution from the ice', color='blue', linestyle="dotted")
plot_tA_Ice_RD = ax.errorbar(dist_bins[0:11], tA_sigma_Ice_RD, yerr=tA_sigmaunc_Ice_RD, linewidth=2, label='Data tA resolution from the ice', color='red', linestyle="dotted")
ax.set_xlabel('Distance travelled by cherenkov photon to DOM (m)', fontsize=18)
ax.set_ylabel('Time resolution of tA for these hits (ns)', fontsize=18);
ax.legend(loc='upper left', fontsize=18)
ax.set_ylim(ymin=0.0)
fig.tight_layout()
fig.show()
if (SaveFig) :
    fig.savefig('Figures/Plot_MCTRRD_tAresolution_vs_Distance_ResolutionFromIce.pdf', dpi=600)


# ---------------------------------------------------------------------------------------------- #
# Conclusions on the difference between Truth and Reco:
# ---------------------------------------------------------------------------------------------- #

print(f"\n  The width difference of the full fit between MC and Truth is: {np.sqrt(fit_values_MC0[2]**2-fit_values_TR0[2]**2):4.2f} +- {np.sqrt(fit_errors_MC0[2]**2+fit_errors_TR0[2]**2):4.2f} ns")
print(f"  The corresponding value obtained from the first bin (0-10m) is: {tA_sigma_FromReco[0]:4.2f} +- {tA_sigmaunc_FromReco[0]:4.2f} ns")
print(f"  The average value obtained from the first six bin (0-60m) is: {wmean:4.2f} +- {ewmean:4.2f} ns")
print(f"  The average value obtained from all the bins (0-110m) is: {wmean2:4.2f} +- {ewmean2:4.2f} ns")


# ---------------------------------------------------------------------------------------------- #
input()
# ---------------------------------------------------------------------------------------------- #

"""
  Fitting MC, TRUTH, and DATA: Distance bin 0 (  0- 10m)
-------------------------------------------------------------------------------------------------
Fit value: N = 13799.67331 +/- 130.46054
Fit value: mu = 109.89268 +/- 0.62385
Fit value: sigma = 50.34717 +/- 0.66524
Fit value: beta = 1.19582 +/- 0.02881
Fit value: m = 2.06179 +/- 0.08090
Fit value: cst = 2.03884 +/- 0.30254
Chi2 value: 439.1   Ndof = 103    Prob(Chi2,Ndof) = 0.000
Fit value: N = 12218.72572 +/- 121.56684
Fit value: mu = 112.98614 +/- 0.61869
Fit value: sigma = 48.37313 +/- 0.65071
Fit value: beta = 1.24321 +/- 0.03359
Fit value: m = 2.15564 +/- 0.10048
Fit value: cst = 2.04288 +/- 0.31249
Chi2 value: 389.2   Ndof = 102    Prob(Chi2,Ndof) = 0.000
Fit value: N = 10985.85957 +/- 111.85664
Fit value: mu = 109.45951 +/- 0.82403
Fit value: sigma = 54.37558 +/- 0.80635
Fit value: beta = 1.05742 +/- 0.03386
Fit value: m = 2.57152 +/- 0.13579
Fit value: cst = 1.76146 +/- 0.28976
Chi2 value: 331.3   Ndof = 101    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 1 ( 10- 20m)
-------------------------------------------------------------------------------------------------
Fit value: N = 23860.79231 +/- 171.77722
Fit value: mu = 114.33637 +/- 0.58509
Fit value: sigma = 55.06220 +/- 0.61243
Fit value: beta = 1.06000 +/- 0.02224
Fit value: m = 2.09668 +/- 0.06066
Fit value: cst = 3.33053 +/- 0.34493
Chi2 value: 647.2   Ndof = 110    Prob(Chi2,Ndof) = 0.000
Fit value: N = 22050.32821 +/- 163.23178
Fit value: mu = 115.02009 +/- 0.50702
Fit value: sigma = 51.29379 +/- 0.56314
Fit value: beta = 1.10784 +/- 0.02150
Fit value: m = 2.01616 +/- 0.05395
Fit value: cst = 2.05977 +/- 0.30324
Chi2 value: 868.0   Ndof = 107    Prob(Chi2,Ndof) = 0.000
Fit value: N = 19889.16069 +/- 149.60915
Fit value: mu = 116.60064 +/- 0.74176
Fit value: sigma = 59.50610 +/- 0.72433
Fit value: beta = 0.87242 +/- 0.02114
Fit value: m = 2.98099 +/- 0.11895
Fit value: cst = 3.04058 +/- 0.34696
Chi2 value: 504.5   Ndof = 110    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 2 ( 20- 30m)
-------------------------------------------------------------------------------------------------
Fit value: N = 23880.55203 +/- 162.14296
Fit value: mu = 112.56884 +/- 0.69678
Fit value: sigma = 62.84348 +/- 0.73253
Fit value: beta = 0.95096 +/- 0.01969
Fit value: m = 2.16998 +/- 0.06046
Fit value: cst = 3.17806 +/- 0.32234
Chi2 value: 694.5   Ndof = 111    Prob(Chi2,Ndof) = 0.000
Fit value: N = 23630.30817 +/- 167.01499
Fit value: mu = 112.16768 +/- 0.65017
Fit value: sigma = 59.87655 +/- 0.71491
Fit value: beta = 0.95937 +/- 0.01987
Fit value: m = 2.17657 +/- 0.06078
Fit value: cst = 2.92185 +/- 0.32625
Chi2 value: 731.3   Ndof = 111    Prob(Chi2,Ndof) = 0.000
Fit value: N = 20400.23632 +/- 150.61104
Fit value: mu = 113.32180 +/- 0.90306
Fit value: sigma = 64.55608 +/- 0.86350
Fit value: beta = 0.80281 +/- 0.02084
Fit value: m = 2.98605 +/- 0.11538
Fit value: cst = 2.56655 +/- 0.29212
Chi2 value: 564.6   Ndof = 115    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 3 ( 30- 40m)
-------------------------------------------------------------------------------------------------
Fit value: N = 22067.11303 +/- 158.64389
Fit value: mu = 105.75808 +/- 0.86898
Fit value: sigma = 71.05955 +/- 0.87933
Fit value: beta = 0.86772 +/- 0.01969
Fit value: m = 2.26197 +/- 0.07105
Fit value: cst = 2.65639 +/- 0.29416
Chi2 value: 601.9   Ndof = 115    Prob(Chi2,Ndof) = 0.000
Fit value: N = 22430.73618 +/- 169.54363
Fit value: mu = 105.13668 +/- 0.80912
Fit value: sigma = 67.82594 +/- 0.87805
Fit value: beta = 0.86331 +/- 0.01965
Fit value: m = 2.24642 +/- 0.06779
Fit value: cst = 2.68456 +/- 0.30773
Chi2 value: 719.1   Ndof = 113    Prob(Chi2,Ndof) = 0.000
Fit value: N = 19682.14285 +/- 150.62186
Fit value: mu = 106.99713 +/- 1.13379
Fit value: sigma = 74.28499 +/- 1.07402
Fit value: beta = 0.72923 +/- 0.02020
Fit value: m = 3.14513 +/- 0.14563
Fit value: cst = 2.83124 +/- 0.32854
Chi2 value: 470.7   Ndof = 112    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 4 ( 40- 50m)
-------------------------------------------------------------------------------------------------
Fit value: N = 20343.90512 +/- 159.90524
Fit value: mu = 97.56313 +/- 1.14963
Fit value: sigma = 75.89604 +/- 1.11060
Fit value: beta = 0.71609 +/- 0.01934
Fit value: m = 2.62601 +/- 0.09878
Fit value: cst = 2.76152 +/- 0.30298
Chi2 value: 511.9   Ndof = 115    Prob(Chi2,Ndof) = 0.000
Fit value: N = 21118.73147 +/- 166.10162
Fit value: mu = 98.68929 +/- 1.08642
Fit value: sigma = 72.23901 +/- 1.04989
Fit value: beta = 0.73219 +/- 0.01951
Fit value: m = 2.44157 +/- 0.08424
Fit value: cst = 2.62381 +/- 0.30638
Chi2 value: 569.4   Ndof = 113    Prob(Chi2,Ndof) = 0.000
Fit value: N = 18684.02539 +/- 148.18837
Fit value: mu = 98.11241 +/- 1.29132
Fit value: sigma = 79.19386 +/- 1.16198
Fit value: beta = 0.69255 +/- 0.02009
Fit value: m = 3.10929 +/- 0.14253
Fit value: cst = 3.07927 +/- 0.32210
Chi2 value: 396.1   Ndof = 116    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 5 ( 50- 60m)
-------------------------------------------------------------------------------------------------
Fit value: N = 19196.63124 +/- 156.52434
Fit value: mu = 83.77016 +/- 1.49274
Fit value: sigma = 83.51535 +/- 1.33948
Fit value: beta = 0.64067 +/- 0.01964
Fit value: m = 3.00419 +/- 0.14158
Fit value: cst = 3.28855 +/- 0.33059
Chi2 value: 377.6   Ndof = 115    Prob(Chi2,Ndof) = 0.000
Fit value: N = 19606.23601 +/- 160.49792
Fit value: mu = 83.30180 +/- 1.37952
Fit value: sigma = 79.56910 +/- 1.20942
Fit value: beta = 0.65461 +/- 0.01951
Fit value: m = 2.74818 +/- 0.11802
Fit value: cst = 3.37419 +/- 0.33666
Chi2 value: 425.2   Ndof = 115    Prob(Chi2,Ndof) = 0.000
Fit value: N = 17318.14158 +/- 142.28984
Fit value: mu = 87.58295 +/- 1.72604
Fit value: sigma = 84.01346 +/- 1.47376
Fit value: beta = 0.58525 +/- 0.01962
Fit value: m = 3.76210 +/- 0.22324
Fit value: cst = 2.83791 +/- 0.30133
Chi2 value: 408.5   Ndof = 117    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 6 ( 60- 70m)
-------------------------------------------------------------------------------------------------
Fit value: N = 17788.87166 +/- 149.38823
Fit value: mu = 70.43994 +/- 1.85452
Fit value: sigma = 91.11197 +/- 1.59373
Fit value: beta = 0.61813 +/- 0.02164
Fit value: m = 2.90560 +/- 0.15513
Fit value: cst = 2.86704 +/- 0.30249
Chi2 value: 364.3   Ndof = 117    Prob(Chi2,Ndof) = 0.000
Fit value: N = 18133.95657 +/- 154.17679
Fit value: mu = 68.61692 +/- 1.85636
Fit value: sigma = 88.79526 +/- 1.55234
Fit value: beta = 0.59878 +/- 0.02051
Fit value: m = 3.04857 +/- 0.16161
Fit value: cst = 2.68450 +/- 0.29380
Chi2 value: 350.2   Ndof = 117    Prob(Chi2,Ndof) = 0.000
Fit value: N = 16062.76225 +/- 139.83365
Fit value: mu = 69.79218 +/- 2.06173
Fit value: sigma = 93.85288 +/- 1.79296
Fit value: beta = 0.55183 +/- 0.01949
Fit value: m = 4.31932 +/- 0.31600
Fit value: cst = 3.54513 +/- 0.33734
Chi2 value: 297.8   Ndof = 116    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 7 ( 70- 80m)
-------------------------------------------------------------------------------------------------
Fit value: N = 16797.27347 +/- 151.99744
Fit value: mu = 47.91280 +/- 2.36228
Fit value: sigma = 101.79405 +/- 1.97519
Fit value: beta = 0.58519 +/- 0.02273
Fit value: m = 3.34537 +/- 0.22177
Fit value: cst = 2.93855 +/- 0.32081
Chi2 value: 281.6   Ndof = 115    Prob(Chi2,Ndof) = 0.000
Fit value: N = 17056.27633 +/- 149.81666
Fit value: mu = 55.12913 +/- 2.29525
Fit value: sigma = 95.03212 +/- 1.88351
Fit value: beta = 0.54258 +/- 0.02124
Fit value: m = 3.46683 +/- 0.23826
Fit value: cst = 3.60126 +/- 0.34211
Chi2 value: 290.7   Ndof = 116    Prob(Chi2,Ndof) = 0.000
Fit value: N = 14665.90443 +/- 137.12323
Fit value: mu = 58.42300 +/- 2.61558
Fit value: sigma = 98.75287 +/- 2.02742
Fit value: beta = 0.52956 +/- 0.02275
Fit value: m = 4.28794 +/- 0.38180
Fit value: cst = 4.09468 +/- 0.35353
Chi2 value: 253.4   Ndof = 118    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 8 ( 80- 90m)
-------------------------------------------------------------------------------------------------
Fit value: N = 15286.93688 +/- 142.36641
Fit value: mu = 42.03297 +/- 2.71354
Fit value: sigma = 101.63127 +/- 2.19293
Fit value: beta = 0.50424 +/- 0.02086
Fit value: m = 3.78258 +/- 0.30383
Fit value: cst = 3.10003 +/- 0.30708
Chi2 value: 242.1   Ndof = 118    Prob(Chi2,Ndof) = 0.000
Fit value: N = 15308.86458 +/- 145.15132
Fit value: mu = 42.11670 +/- 2.69487
Fit value: sigma = 99.53582 +/- 2.17667
Fit value: beta = 0.48562 +/- 0.02032
Fit value: m = 3.88640 +/- 0.31346
Fit value: cst = 2.92346 +/- 0.30066
Chi2 value: 273.0   Ndof = 117    Prob(Chi2,Ndof) = 0.000
Fit value: N = 13270.78287 +/- 133.40803
Fit value: mu = 33.83709 +/- 3.19042
Fit value: sigma = 110.84432 +/- 2.56273
Fit value: beta = 0.51981 +/- 0.02431
Fit value: m = 5.30477 +/- 0.64750
Fit value: cst = 4.50948 +/- 0.37805
Chi2 value: 250.3   Ndof = 118    Prob(Chi2,Ndof) = 0.000

  Fitting MC, TRUTH, and DATA: Distance bin 9 ( 90-100m)
-------------------------------------------------------------------------------------------------
Fit value: N = 13511.81193 +/- 139.33380
Fit value: mu = 14.09174 +/- 3.51484
Fit value: sigma = 113.44175 +/- 2.68394
Fit value: beta = 0.51897 +/- 0.02662
Fit value: m = 4.09818 +/- 0.46342
Fit value: cst = 3.25529 +/- 0.31839
Chi2 value: 231.4   Ndof = 118    Prob(Chi2,Ndof) = 0.000
Fit value: N = 13951.41827 +/- 150.42115
Fit value: mu = 19.57677 +/- 3.04047
Fit value: sigma = 108.31814 +/- 2.45103
Fit value: beta = 0.52388 +/- 0.02468
Fit value: m = 3.42717 +/- 0.30507
Fit value: cst = 2.90836 +/- 0.29745
Chi2 value: 228.5   Ndof = 117    Prob(Chi2,Ndof) = 0.000
Fit value: N = 12127.24562 +/- 126.01791
Fit value: mu = 25.64448 +/- 3.72131
Fit value: sigma = 111.10708 +/- 2.84771
Fit value: beta = 0.44072 +/- 0.02236
Fit value: m = 7.51243 +/- 1.35160
Fit value: cst = 4.24473 +/- 0.36900
Chi2 value: 170.5   Ndof = 117    Prob(Chi2,Ndof) = 0.001

  Fitting MC, TRUTH, and DATA: Distance bin 10 (100-110m)
-------------------------------------------------------------------------------------------------
Fit value: N = 11935.95861 +/- 133.48217
Fit value: mu = 8.58295 +/- 3.82468
Fit value: sigma = 110.23186 +/- 3.03938
Fit value: beta = 0.42742 +/- 0.02304
Fit value: m = 5.59692 +/- 0.85213
Fit value: cst = 4.37352 +/- 0.37654
Chi2 value: 227.0   Ndof = 116    Prob(Chi2,Ndof) = 0.000
Fit value: N = 11795.39540 +/- 132.22841
Fit value: mu = 10.04675 +/- 3.63922
Fit value: sigma = 105.83109 +/- 2.80134
Fit value: beta = 0.43938 +/- 0.02355
Fit value: m = 4.48762 +/- 0.54546
Fit value: cst = 3.81756 +/- 0.34399
Chi2 value: 190.4   Ndof = 117    Prob(Chi2,Ndof) = 0.000
Fit value: N = 10629.21699 +/- 109.69598
Fit value: mu = 3.14255 +/- 3.50551
Fit value: sigma = 118.25341 +/- 2.95399
Fit value: beta = 0.45848 +/- 0.01222
Fit value: m = 6.57043 +/- 0.22765
Fit value: cst = 3.50759 +/- 0.31515
Chi2 value: 207.3   Ndof = 118    Prob(Chi2,Ndof) = 0.000



# ---------------------------------------------------------------------------------------------- #
def tAfit_adv(data, N_init, mu_init, sigma_init, beta_init, m_init, fCB_init, sigma2_init, cst_init, verbose) :

    # Put data into histogram:
    counts, bin_edges = np.histogram(data, bins=binning_tA)
    X = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0
    unc_counts = np.sqrt(counts)

    # Prepare and perform the Chi2 fit:
    chi2reg = Chi2Regression(func_CBGpol0, X[counts>0], counts[counts>0], unc_counts[counts>0])
    minuit_chi2 = Minuit(chi2reg, N=N_init, mu=mu_init, sigma=sigma_init, beta=beta_init, m=m_init, fCB=fCB_init, sigma2=sigma2_init, cst=cst_init)
    minuit_chi2.errordef = 1.0     # ChiSquare fit
    minuit_chi2.migrad()
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
         'fCB'       : "{:.4f} +/- {:.4f}".format(minuit_chi2.values['fCB'], minuit_chi2.errors['fCB']),
         'sigma2'    : "{:.1f} +/- {:.1f}".format(minuit_chi2.values['sigma2'], minuit_chi2.errors['sigma2']),
         'cst'       : "{:.1f} +/- {:.1f}".format(minuit_chi2.values['cst'], minuit_chi2.errors['cst'])}

    # Return histogram values, fit parameters and errors, Chi2 value, Ndof, and Chi2 probability, fit validity, and fit result string:
    return X, counts, unc_counts, minuit_chi2.values[:], minuit_chi2.errors[:], chi2_value, Ndof_value, Prob_value, minuit_chi2.fmin.is_valid, d


#x_MC0, y_MC0, sy_MC0, fit_values_MC0, fit_errors_MC0, chi2_MC0, Ndof_MC0, Prob_MC0, isvalid_MC0, fittext_MC0 = tAfit_adv(data_tAcorr[0], 264550.6, 97.4, 68.37, 0.451, 2.53, 0.99, 50.0, 244.0, True)
#ax.plot(x_MC0, func_CBGpol0(x_MC0, *fit_values_MC0), 'orange', linewidth=1.0, label='CrystalBall + Gauss + Const fit ()')
#ax.text(0.05, 0.95, nice_string_output(fittext_MC0, 0), family='monospace', transform=ax.transAxes, fontsize=13, color='orange', verticalalignment='top')


"""

