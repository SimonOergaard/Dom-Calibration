# ---------------------------------------------------------------------------------------------- #
#                                Cherenkov Photon Calculation
# ---------------------------------------------------------------------------------------------- #
#
#   Inspection of cherenkov photon timing
#   
#   Author: Troels Petersen (NBI)
#   Date: 29th of June 2023 (first version)
#   
# ---------------------------------------------------------------------------------------------- #
"""
"""   
# ---------------------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SaveFig = True

# ---------------------------------------------------------------------------------------------- #
# 
# ---------------------------------------------------------------------------------------------- #

data1 = pd.read_csv("PhotonTimes1.csv")
data2 = pd.read_csv("PhotonTimes2.csv")
data3 = pd.read_csv("PhotonTimes3.csv")

data1 = data1.sort_values("dom_time")
print(data1.keys())

with pd.option_context('display.max_rows', None, 'display.max_columns', None) :
    print(data1)
    
    


# ---------------------------------------------------------------------------------------------- #
# Time: Cherenkov emission to DOM:
# ---------------------------------------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(12, 6))
hist_tCD1 = ax.hist(data1['t_cher_dom'].values, bins=100, range=(0.0, 5000.0), histtype='step', linewidth=2, label='Data1: tCherDOM (ns)', color='blue')
hist_tCD2 = ax.hist(data2['t_cher_dom'].values, bins=100, range=(0.0, 5000.0), histtype='step', linewidth=2, label='Data2: tCherDOM (ns)', color='red')
#hist_tCD3 = ax.hist(data3['t_cher_dom'].values, bins=100, range=(0.0, 5000.0), histtype='step', linewidth=2, label='Data3: tCherDOM (ns)', color='black')
ax.set_xlabel("Time from Cherenkov emission to DOM (ns)", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.legend(loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()

if (SaveFig) :
    fig.savefig('Hist_timeCherenkovDOM.pdf', dpi=600)



# ---------------------------------------------------------------------------------------------- #
# Time: Cherenkov emission to DOM:
# ---------------------------------------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(12, 6))
hist_tAC1 = ax.hist(data1['t_A_cher'].values, bins=100, range=(0.0, 5000.0), histtype='step', linewidth=2, label='Data1: t_AtoCher (ns)', color='blue')
hist_tAC2 = ax.hist(data2['t_A_cher'].values, bins=100, range=(0.0, 5000.0), histtype='step', linewidth=2, label='Data2: t_AtoCher (ns)', color='red')
#hist_tAC3 = ax.hist(data3['t_A_cher'].values, bins=100, range=(0.0, 5000.0), histtype='step', linewidth=2, label='Data3: t_AtoCher (ns)', color='black')
ax.set_xlabel("Time from muon stopping point A to Cherenkov emission (ns)", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.legend(loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()

if (SaveFig) :
    fig.savefig('Hist_timePointACherenkov.pdf', dpi=600)



# ---------------------------------------------------------------------------------------------- #
# Time: Point of arrival to A
# ---------------------------------------------------------------------------------------------- #

Nbins, xmin, xmax = 100, 5000, 15000
fig, ax = plt.subplots(figsize=(12, 6))
hist_tAC1 = ax.hist(data1['t_A'].values, bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Data1: t_A (ns)', color='blue')
hist_tAC2 = ax.hist(data2['t_A'].values, bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Data2: t_A (ns)', color='red')
#hist_tAC3 = ax.hist(data3['t_A'].values, bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, label='Data3: t_A (ns)', color='black')
ax.set_xlabel("Estimated time of muon stopping at point A (ns)", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.legend(loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()

if (SaveFig) :
    fig.savefig('Hist_timeAtPointA.pdf', dpi=600)



# ---------------------------------------------------------------------------------------------- #
# Biases: Does dt_A change with z?
# ---------------------------------------------------------------------------------------------- #

# Function that takes dataframe and two ranges (in ns) as input, and returns an estimate of the peak value and width in t_A:
def FindMeanTimeAtA(data, rangeMedian, rangeMean, verbose=False) :
    median_t_A = data['t_A'].median(axis=0)
    if (verbose) : print(f"  Median time at A is: {median_t_A:7.1f} ns")
    mean_t_A = data.loc[np.abs(data["t_A"]-median_t_A) < rangeMedian, "t_A"].mean(axis=0)
    if (verbose) : print(f"  Mean time of values 400ns around A is: {mean_t_A:7.1f} ns")
    std_t_A = data.loc[np.abs(data["t_A"]-mean_t_A) < rangeMean, "t_A"].std(axis=0)
    if (verbose) : print(f"  Standard Deviation of values 300ns around mean time is: {std_t_A:7.1f} ns")
    return mean_t_A, std_t_A

mean_t_A1, std_t_A1 = FindMeanTimeAtA(data1, 400.0, 300.0, True)
mean_t_A2, std_t_A2 = FindMeanTimeAtA(data2, 400.0, 300.0, True)


# Given an estimate of time at point A, calculate the change/bias in time dt_A:
dt_A1 = data1.loc[np.abs(data1["t_A"]-mean_t_A1) < 300.0, "t_A"] - mean_t_A1
z1    = data1.loc[np.abs(data1["t_A"]-mean_t_A1) < 300.0, "photon_z"]
dt_A2 = data2.loc[np.abs(data2["t_A"]-mean_t_A2) < 300.0, "t_A"] - mean_t_A2
z2    = data2.loc[np.abs(data2["t_A"]-mean_t_A2) < 300.0, "photon_z"]
print(dt_A1.mean())
print(z1.mean())

# Plot the bias as a function of depth:
fig, ax = plt.subplots(figsize=(8, 10))
hist_dtA1vsZ = ax.scatter(dt_A1, z1, label='Data1', color='blue')
hist_dtA2vsZ = ax.scatter(dt_A2, z2, label='Data2', color='red')
ax.set_xlabel("Bias of muon stopping time at point A (ns)", fontsize=14)
ax.set_ylabel("Photon z (m)", fontsize=14)
ax.set_ylim(-400, -200)

ax.vlines(x=0.0, ymin=-390, ymax=-210, colors='black', linestyle="dashed")

ax.legend(loc='upper right', fontsize=18)
fig.tight_layout()
fig.show()

if (SaveFig) :
    fig.savefig('Hist2D_timebiasAtPointAvsZ.pdf', dpi=600)




# ---------------------------------------------------------------------------------------------- #
# input()
# ---------------------------------------------------------------------------------------------- #
input()
