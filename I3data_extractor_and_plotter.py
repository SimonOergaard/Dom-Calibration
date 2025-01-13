#!/usr/bin/env python
#"eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)"
print('starting_script')
import os
import numpy as np
import matplotlib as mpl
from collections import OrderedDict

# Set custom color cycle
custom_colors = [    
    '#0081C8',  # Olynmpic Blue
    '#FCB131', # Olympic Yellow
    '#000000', # Olympic Black
    '#00A651',  # Olympic Green
    '#EE334E',  # Olynmpic Red
    '#F47835',  # Olympic Orange
    '#7C878E',  # Olympic Grey
    '#C8102E',  # Olympic Red
    '#EF3340',  # Olympic Red
    '#FFD662',  # Olympic Yellow
    '#00539C',  # Olympic Blue
]


mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colors)
assert 'I3_BUILD' in os.environ.keys()


def harvest_quantities(frame, data_container=None):
   # from icetray import icetray,dataclasses, dataio
    assert isinstance(data_container, (dict,OrderedDict))
    #Extracting from I3Map<OMKey, vector<I3RecoPulse> >  
    if frame.Has('SplitInIcePulsesCleaned'):
        #Extracting the charge, wher The columns are (String, OM, PMT, Time, Charge, Width), with one row per pulse. 
        data_container['Charge'].append(np.asarray(frame['SplitInIcePulsesCleaned'], dtype=np.float32)[:,4])
        data_container['time'].append(np.asarray(frame['SplitInIcePulsesCleaned'], dtype=np.float32)[:,3])
        data_container['mean_charge'].append(np.mean(np.asarray(frame['SplitInIcePulsesCleaned'], dtype=np.float32)[:,4])/len(np.asarray(frame['SplitInIcePulsesCleaned'], dtype=np.float32)[:,4]))
        data_container['OMKey'].append(np.asarray(frame['SplitInIcePulsesCleaned'], dtype=np.float32)[:,1])
        data_container['string'] = np.asarray(frame['SplitInIcePulsesCleaned'], dtype=np.float32)[:,0]
        
    else:
        data_container['Charge'].append([np.NaN])
        data_container['time'].append([np.NaN])
        data_container['mean_charge'].append(np.NaN)
        data_container['OMKey'].append([np.NaN])
        data_container['string'] = [np.NaN]
        
    # Extracting from I3Map with no vector in it.    
    if frame.Has('I3MCWeightDict'):
        data_container['GENIEWeight'].append(frame['I3MCWeightDict']['GENIEWeight'])
    else:
        data_container['GENIEWeight'].append(np.NaN)
    
    
    if frame.Has('I3MCTree'):
        #print('I3MCTree found')
        data_container['X_position_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].pos.x))
        data_container['Y_position_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].pos.y))
        data_container['Z_position_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].pos.z))
        data_container['Azimuth_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].dir.azimuth))
        data_container['Zenith_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].dir.zenith))
        data_container['Energy_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].energy))
        data_container['Time_Prim'].append(np.asarray(frame['I3MCTree'].get_primaries()[0].time))
    # Assuming 'frame' and 'data_container' are already defined
    nu = frame['I3MCTree'].get_primaries()[0]

    # Check if the primary particle is a neutrino with pdg_encoding 14
    if np.abs(nu.pdg_encoding) == 14:
        secoundaries = frame["I3MCTree"].get_daughters(nu)
        
        # Check if the first secondary particle is a muon with pdg_encoding 13
        if len(secoundaries) > 0 and np.abs(secoundaries[0].pdg_encoding) == 13:
            # Store data if the condition is met
            data_container['X_position_secoundary'].append(np.asarray(secoundaries[0].pos.x))
            data_container['Y_position_secoundary'].append(np.asarray(secoundaries[0].pos.y))
            data_container['Z_position_secoundary'].append(np.asarray(secoundaries[0].pos.z))
            data_container['Azimuth_secoundary'].append(np.asarray(secoundaries[0].dir.azimuth))
            data_container['Zenith_secoundary'].append(np.asarray(secoundaries[0].dir.zenith))
            data_container['length_secoundary'].append(np.asarray(secoundaries[0].length))
            data_container['Energy_secoundary'].append(np.asarray(secoundaries[0].energy))
        else:
            # If the condition is not met, append NaN or handle accordingly
            data_container['X_position_secoundary'].append(np.NaN)
            data_container['Y_position_secoundary'].append(np.NaN)
            data_container['Z_position_secoundary'].append(np.NaN)
            data_container['Azimuth_secoundary'].append(np.NaN)
            data_container['Zenith_secoundary'].append(np.NaN)
            data_container['length_secoundary'].append(10000)
            data_container['Energy_secoundary'].append(np.NaN)
    else:
        # Handling for primary particles that do not meet the condition
        data_container['X_position_Prim'].append(np.NaN)
        data_container['Y_position_Prim'].append(np.NaN)
        data_container['Z_position_Prim'].append(np.NaN)
        data_container['Azimuth_Prim'].append(np.NaN)
        data_container['Zenith_Prim'].append(np.NaN)
        data_container['Energy_Prim'].append(np.NaN)
        # Secondary particle (all NaN or default values as needed)
        data_container['X_position_secoundary'].append(np.NaN)
        data_container['Y_position_secoundary'].append(np.NaN)
        data_container['Z_position_secoundary'].append(np.NaN)
        data_container['Azimuth_secoundary'].append(np.NaN)
        data_container['Zenith_secoundary'].append(np.NaN)
        data_container['length_secoundary'].append(np.NaN)
        data_container['Energy_secoundary'].append(np.NaN)


def harvest_geometry(frame, data_container=None):
    assert isinstance(data_container, (dict, OrderedDict))
    if frame.Has('I3Geometry'):
        print('I3Geometry found')
        
        # Initialize the lists if they don't exist
        if 'X_position_G' not in data_container:
            data_container['X_position_G'] = []
        if 'Y_position_G' not in data_container:
            data_container['Y_position_G'] = []
        if 'Z_position_G' not in data_container:
            data_container['Z_position_G'] = []
        if 'StringNumber' not in data_container:
            data_container['StringNumber'] = []
        if 'OMNumber' not in data_container:
            data_container['OMNumber'] = []

        # Iterate over all omgeo values and extract the x, y, and z positions
        
        for omgeo in frame['I3Geometry'].omgeo.values():
            x_position = omgeo.position.x
            y_position = omgeo.position.y
            z_position = omgeo.position.z
            data_container['X_position_G'].append(x_position)
            data_container['Y_position_G'].append(y_position)
            data_container['Z_position_G'].append(z_position)

        
        # Make a loop to extract the string number and the OM number
        for omkey in frame['I3Geometry'].omgeo.keys():
            string_number = omkey.string
            om_number = omkey.om
            data_container['StringNumber'].append(string_number)
            data_container['OMNumber'].append(om_number)
            #print(string_number, om_number)
    else:
        data_container['X_position_G'].append(np.NaN)
        data_container['Y_position_G'].append(np.NaN)
        data_container['Z_position_G'].append(np.NaN)
        data_container['StringNumber'].append(np.NaN)
        data_container['OMNumber'].append(np.NaN)
        
            

def make_plot(data_container=None, outputname=None):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    pdf = PdfPages(outputname)

    #plot the weights
    fig,ax = plt.subplots(figsize=(6,6))
    ax.hist(data_container['GENIEWeight'], bins=10)
    ax.set_xlabel('GENIE Weight')
    ax.set_ylabel('Counts')
    ax.set_title('GENIE Weight Distribution')
    pdf.savefig(fig)
    
    #plot the charge
    fig,ax = plt.subplots(figsize=(6,6))
    temp_Charge = np.concatenate(data_container['Charge'],axis=0)
    ax.hist(temp_Charge, bins=50,range = (0, 4))
    ax.set_xlabel('Charge')
    ax.set_ylabel('Counts')
    ax.set_title('Charge Distribution')
    pdf.savefig(fig)
    #Create 3 plots showing the distribution of the primary neutrino positions
    fig,ax = plt.subplots(3,1, figsize=(6,18))
    ax[0].hist(data_container['X_position_Prim'], bins=50)
    ax[0].set_xlabel('X position')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('X position of primary neutrino')
    
    ax[1].hist(data_container['Y_position_Prim'], bins=50)
    ax[1].set_xlabel('Y position')
    ax[1].set_ylabel('Counts')
    ax[1].set_title('Y position of primary neutrino')
    
    ax[2].hist(data_container['Z_position_Prim'], bins=50)
    ax[2].set_xlabel('Z position')
    ax[2].set_ylabel('Counts')
    ax[2].set_title('Z position of primary neutrino')
    pdf.savefig(fig)
    
    #Create 3 plots showing the distribution of the secoundary positions
    fig,ax = plt.subplots(3,1,figsize=(6,18))
    ax[0].hist(data_container['X_position_secoundary'], bins=50)
    ax[0].set_xlabel('X position')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('X position of secoundary neutrino')
    
    ax[1].hist(data_container['Y_position_secoundary'], bins=50)
    ax[1].set_xlabel('Y position')
    ax[1].set_ylabel('Counts')
    ax[1].set_title('Y position of secoundary neutrino')
    
    ax[2].hist(data_container['Z_position_secoundary'], bins=50)
    ax[2].set_xlabel('Z position')
    ax[2].set_ylabel('Counts')
    ax[2].set_title('Z position of secoundary neutrino')
    pdf.savefig(fig)
    
    # Create two plots showing the energy of the primary neutrino and the secoundary muon
    fig,ax = plt.subplots(2,1,figsize=(6,6))
    ax[0].hist(data_container['Energy_Prim'], bins=50)
    ax[0].set_xlabel('Energy')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Energy of primary neutrino')
    ax[0].set_xlim(0, 150)
    
    ax[1].hist(data_container['Energy_secoundary'], bins=50)
    ax[1].set_xlabel('Energy')
    ax[1].set_ylabel('Counts')
    ax[1].set_title('Energy of secoundary muon')
    ax[1].set_xlim(0, 100)
    pdf.savefig(fig)
    
    
    #Make a plot showing the mean charge as a function of the z position
    fig,ax = plt.subplots(figsize=(6,6))
    temp_Charge_mean = data_container['mean_charge']
    ax.scatter(data_container['Z_position_secoundary'], temp_Charge_mean, s=1)
    ax.set_xlabel('Z position')
    ax.set_ylabel('Mean Charge / length of track')
    ax.set_title('Mean Charge vs Z position')
    #ax.set_ylim(0, 1.5)
    pdf.savefig(fig)
    
    # Initialize a dictionary to store sum of charges and count of hits for each OMKey
    omkey_stats = {}

    # Iterate through both lists of arrays simultaneously
    for omkeys, charges in zip(data_container['OMKey'], data_container['Charge']):
        for omkey, charge in zip(omkeys, charges):
            #omkey_tuple = tuple(omkey)  # Convert OMKey to a  tuple to make it hashable
            if omkey not in omkey_stats:
                omkey_stats[omkey] = {'sum_charge': 0, 'count_hits': 0}
            omkey_stats[omkey]['sum_charge'] += charge
            omkey_stats[omkey]['count_hits'] += 1

    # Calculate the mean charge for each OMKey
    mean_charge_per_omkey = {}
    for omkey in omkey_stats:
        sum_charge = omkey_stats[omkey]['sum_charge']
        count_hits = omkey_stats[omkey]['count_hits']
        #Print the average number of hits per OMKey
       # print('OMKey:', omkey, 'Average number of hits:', count_hits)
        mean_charge = sum_charge / count_hits
        mean_charge_per_omkey[omkey] = mean_charge
    
    # Sort the OMKeys and their corresponding mean charges in ascending order
    sorted_omkeys = sorted(mean_charge_per_omkey.items())
    # Convert OMKey values to a plottable format (e.g., strings)
    #omkey_strings = [str(omkey) for omkey in mean_charge_per_omkey.keys()]
    #mean_charges = list(mean_charge_per_omkey.values())
    omkey_strings = [str(omkey) for omkey, _ in sorted_omkeys]
    mean_charges = [mean_charge for _, mean_charge in sorted_omkeys]
    # Plot the mean charge per hit as a function of OMKey
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(omkey_strings, mean_charges, s=10)
    ax.set_xlabel('OMKey')
    ax.set_ylabel('Mean Charge per Hit')
    ax.set_title('Mean Charge per Hit vs OMKey')
    
    # Set x-ticks at regular intervals
    num_ticks = 40  # Number of ticks you want to display
    x_ticks = np.arange(0, len(omkey_strings), max(1, len(omkey_strings) // num_ticks))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([omkey_strings[i] for i in x_ticks], rotation=90, fontsize=8)

    # Adjust the x-axis limits to ensure the first tick is near x=0 and the last tick is at the end
    ax.set_xlim(-0.5, len(omkey_strings) - 0.5)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    pdf.savefig(fig)
    
    #plot the path travelled by the neutrino
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    zenith = data_container['Zenith_secoundary']
    azimuth = data_container['Azimuth_secoundary']

    x_end = data_container['X_position_secoundary'] + data_container['length_secoundary']*np.sin(zenith)*np.cos(azimuth)
    y_end = data_container['Y_position_secoundary'] + data_container['length_secoundary']*np.sin(zenith)*np.sin(azimuth)
    z_end = data_container['Z_position_secoundary'] + data_container['length_secoundary']*np.cos(zenith)
    # #print(data_container['X_position_secoundary'][:5])
    # range_val = 40
    # for i in range(range_val):
    #     ax.plot([data_container['X_position_secoundary'][i], x_end[i]], [data_container['Y_position_secoundary'][i], y_end[i]], [data_container['Z_position_secoundary'][i], z_end[i]], linestyle='--', color = "black")
        
    # #for i in range(0, len(data_container['X_position_G']), 100):
    # ax.scatter(data_container['X_position_G'], data_container['Y_position_G'], data_container['Z_position_G'], color='green', label = 'OMKey', alpha = 0.3)
    # #ax.plot([data_container['X_position_secoundary'][:20], x_end[:20]], [data_container['Y_position_secoundary'][:20], y_end[:20]], [data_container['Z_position_secoundary'][:20], z_end[:20]], linestyle='--', color = "black")
    # ax.scatter(data_container['X_position_secoundary'][:range_val], data_container['Y_position_secoundary'][:range_val], data_container['Z_position_secoundary'][:range_val], color = "red", label = 'Start',marker='o')
    # ax.scatter(x_end[:range_val], y_end[:range_val], z_end[:range_val], color = "blue", label = 'End',marker='o')
    # ax.set_xlim(-300,300)
    # ax.set_ylim(-300,300)
    # ax.set_zlim(-500,0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()

    # def rotate(angle):
    #     ax.view_init(elev = 30, azim=angle)

    # #Create and save the animation
    # ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)
    # from matplotlib.animation import FFMpegWriter
    
    # # # # Check if FFMpegWriter is available
    # try:
    #     writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    #     ani.save('neutrino_path_rotation.mp4', writer=writer)
    # except Exception as e:
    #     print(f"Failed to save animation as MP4: {e}. Try installing ffmpeg or use another format like GIF.")
    # plt.close(fig)    
    
    
    
    times = []
    charges = []
    total_charge_difference = []

    for time_array, charge_array,time_prim in zip(data_container['time'], data_container['Charge'],data_container['Time_Prim']):
        if time_prim < 0:
            continue  # Skip the arrays if time_prim is less than 0
        current_time = 0
        current_charge = 0
        count = 0
        sorted_indices = time_array.argsort()
        sorted_time_array = time_array[sorted_indices]
        sorted_charge_array = charge_array[sorted_indices]
        
        normalized_time_array = sorted_time_array - time_prim
        #print(time_prim)
        #Discard all the times that are less than 0
        mask = normalized_time_array > 0
        normalized_time_array = normalized_time_array[mask]
        sorted_charge_array = sorted_charge_array[mask]
       # print(sorted_charge_array)
        
         # Store the first and last charge values
        if len(sorted_charge_array) == 0:
            continue  # Skip if no valid times remain after masking
        # Store the first and last charge values
        first_charge = sorted_charge_array[0]
        last_charge = sorted_charge_array[-1]

        for time, charge in zip(normalized_time_array, sorted_charge_array):
            time = float(time)
            if time - current_time < 500:
                current_charge += charge
                count += 1

            else:
                if count > 0:
                    times.append(current_time)
                    charges.append(current_charge/count)  # Calculate mean charge
                current_time = time
                current_charge = charge
                count = 1
    
    # Append the last time and charge
        if count > 0:
            times.append(current_time)
            charges.append(current_charge / count)  # Calculate mean charge
    
        total_charge_difference.append(last_charge - first_charge)

    # Plot the charge as a function of time
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(times, charges, marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Charge')
    ax.set_title('Mean Charge as a Function of Time')
    pdf.savefig(fig)
    #Make a dictionary to store the energy and the zenith angle of the secoundary muon
    #The key is the energy and the value is the zenith angle
    #print(len(total_charge_difference))
    # Make a histogram showing the total charge difference between the first and last hit for each event
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(total_charge_difference, bins=50, range = (-5,5))
    ax.set_xlabel('Total Charge Difference')
    ax.set_ylabel('Counts')
    ax.set_title('Total Charge Difference between First and Last Hit')
    pdf.savefig(fig)    
    
    # Make a 3D plot showing the geometry of the detector
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, len(data_container['X_position_G']), 100):
        ax.scatter(data_container['X_position_G'][i], data_container['Y_position_G'][i], data_container['Z_position_G'][i], color='blue', label = 'OMKey')
    #ax.scatter(data_container['X_position_G'], data_container['Y_position_G'], data_container['Z_position_G'], color='blue', label='OMKey')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Geometry of the Detector')
    pdf.savefig(fig)
    
    pdf.close()
    print('Saved figures into', outputname)

# Make a function which matches the data_container['OMkey'] and data_container['string'] with data_container[StringNumber] and data_container[OMNumber]
# And from there we can get the x,y,z position of the DOMS which has been hit.
# We can then plot the path of the neutrino and the hits on the DOMS
# We can also plot the charge as a function of depth in the ice.

def match_and_plot(data_container=None,outputname=None):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    pdf2 = PdfPages(outputname)
    
    required_keys = ['OMKey', 'string', 'StringNumber', 'OMNumber', 'X_position_G', 'Y_position_G', 'Z_position_G', 'Charge']
    for key in required_keys:
        if key not in data_container:
            raise KeyError(f"Key '{key}' not found in data_container")
    # Create a dictionary to map (string_number, om_number) pairs to their indices
    index_map = {}
    for j, (string_number, om_number) in enumerate(zip(data_container['StringNumber'], data_container['OMNumber'])):
        key = (string_number, om_number)
        if key in index_map:
            print(f"Key {key} already exists in index_map¨,skipping")
            continue
        index_map[key] = j
    print(f"Index map size: {len(index_map)}")

    # Match the OMKey and string with the StringNumber and OMNumber
    matched_indices = []
    for i, (strings,omkeys) in enumerate(zip(data_container['string'],data_container['OMKey'])):
        if isinstance(omkeys, np.ndarray):
            for omkey in omkeys:
                key = (strings, omkey)
                #print(key)
        #print(key)
                if key in index_map:
                    matched_indices.append(index_map[key])
                #else:
                    #print(f"Could not find key {key}")
        else:
            key = (strings, omkeys)
            if key in index_map:
                matched_indices.append(index_map[key])
    print(f"Matched {len(matched_indices)} out of {len(data_container['OMKey'])} hits")
    
    # Extract the x, y, and z positions of the DOMs that were hit
    x_positions = [data_container['X_position_G'][j] for j in matched_indices]
    y_positions = [data_container['Y_position_G'][j] for j in matched_indices]
    z_positions = [data_container['Z_position_G'][j] for j in matched_indices]
    
    #charges = [data_container['Charge'][j] for j in matched_indices]
    #charges = [charge if np.isscalar(charge) else charge[0] for charge in charges]
    
    # Plot the path of the neutrino and the hits on the DOMs
    pdf2 = PdfPages(outputname)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_positions, y_positions, z_positions, c=charges, cmap='viridis', label='DOM Hits')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Neutrino Path and DOM Hits')
    plt.legend()
    pdf2.savefig(fig)
    
    # Plot the charge as a function of depth in the ice
    plt.figure()
    plt.scatter(z_positions, charges, c='blue', label='Charge vs Depth')
    plt.xlabel('Depth (Z Position)')
    plt.ylabel('Charge')
    plt.title('Charge as a Function of Depth in the Ice')
    plt.legend()
    pdf2.savefig(fig)
    
    pdf2.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process I3 files')
    
    parser.add_argument('-i', '--input', help='Input file', type=str, default='/groups/icecube/simon/GNN/workspace/data/I3_files/upgrade_genie_step4_140029_000000.i3')
    parser.add_argument('-g', '--gcd', help='Name of an input gcd file',
						default='/groups/icecube/simon/GNN/workspace/data/GCD_files/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2')
    parser.add_argument('-o','--output', help='name of the output pdf containing the plots.', default='TEST.pdf')
    
    #parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    from icecube import dataio, dataclasses, icetray
    from icecube.icetray import I3Tray
    
    #create the data_container
    container = OrderedDict()
    container['GENIEWeight'] = []
    
    container['time'] = []
    container['Charge'] = []
    container['mean_charge'] = []
    container['OMKey'] = []
    container['string'] = []
    
    container['X_position_Prim'] = []
    container['Y_position_Prim'] = []
    container['Z_position_Prim'] = []
    container['Azimuth_Prim'] = []
    container['Zenith_Prim'] = []
    container['Energy_Prim'] = []
    container['Time_Prim'] = []
    
    container['X_position_secoundary'] = []
    container['Y_position_secoundary'] = []
    container['Z_position_secoundary'] = []
    container['Azimuth_secoundary'] = []
    container['Zenith_secoundary'] = []
    container['length_secoundary'] = []
    container['Energy_secoundary'] = []
    
    container['X_position_G'] = []
    container['Y_position_G'] = []
    container['Z_position_G'] = []
    container['StringNumber'] = []
    container['OMNumber'] = []
    

    tray = I3Tray()
    def county(frame):
        if not hasattr(county, 'n_physics'):
            county.n_physics =0
        if frame.stop == icetray.I3Frame.Physics:
            county.n_physics += 1
            print('Physics frame:', county.n_physics)
    
        
  #  tray.Add(county,'my_counting_module')

    tray.AddModule('I3Reader', 'read_stuff', FilenameList=[args.gcd]+[args.input])
    
    #tray.AddModule(compute_radius, 'compute_radius')
    
    tray.AddModule(harvest_quantities, data_container=container, Streams = [icetray.I3Frame.Physics])
    tray.AddModule(harvest_geometry, data_container=container, Streams = [icetray.I3Frame.Geometry])
    #print(tray)
    #if args.debug:
    #    icetray.logging.set_level('TRACE')
    #    tray.Execute(100)
    #else:
    tray.Execute() 
    
    #print(container['GENIEWeight'])
    #Make the plots
    make_plot(outputname=args.output,data_container=container)
    
    match_and_plot(data_container=container, outputname='Test2.pdf')
  