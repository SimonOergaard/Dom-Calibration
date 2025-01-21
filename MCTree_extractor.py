from icecube import dataclasses, dataio, icetray
import csv
import os

def extract_mctree_info(mctree, event_id):
    """
    Extracts relevant information from the I3MCTree.
    """
    data = []
    for particle in mctree:
        info = {
            "EventID": event_id,  # Include EventID
            "ParticleID": particle.id,
            "Type": str(particle.type),
            "Energy": particle.energy,  # Energy in GeV
            "PosX": particle.pos.x,  # Position x in meters
            "PosY": particle.pos.y,  # Position y in meters
            "PosZ": particle.pos.z,  # Position z in meters
            "DirZenith": particle.dir.zenith,  # Zenith angle in radians
            "DirAzimuth": particle.dir.azimuth,  # Azimuth angle in radians
            "Time": particle.time,  # Time in nanoseconds
            "Length": particle.length  # Length in meters
        }
        data.append(info)
    return data

def main(input_file, output_csv):
    """
    Reads an I3 file containing Q frames with I3MCTree and writes particle data to a CSV file.
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    # Create the directory for the output CSV if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the input I3 file
    i3_file = dataio.I3File(input_file, 'r')
    
    # Prepare a list to store extracted data
    all_data = []
    
    # Loop through frames
    while i3_file.more():
        frame = i3_file.pop_frame()
        if frame.Stop == icetray.I3Frame.DAQ:  # Process only Q frames
            if "I3MCTree" in frame and "I3EventHeader" in frame:  # Check for I3MCTree and I3EventHeader
                mctree = frame["I3MCTree"]
                event_header = frame["I3EventHeader"]
                event_id = event_header.event_id  # Extract EventID
                frame_data = extract_mctree_info(mctree, event_id)
                all_data.extend(frame_data)
    
    # Write the extracted data to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["EventID", "ParticleID", "Type", "Energy", "PosX", "PosY", "PosZ", "DirZenith", "DirAzimuth", "Time", "Length"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Data successfully written to {output_csv}")

if __name__ == "__main__":
    input_file = "/groups/icecube/simon/GNN/workspace/data/I3_files/132028/upgrade_muongun_step1_132028_000000.i3.zst"  # Replace with your input I3 file path
    output_csv = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/Step_1_MuonGun.csv"  # Replace with your desired output CSV file path
    main(input_file, output_csv)
