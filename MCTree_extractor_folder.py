from icecube import dataclasses, dataio, icetray
import csv
import os
import glob

def extract_mctree_info(mctree, file_id, event_id):
    """
    Extracts relevant information from the I3MCTree.
    """
    data = []
    for particle in mctree:
        info = {
            "FileID": file_id,  # Include File_id
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

def process_i3_file(input_file):
    """
    Processes a single I3 file and extracts MCTree data.
    """
    file_name = os.path.basename(input_file)
    file_id = file_name.split("_")[-1].split(".")[0]
    i3_file = dataio.I3File(input_file, 'r')
    file_data = []
    
    while i3_file.more():
        frame = i3_file.pop_frame()
        if frame.Stop == icetray.I3Frame.DAQ:  # Process only Q frames
            if "I3MCTree" in frame and "I3EventHeader" in frame:
                event_header = frame["I3EventHeader"]
                mctree = frame["I3MCTree"]
                event_id = event_header.event_id  # Extract EventID
                frame_data = extract_mctree_info(mctree, file_id, event_id)
                file_data.extend(frame_data)
    
    return file_data

def main(input_folder, output_csv):
    """
    Processes all I3 files in a folder and writes combined particle data to a CSV file.
    """
    # Find all .i3 files in the folder
    input_files = glob.glob(os.path.join(input_folder, "*.i3.zst"))
    
    if not input_files:
        print(f"No .i3 files found in folder '{input_folder}'. Exiting.")
        return
    
    all_data = []
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File '{input_file}' does not exist. Skipping.")
            continue
        
        print(f"Processing file: {input_file}")
        file_data = process_i3_file(input_file)
        all_data.extend(file_data)
    
    if not all_data:
        print("No data extracted. Exiting.")
        return

    # Write the combined data to the output CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["FileID","EventID","ParticleID", "Type", "Energy", "PosX", "PosY", "PosZ", "DirZenith", "DirAzimuth", "Time", "Length"] 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Combined data successfully written to {output_csv}")

if __name__ == "__main__":
    # Specify the folder containing the I3 files
    input_folder = "/groups/icecube/simon/GNN/workspace/data/I3_files/132028_part2/132028"  # Replace with the path to your folder
    output_csv = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/combined_output_particles.csv"  # Replace with your desired output file path
    main(input_folder, output_csv)
