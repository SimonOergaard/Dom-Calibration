#!/usr/bin/env python3
"""Process one GCD file and one physics file to add robust LineFit,
   segmentation (n=3), StartingTrackVeto, and I3StartStopPoint results.
   The output is written to a new I3 file.
"""

import sys, os, numpy as np
from icecube import icetray, dataio, dataclasses, photonics_service, StartingTrackVeto, recclasses, lilliput, finiteReco, phys_services
from icecube.linefit import simple as robust_linefit_simple
from icecube.icetray import I3Units
from icecube.finiteReco.segments import simpleLengthReco, advancedLengthReco
import glob

def get_pxs():    
    inf_muon_service = photonics_service.I3PhotoSplineService(
                    amplitudetable = os.path.join( os.path.expandvars("$/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/") ,"InfBareMu_mie_abs_z20a10_V2.fits"),  ## Amplitude tables
                    timingtable = os.path.join( os.path.expandvars("$/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/") ,"InfBareMu_mie_prob_z20a10_V2.fits"),    ## Timing tables
                    timingSigma  = 0.0,
                    maxRadius    = 600.0)
    return inf_muon_service
inf_muon_service = get_pxs()

# --- Utility Module: Segment Generator ---
def make_n_segment_vector(frame, fit, n=1):
    """
    Splits the track in 'fit' into n segments.
    For n==1, produces one infinite track segment.
    The segments are saved to the frame under key: <fit>_<n>_segments.
    """
    if n % 2 == 0:
        print("n =", n, "is even! Change this!")
        sys.exit(910)
    try:
        basep = frame[fit]
    except Exception as e:
        print("Error: Cannot find fit", fit, "in frame:", e)
        return False
    origin_cap = phys_services.I3Calculator.closest_approach_position(basep, dataclasses.I3Position(0,0,0))
    basep_shift_d = np.sign(origin_cap.z - basep.pos.z) * np.sign(basep.dir.z) * (origin_cap - basep.pos).magnitude
    basep_shift_pos = basep.pos + basep.dir * basep_shift_d
    basep_shift_t = basep_shift_d / basep.speed
    basep.pos = basep_shift_pos
    basep.time = basep.time + basep_shift_t

    segments = []
    segment_length = 1950. / n
    for idx in range(n):
        dshift = segment_length * (idx - ((n - 1) / 2.))
        particle = dataclasses.I3Particle()
        particle.time = basep.time + (dshift / basep.speed)
        particle.pos = basep.pos + basep.dir * dshift
        particle.dir = basep.dir
        particle.energy = 0.01
        if n == 1:
            particle.shape = particle.shape.InfiniteTrack
            particle.length = 0
        else:
            particle.shape = particle.shape.ContainedTrack
            particle.length = segment_length
        segments.append(particle)
    seg_key = f"{fit}_{n}_segments"
    if frame.Has(seg_key):
        del frame[seg_key]
    frame.Put(seg_key, dataclasses.I3VectorI3Particle(segments))
    #print("Injected segments with key:", seg_key)
    # Optionally, if the original fit is no longer needed, delete it.
    #if frame.Has(fit):
    #    del frame[fit]
    return True

# --- Main Processing Function ---
def process_file(gcd_file: str, physics_file: str, output_file: str,
                 name="robust",
                 inputResponse="SplitInIcePulsesSRT",
                 linefitKey="linefit_improved",
                 If=lambda f: True):
    """
    Reads the GCD file and a physics file (in that order),
    runs robust_linefit_simple, then generates segments,
    runs StartingTrackVeto and I3StartStopPoint,
    and writes out a new I3 file with the added information.
    """
    tray = icetray.I3Tray()
    
    # Build a combined file list with the GCD file first.
    combined_list = [gcd_file, physics_file]
    tray.AddModule("I3Reader", "reader", FileNameList=combined_list)
    
    # Optionally pull out the geometry for debugging.
    def pulli3omgeo(frame):
        global i3omgeo
        i3omgeo = frame["I3Geometry"].omgeo
    tray.AddModule(pulli3omgeo, "pull_geometry", Streams=[icetray.I3Frame.Geometry])
    
    # Run the robust linefit reconstruction.
    robust_linefit_simple(tray, name,
                          inputResponse=inputResponse,
                          fitName=linefitKey,
                          If=If)
    
    # # Generate segments from the linefit result.
    tray.AddModule(make_n_segment_vector, "make_segments", fit=linefitKey, n=1)
    
   # Run StartingTrackVeto to estimate containment and veto info.
    tray.AddModule("StartingTrackVeto", "stv",
                   Pulses=inputResponse,
                   Photonics_Service=inf_muon_service,
                   Miss_Prob_Thresh=1,
                   Fit=linefitKey,
                   Particle_Segments=f"{linefitKey}_1_segments",
                   Distance_Along_Track_Type="cherdat",
                   Supress_Stochastics=True,
                   Min_CAD_Dist=300,
                   If=If)
    # print("Running I3StartStopPoint module.")
    # def print_stv(frame):
    #     key = f"{linefitKey}_3_segments"  # or another key if StartingTrackVeto writes elsewhere
    #     print("STV module processed frame keys:", list(frame.keys()), flush=True)
    #     if frame.Has(key):
    #         stv_out = frame[key]
    #         print("STV output (segments) present.", flush=True)
    #     return
    # tray.AddModule(print_stv, "print_stv", Streams=[icetray.I3Frame.Physics])

    # # Run I3StartStopPoint to compute the stopping point.
    # tray.AddModule("I3StartStopPoint", "startstop",
    #                Name=linefitKey,
    #                InputRecoPulses=inputResponse,
    #                ExpectedShape=dataclasses.I3Particle.StoppingTrack,
    #                CylinderRadius=200*I3Units.m,
    #                If=If)
    

    
    # Write out the processed frames.
    tray.AddModule("I3Writer", "writer",
         Filename=output_file,
         Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
         DropOrphanStreams=[icetray.I3Frame.Geometry,
                              icetray.I3Frame.Calibration,
                              icetray.I3Frame.DetectorStatus,
                              icetray.I3Frame.DAQ])
    
    tray.Execute()
    print("Output written to", output_file)

# --- Example Usage ---
if __name__ == "__main__":
    # Define the file paths.
    gcd_file = "/groups/icecube/simon/GNN/workspace/data/GCD_files/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
    physics_file = "/groups/icecube/simon/GNN/workspace/data/I3_files/132028/upgrade_muongun_step4_132028_000000.i3.zst"
    output_file = "/groups/icecube/simon/GNN/workspace/data/Converted_I3_file/linefit_output.i3.zst"
    
    process_file(gcd_file, physics_file, output_file)
