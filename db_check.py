from icecube import icetray, dataio
import glob, os
from icecube.linefit import simple as robust_linefit_simple
# Build combined file list: GCD first, then physics files.
gcd_file = "/groups/icecube/simon/GNN/workspace/data/GCD_files/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
input_folder = "/groups/icecube/simon/GNN/workspace/data/I3_files/132028/"
physics_files = glob.glob(os.path.join(input_folder, "*.i3.zst*"))
combined_list = [gcd_file] + physics_files

tray = icetray.I3Tray()
tray.AddModule("I3Reader", "reader", FilenameList=combined_list)
tray.AddModule(lambda frame: print("Frame keys:", list(frame.keys())), "printer", Streams=[icetray.I3Frame.Geometry])
def pulli3omgeo(frame):
    global i3omgeo
    i3omgeo = frame["I3Geometry"].omgeo
tray.Add(pulli3omgeo,"soitonlyhappensonce",Streams=[icetray.I3Frame.Geometry])

tray.Execute()
