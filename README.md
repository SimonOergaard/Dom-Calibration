This is the scribts developed for my master's thesis titled: "Towards Stable Calibration: DOM Response and Depth Effects in IceCube".
Look in the ReadMe folder for a more detailed describtion of various scribts.


Abstract:
This thesis investigates the relative quantum efficiency (QE) of Digital Optical Modules (DOMs) in the IceCube Neutrino Observatory using atmospheric stopping muons. These muons are abundant, approximately minimum ionizing, and emit stable Cherenkov light, making them ideal for in-situ calibration.

This thesis is divided into two parts: A theoretical and experimental overview of neutrino physics and the IceCube detector, followed by the DOM efficiency analysis. The key metric used is the Relative Individual DOM Efficiency (RIDE), which compares observed DOM charge to an expected value under matched geometric conditions, e.g. a cylinder surrounding a track with the radii defined as the shortest euclidean distance between DOM and track.

The majority of the analyses are based on Monte Carlo simulations with truth labels for muon stopping positions and directions. To extend the method to real data, a machine learning model using GraphNeT’s DynEdge architecture is trained to reconstruct muon stopping points and direction vectors.

Two main analyses strategies are employed. A full-depth comparison shows that RIDE clearly distinguishes between strings with equal and differing QEs. Median RIDE values for high-QE (HQE) strings are compared to those from string 80, with separate studies in the veto cap and the dust layer. In the veto cap, only cross-QE comparisons are possible; RIDE values deviate from the expected 1.35 ratio but stabilize beyond ~80–90 m. In DeepCore, both same- and mixed-QE comparisons are possible: same-QE strings yield RIDE values near 1.0, while mixed-QE strings consistently exceed the expected 1.35 ratio.

A final cross-check compares real and simulated data via a double ratio method, confirming good agreement, with some discrepancies in specific DOM groups. This work provides a robust method for DOM calibration using stopping muons, with potential to reduce systematic uncertainties in IceCube event reconstructions.

