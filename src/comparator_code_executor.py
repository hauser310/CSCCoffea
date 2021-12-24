"""Executor that runs the processor analyzing pre-compiled comparator code (in a distributed way, or otherwise)."""
import coffea.hist as hist
import coffea.processor as processor
import matplotlib.pyplot as plt
from coffea.nanoevents import BaseSchema
from comparator_code_processor import ComparatorCodeProcessor
import pandas as pd

"""increase resolution of output .png files"""
plt.figure(dpi=400)

"""Select the files to run over"""

df = pd.read_csv(
    "/afs/cern.ch/user/w/wnash/CSCCoffea/data/comparator_codes.csv",
    header=None,
    names=["key_pattern", "key_code", "foundSegment", "entry_layers", "entry_chi2"],
)

fileset = {
    "dummy": [
        df,
    ]
}

out = processor.run_uproot_job(
    fileset=fileset,
    treename="CSCDigiTree",
    processor_instance=ComparatorCodeProcessor(),
    executor=processor.iterative_executor,
    executor_args={"schema": BaseSchema, "workers": 8},
)

"""Here is where we receive output from template_processor.py to generate plots.
fig, ax = plt.subplots() #use for first plot, otherwise delete
fig.clear() #use only if not the first plot, otherwise delete
ax = hist.plot1d(out["variable"].project("leaf"))
plt.savefig("variable/variable_leaf.png")"""

fig, ax = plt.subplots()
ax = hist.plot1d(out["LUT"].project("position", "pcc"), overlay="pcc", density=True)
plt.savefig("LUT/LUT_position.png")

fig.clear()
ax = hist.plot1d(out["LUT"].project("slope", "pcc"), overlay="pcc", density=True)
plt.savefig("LUT/LUT_slope.png")

fig.clear()
ax = hist.plot1d(out["LUT"].project("pt", "pcc"), overlay="pcc", density=True)
plt.savefig("LUT/LUT_pt.png")

fig.clear()
ax = hist.plot1d(out["LUT"].project("multiplicity", "pcc"), overlay="pcc", density=True)
plt.savefig("LUT/LUT_multiplicity.png")
