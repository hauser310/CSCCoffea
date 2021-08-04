"""Template executor that runs the processor (in a distributed way, or otherwise)."""
import glob
import coffea.hist as hist
import coffea.processor as processor
import matplotlib.pyplot as plt
from coffea.nanoevents import BaseSchema
from comparator_code_processor import ComparatorCodeProcessor

# increase resolution of output .png files
plt.figure(dpi=400)

# in the future, one could use frameworks such as dask for
# better parallelization
# from dask.distributed import Client
# https://github.com/CoffeaTeam/coffea/blob/master/binder/processor.ipynb

"""
Select the files to run over
"""
# files = glob.glob("/eos/cms/store/user/wnash/CSCDigiTree-PDF*.root")
files = glob.glob("/eos/cms/store/user/wnash/CSCDigiTree_*.root")

fileset = {"dummy": files}

out = processor.run_uproot_job(
    fileset=fileset,
    treename="CSCDigiTree",
    processor_instance=ComparatorCodeProcessor(),
    executor=processor.futures_executor,
    # executor=processor.iterative_executor,
    executor_args={"schema": BaseSchema, "workers": 8},
)

"""Here is where we receive output from template_processor.py to generate plots.
fig, ax = plt.subplots() #use for first plot, otherwise delete
fig.clear() #use only if not the first plot, otherwise delete
ax = hist.plot1d(out["variable"].project("leaf"))
plt.savefig("variable/variable_leaf.png")"""

fig, ax = plt.subplots()
ax = hist.plot1d(out["chamber"].project("ch_id"))
plt.savefig("chamber/chamber_ch_id.png")


fig.clear()
ax = hist.plot1d(out["chamber"].project("diff"))
plt.savefig("chamber/chamber_diff.png")
