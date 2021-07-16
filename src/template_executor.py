"""Template executor that runs the processor (in a distributed way, or otherwise)."""
import glob

import coffea.hist as hist
import coffea.processor as processor
import matplotlib.pyplot as plt
from coffea.nanoevents import BaseSchema
from template_processor import TemplateProcessor

# increase resolution of output .png files
plt.figure(dpi=400)

# in the future, one could use frameworks such as dask for
# better parallelization
# from dask.distributed import Client
# https://github.com/CoffeaTeam/coffea/blob/master/binder/processor.ipynb

"""
Select the files to run over
"""
files = glob.glob("/eos/home-w/wnash/public/CSCUCLA/CSCDigiTree*.root")

fileset = {"dummy": files}

out = processor.run_uproot_job(
    fileset=fileset,
    treename="CSCDigiTree",
    processor_instance=TemplateProcessor(),
    executor=processor.futures_executor,
    executor_args={"schema": BaseSchema, "workers": 8},
)

fig, ax = plt.subplots()
ax = hist.plot1d(out["muons"].project("pt"))
plt.savefig("muon_pt.png")
