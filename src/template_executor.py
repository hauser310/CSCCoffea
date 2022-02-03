"""Template executor that runs the processor (in a distributed way, or otherwise)."""
import glob
import coffea.hist as hist
import coffea.processor as processor
import matplotlib.pyplot as plt
from coffea.nanoevents import BaseSchema
from template_processor import TemplateProcessor

# increase resolution of output .png files
plt.figure(dpi=400)
"""Select the files to run over"""
files = glob.glob("/eos/cms/store/user/wnash/CSCDigiTree_*.root")

fileset = {"dummy": files}

out = processor.run_uproot_job(
    fileset=fileset,
    treename="CSCDigiTree",
    processor_instance=TemplateProcessor(),
    executor=processor.iterative_executor,
    executor_args={"schema": BaseSchema, "workers": 8},
)


fig, ax = plt.subplots()
ax = hist.plot1d(out["segment_slice_dxdz"], overlay="pt_slice", density=True)
plt.savefig("segment/segment_slice_dxdz_vs_pt.png")

fig.clear()
ax = hist.plot1d(out["segment"].project("chisq"), density=True)
ax = hist.plot1d(out["segment_muon"].project("chisq"), density=True)
plt.ylim(0, 0.15)
ax.legend(["entire dataset", "muon-associated section"])
plt.savefig("segment/segment_muon_chisq.png")

fig.clear()
ax = hist.plot1d(out["segment"].project("nHits"), density=True)
ax = hist.plot1d(out["segment_muon"].project("nHits"), density=True)
plt.ylim(0, 1)
ax.legend(["entire dataset", "muon-associated section"])
plt.savefig("segment/segment_muon_nHits.png")

ax = hist.plot1d(out["muons"].project("pt"))
plt.savefig("muon_pt.png")

fig.clear()
ax = hist.plot2d(out["muons"].project("pt", "eta"), xaxis="pt")
plt.savefig("muon_eta_vs_pt.png")
