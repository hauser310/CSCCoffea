"""Template executor that runs the processor (in a distributed way, or otherwise)."""
import glob
import coffea.hist as hist
import coffea.processor as processor
import matplotlib.pyplot as plt
from coffea.nanoevents import BaseSchema
from matplotlib.colors import LogNorm
from bremsstrahlung_processor import BremsstrahlungProcessor

OUTPUT_DIR = "../output/"

"""
Select the files to run over, here use a generated muon gun with gen information
"""
files = glob.glob("/eos/cms/store/user/wnash/CSCDigiTree-PDFSet12.root")

fileset = {"dummy": files}

out = processor.run_uproot_job(
    fileset=fileset,
    treename="CSCDigiTree",
    processor_instance=BremsstrahlungProcessor(),
    executor=processor.futures_executor,
    # executor=processor.iterative_executor,
    executor_args={"schema": BaseSchema, "workers": 8},
)


fig, ax = plt.subplots()

detectors = ["ecal", "hcal"]

for detector in detectors:
    fig.clear(True)
    ax.clear()
    ax = hist.plot1d(out["muons"].project("p", detector), overlay="p", overflow="all")
    ax.set_xscale("log")
    plt.savefig(OUTPUT_DIR + f"muon_{detector}_slices.png")

    fig.clear(True)
    ax.clear()
    ax = hist.plot2d(
        out["muons"].project("p", detector), xaxis="p", patch_opts={"norm": LogNorm()}
    )
    ax.set_yscale("log")
    plt.savefig(OUTPUT_DIR + f"muon_{detector}_vs_p.png")


fig.clear()
ax = hist.plot2d(
    out["muons"].project("hcal", "ecal"), xaxis="hcal", patch_opts={"norm": LogNorm()}
)
ax.set_yscale("log")
ax.set_xscale("log")
plt.savefig(OUTPUT_DIR + "muon_ecal_vs_hcal.png")
