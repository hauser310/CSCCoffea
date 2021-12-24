"""Template executor that runs the processor (in a distributed way, or otherwise)."""
import glob
import coffea.hist as hist
import coffea.processor as processor
import matplotlib.pyplot as plt
import pandas as pd
from coffea.nanoevents import BaseSchema
from matplotlib.colors import LogNorm
from bremsstrahlung_processor import BremsstrahlungProcessor

OUTPUT_DIR = "../output/"

# increase resolution of output .png files
plt.figure(dpi=400)

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

for var in ["p", "eta", "phi"]:
    fig.clear(True)
    ax.clear()
    p_dp = out["all_muons"].project(var)
    ax = hist.plot1d(out["all_muons"].project(var))
    ax.get_legend().remove()
    plt.savefig(OUTPUT_DIR + f"all_muons_{var}.pdf")


fig.clear(True)
ax.clear()
p_dp = out["p_loss"].project("p", "dp")
p_dp = p_dp.rebin("p", hist.Bin("p_rebinned", "$p$ [GeV]", 5, 0, 4000))
ax = hist.plot1d(p_dp, overlay="p_rebinned", overflow="all", density="true")
plt.savefig(OUTPUT_DIR + "muon_dp_vs_p_slices.pdf")

ax = hist.plot2d(
    out["p_loss"].project("p", "dp"),
    xaxis="p",
    xoverflow="all",
    yoverflow="all",
    patch_opts={"norm": LogNorm()},
)
plt.savefig(OUTPUT_DIR + "muon_dp_vs_p.pdf")


detectors = ["ecal", "hcal"]

for detector in detectors:
    fig.clear(True)
    ax.clear()
    ax = hist.plot1d(
        out["muon_deposits"].project("p", detector), overlay="p", overflow="all"
    )
    ax.set_xscale("log")
    plt.savefig(OUTPUT_DIR + f"muon_{detector}_slices.png")

    fig.clear(True)
    ax.clear()
    ax = hist.plot2d(
        out["muon_deposits"].project("p", detector),
        xaxis="p",
        patch_opts={"norm": LogNorm()},
    )
    ax.set_yscale("log")
    plt.savefig(OUTPUT_DIR + f"muon_{detector}_vs_p.png")

fig.clear(True)
ax = hist.plot2d(
    out["muon_deposits"].project("hcal", "ecal"),
    xaxis="hcal",
    patch_opts={"norm": LogNorm()},
)
ax.set_yscale("log")
ax.set_xscale("log")
plt.savefig(OUTPUT_DIR + "muon_ecal_vs_hcal.png")


fig.clear(True)
ax = hist.plot2d(
    out["p_loss"].project("p", "p_exit"),
    xaxis="p",
    xoverflow="all",
    yoverflow="all",
    patch_opts={"norm": LogNorm()},
)
plt.savefig(OUTPUT_DIR + "muon_pexit_vs_p.png")

# columns = ["p", "dp", "eta", "phi", "hcal", "ecal", "csc"]
columns = ["dp", "eta", "phi", "hcal", "ecal", "csc"]
data = {}
for col in columns:
    data[col] = out[col].value

df = pd.DataFrame(data)
df.to_pickle(OUTPUT_DIR + "brem_dataset.pkl")
