"""Fit the dp vs p distribution to a Landau distribution which is dependent on p."""
import glob
import matplotlib.pyplot as plt
import coffea.processor as processor
from coffea.nanoevents import BaseSchema
from bremsstrahlung_processor import BremsstrahlungProcessor
from helpers import landau, OUTPUT_DIR
from scipy.optimize import curve_fit
import numpy as np
import logging
import pickle

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


# get arrays of p, dp, and the effective probabilities
# (number of events in each bin)
for i, (_, values) in enumerate(out["p_loss"].project("p", "dp").values().items()):
    if i > 0:
        raise RuntimeError("Unexpected > 1 number of sparse axes in dp vs p graph")
    else:
        dp_vs_p_data = values

dp_axis = out["p_loss"].project("p", "dp").axis("dp").centers()
p_axis = out["p_loss"].project("p", "dp").axis("p").centers()


# now make a single matrix which has all the relevant parameters:
dp_data = []
p_data = []
ydata = []
for i, dp in enumerate(dp_axis):
    for j, p in enumerate(p_axis):
        p_data.append(p)
        dp_data.append(dp)
        ydata.append(dp_vs_p_data[j][i])
xdata = (np.array(dp_data), np.array(p_data))

# we know that dp should be distributed
# according to the Landau distribution
# for a fixed p, and we take that the
# mean and scale parameters are linearly
# dependent on the momentum and fit the distribution
# accordingly.
popt, pcov = curve_fit(landau, xdata, np.array(ydata))

chi_2 = 0
ndf = len(ydata) - 4
for y, dp, p in zip(ydata, dp_data, p_data):
    e = landau((dp, p), *popt)
    if e > 1e-2:  # stop div by zero issues
        chi_2 += (y - e) / e
    logging.info(y, e)
logging.info(f"Fit distribution: {popt}")
logging.info(f"\t chi_2: {chi_2}, chi_2/ndf: {chi_2/ndf}")


fig, ax = plt.subplots()

momenta = (400, 1200, 2000, 2800, 3600)
for p in momenta:
    fit_y = []
    for dp in dp_axis:
        fit_y.append(landau((dp, p), *popt))
    ax.plot(dp_axis, np.array(fit_y))
ax.legend([f"$p$ = {p} GeV" for p in momenta])
plt.xlabel("$\\Delta p$ [GeV]")
plt.ylabel("Density (A.U.)")
for suffix in (".png", ".pdf"):
    plt.savefig(OUTPUT_DIR + "landau_fit_p" + suffix)

fig.clear(True)
fig, ax = plt.subplots()

dps = (5, 10, 20, 50, 100)
for dp in dps:
    fit_y = []
    for p in p_axis:
        fit_y.append(landau((dp, p), *popt))
    ax.plot(p_axis, np.array(fit_y))

ax.legend([f"$\\Delta p$ = {dp} GeV" for dp in dps])
plt.xlabel("$p$ [GeV]")
plt.ylabel("Density (A.U.)")
for suffix in (".png", ".pdf"):
    plt.savefig(OUTPUT_DIR + "landau_fit_dp" + suffix)


with open(OUTPUT_DIR + "landau_fit_parameters.pkl", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(popt, f, pickle.HIGHEST_PROTOCOL)
