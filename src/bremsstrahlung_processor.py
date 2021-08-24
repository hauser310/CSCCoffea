"""Derive muon momentum via calorimetry."""
import awkward as ak
import numpy as np
import numba
from coffea import hist, processor

# register our candidate behaviors
from coffea.nanoevents.methods import candidate

ak.behavior.update(candidate.behavior)


"""
todo list:
[] get pAtEntry from simhits at ME41, to get total energy loss

"""


PI = 3.14159
DR_CUT = 0.2


@numba.njit
def delta_r(obj1, obj2):
    """Calculate dR = sqrt(d_eta^2 + d_phi^2)."""
    d_eta = obj1.eta - obj2.eta
    d_phi = np.abs(obj1.phi - obj2.phi)
    if d_phi > PI:
        d_phi -= 2 * PI
    return np.sqrt(d_eta * d_eta + d_phi * d_phi)


@numba.njit
def get_associated_energy(associated_array, gen_muons, detector_deposits):
    """Find the energy deposits associated with each gen muon."""
    for event_muons, event_deposits in zip(gen_muons, detector_deposits):
        associated_array.begin_list()
        for muon in event_muons:
            associated_deposits = 0.0
            for deposit in event_deposits:
                if delta_r(muon, deposit) < DR_CUT:
                    associated_deposits += deposit.energy
            associated_array.append(associated_deposits)
        associated_array.end_list()
    return associated_array


class BremsstrahlungProcessor(processor.ProcessorABC):
    """Runs the analysis."""

    def __init__(self):
        """Initialize."""
        self._accumulator = processor.dict_accumulator(
            {
                "allevents": processor.defaultdict_accumulator(float),
                "muons": hist.Hist(
                    "Muons",
                    hist.Bin(
                        "p", "$p$ [GeV]", np.array([0, 800, 1600, 2400, 3200, 4000])
                    ),
                    # hist.Bin("eta", "$\\eta$", 50, 0, 4),
                    hist.Bin(
                        "hcal", "HCAL energy loss [GeV]", np.logspace(-2.5, 0.5, num=51)
                    ),
                    hist.Bin(
                        "ecal", "ECAL energy loss [GeV]", np.logspace(-1.0, 2.0, num=51)
                    ),
                ),
            }
        )

    @property
    def accumulator(self):
        """Return pieces added together for each parallel processor."""
        return self._accumulator

    def process(self, events):
        """Operation done for each event."""
        output = self.accumulator.identity()

        dataset = events.metadata["dataset"]

        output["allevents"][dataset] += len(events)

        gen_muons = ak.zip(
            {
                "pt": events.gen_pt,
                "eta": events.gen_eta,
                # "theta": 2. * np.arctan(np.exp(-events.gen_eta)),
                "phi": events.gen_phi,
                # "mass": ak.Array([0.105658375 for _ in range(len(events.muon_pt))]),
                # "charge": events.gen_q,
            }
        )

        gen_muons.theta = 2.0 * np.arctan(np.exp(-gen_muons.eta))
        gen_muons.p = gen_muons.pt / np.sin(gen_muons.theta)

        gen_muons = ak.zip(
            {
                "pt": gen_muons.pt,
                "eta": gen_muons.eta,
                "phi": gen_muons.phi,
                "p": gen_muons.p,
            }
        )

        had_calorimeters = [
            "hcal",
        ]

        em_calorimeters = ["ecalPreshower", "ecalBarrel", "ecalEndcap"]

        calorimeter_names = had_calorimeters + em_calorimeters

        calorimeters = {}
        for calorimeter in calorimeter_names:
            calorimeters[calorimeter] = ak.zip(
                {
                    "had": events[calorimeter + "_calo_hits_energyHad"],
                    "em": events[calorimeter + "_calo_hits_energyEM"],
                    "energy": events[calorimeter + "_calo_hits_energyEM"]
                    + events[calorimeter + "_calo_hits_energyHad"],
                    "eta": events[calorimeter + "_calo_hits_eta"],
                    "phi": events[calorimeter + "_calo_hits_phi"],
                }
            )

        calorimeters["ecal"] = ak.concatenate(
            (calorimeters[em] for em in em_calorimeters), axis=-1
        )

        associated_hcal_energy = ak.ArrayBuilder()
        associated_hcal_energy = get_associated_energy(
            associated_hcal_energy, gen_muons, calorimeters["hcal"]
        )

        associated_ecal_energy = ak.ArrayBuilder()
        associated_ecal_energy = get_associated_energy(
            associated_ecal_energy, gen_muons, calorimeters["ecal"]
        )

        output["muons"].fill(
            p=ak.flatten(gen_muons.p),
            # eta=ak.flatten(gen_muons.eta),
            ecal=ak.flatten(associated_ecal_energy),
            hcal=ak.flatten(associated_hcal_energy),
        )

        return output

    def postprocess(self, accumulator):
        """Return our total."""
        return accumulator
