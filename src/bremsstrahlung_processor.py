"""Derive muon momentum via calorimetry."""
import awkward as ak
import numpy as np
import numba
from coffea import hist, processor
from helpers import (
    serial_to_endcap,
    serial_to_station,
    serial_to_ring,
    serial_to_chamber,
    theta_to_eta,
    pt_eta_to_p,
)

# register our candidate behaviors
from coffea.nanoevents.methods import candidate

ak.behavior.update(candidate.behavior)


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


@numba.njit
def get_deltar_weighted_associated_energy(
    associated_array, gen_muons, detector_deposits
):
    """Find the energy deposits associated with each gen muon."""
    for event_muons, event_deposits in zip(gen_muons, detector_deposits):
        associated_array.begin_list()
        for muon in event_muons:
            associated_deposits = 0.0
            for deposit in event_deposits:
                # far away deposits count for more
                associated_deposits += delta_r(muon, deposit) * deposit.energy
            associated_array.append(associated_deposits)
        associated_array.end_list()
    return associated_array


@numba.njit
def get_p_at_exit(p_at_exit, gen_muons, sim_hits):
    """Get the muon momentum as it leaves the detector."""
    for event_muons, event_hits in zip(gen_muons, sim_hits):
        p_at_exit.begin_list()
        for muon in event_muons:
            muon_p_at_exit = -1
            for hit in event_hits:
                if delta_r(muon, hit) < DR_CUT:
                    # entry of chamber is last measure part of muon
                    muon_p_at_exit = hit.p_at_entry
            p_at_exit.append(muon_p_at_exit)
        p_at_exit.end_list()
    return p_at_exit


class BremsstrahlungProcessor(processor.ProcessorABC):
    """Runs the analysis."""

    def __init__(self):
        """Initialize."""
        self._accumulator = processor.dict_accumulator(
            {
                "allevents": processor.defaultdict_accumulator(float),
                "all_muons": hist.Hist(
                    "Muons",
                    hist.Bin("p", "$p$ [GeV]", 100, -10, 4010),
                    hist.Bin("eta", "$\\eta$", 100, 0.5, 3.0),
                    hist.Bin("phi", "$\\phi$", 100, -3.15, 3.15),
                ),
                "muon_deposits": hist.Hist(
                    "Muons",
                    hist.Bin(
                        "p", "$p$ [GeV]", np.array([0, 800, 1600, 2400, 3200, 4000])
                    ),
                    hist.Bin(
                        "hcal", "HCAL energy loss [GeV]", np.logspace(-2.5, 0.5, num=51)
                    ),
                    hist.Bin(
                        "ecal", "ECAL energy loss [GeV]", np.logspace(-1.0, 2.0, num=51)
                    ),
                    hist.Bin(
                        "p_exit",
                        "$p$ at exit [GeV]",
                        np.array([0, 800, 1600, 2400, 3200, 4000]),
                    ),
                ),
                "p_loss": hist.Hist(
                    "Muons",
                    hist.Bin(
                        "p",
                        "$p$ [GeV]",
                        100,
                        0,
                        4000,
                    ),
                    hist.Bin(
                        "p_exit",
                        "$p$ at exit [GeV]",
                        50,
                        0,
                        4000,
                    ),
                    hist.Bin(
                        "dp",
                        "$\\Delta p$ [GeV]",
                        50,
                        0,
                        100,
                    ),
                ),
                "p": processor.column_accumulator(np.zeros(shape=(0,))),
                "dp": processor.column_accumulator(np.zeros(shape=(0,))),
                "phi": processor.column_accumulator(np.zeros(shape=(0,))),
                "eta": processor.column_accumulator(np.zeros(shape=(0,))),
                "hcal": processor.column_accumulator(np.zeros(shape=(0,))),
                "ecal": processor.column_accumulator(np.zeros(shape=(0,))),
                "csc": processor.column_accumulator(np.zeros(shape=(0,))),
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
                "p": pt_eta_to_p(events.gen_pt, events.gen_eta),
                "eta": events.gen_eta,
                "phi": events.gen_phi,
            }
        )

        output["all_muons"].fill(
            p=ak.flatten(gen_muons.p),
            eta=ak.flatten(gen_muons.eta),
            phi=ak.flatten(gen_muons.phi),
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

        csc_hits = ak.zip(
            {
                "ch_id": events.sim_hits_ch_id,
                "pdg_id": events.sim_hits_pdg_id,
                "phi": events.sim_hits_phiAtEntry,
                "eta": theta_to_eta(events.sim_hits_thetaAtEntry),
                "endcap": serial_to_endcap(events.sim_hits_ch_id),
                "station": serial_to_station(events.sim_hits_ch_id),
                "ring": serial_to_ring(events.sim_hits_ch_id),
                "chamber": serial_to_chamber(events.sim_hits_ch_id),
                "p_at_entry": events.sim_hits_pAtEntry,
                "energy": events.sim_hits_energyLoss,
            }
        )

        associated_hcal_energy = ak.ArrayBuilder()
        associated_hcal_energy = get_associated_energy(
            associated_hcal_energy, gen_muons, calorimeters["hcal"]
        )

        associated_ecal_energy = ak.ArrayBuilder()
        associated_ecal_energy = get_associated_energy(
            associated_ecal_energy, gen_muons, calorimeters["ecal"]
        )

        # todo, split this up by chamber / station
        associated_csc_energy = ak.ArrayBuilder()
        associated_csc_energy = get_associated_energy(
            associated_csc_energy, gen_muons, csc_hits
        )

        outer_muon_sim_hits = csc_hits[
            (np.abs(csc_hits.pdg_id) == 13) & (csc_hits.station == 4)
        ]
        p_at_exit = ak.ArrayBuilder()
        p_at_exit = get_p_at_exit(p_at_exit, gen_muons, outer_muon_sim_hits)

        muons = ak.zip(
            {
                "p": gen_muons.p,
                "eta": gen_muons.eta,
                "phi": gen_muons.phi,
                "p_exit": p_at_exit,
                "hcal": associated_hcal_energy,
                "ecal": associated_ecal_energy,
                "csc": associated_csc_energy,
                "dp": gen_muons.p - p_at_exit,
            }
        )

        # select those that have a momentum measured in the last station
        # with non-zero calo deposits

        muons_w_deposits_st4 = muons[
            (muons.p_exit != -1) & (muons.ecal != 0) & (muons.hcal != 0)
        ]

        output["muon_deposits"].fill(
            p=ak.flatten(muons_w_deposits_st4.p),
            p_exit=ak.flatten(muons_w_deposits_st4.p_exit),
            ecal=ak.flatten(muons_w_deposits_st4.ecal),
            hcal=ak.flatten(muons_w_deposits_st4.hcal),
        )

        output["p_loss"].fill(
            p=ak.flatten(muons_w_deposits_st4.p),
            p_exit=ak.flatten(muons_w_deposits_st4.p_exit),
            dp=ak.flatten(muons_w_deposits_st4.dp),
        )

        for var in ["p", "dp", "phi", "eta", "hcal", "ecal", "csc"]:
            if var in ["ecal", "hcal", "csc"]:
                # save logarithm of these energies
                output[var] += processor.column_accumulator(
                    np.log10(ak.flatten(muons_w_deposits_st4[var]).to_numpy())
                )
            else:
                output[var] += processor.column_accumulator(
                    ak.flatten(muons_w_deposits_st4[var]).to_numpy()
                )

        return output

    def postprocess(self, accumulator):
        """Return our total."""
        return accumulator
