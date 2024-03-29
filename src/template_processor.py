"""Template processor to show how things work."""
import awkward as ak
from coffea import hist, processor

# register our candidate behaviors
from coffea.nanoevents.methods import candidate

ak.behavior.update(candidate.behavior)


class TemplateProcessor(processor.ProcessorABC):
    """Runs the analysis."""

    def __init__(self):
        """Initialize."""
        dataset_axis = hist.Cat("dataset", "Primary dataset")

        """
        First, you need to define a multi-dimensional histogram to hold
        the data. Follow the form:

            "tree": hist.Hist(
                "Thing we're counting",
                hist.Bin("leaf", "$units$", #number of bins, #min, #max),
            ),

        """
        self._accumulator = processor.dict_accumulator(
            {
                "allevents": processor.defaultdict_accumulator(float),
                "events": hist.Hist(
                    "Events",
                    dataset_axis,
                    hist.Bin("nMuons", "Number of muons", 6, 0, 6),
                ),
                "muons": hist.Hist(
                    "Muons",  # <- things we are counting
                    hist.Bin("pt", "$p_{T}$ [GeV]", 50, 0, 200),
                    hist.Bin("eta", "$\\eta$", 50, -2.5, 2.5),
                    hist.Bin("mass", "$m_{\\mu}$ [GeV]", 50, 0.1, 0.11),
                    hist.Bin("charge", "$q$ [e]", 50, -10, 10),
                ),
                "dimuons": hist.Hist(
                    "Dimuons",
                    hist.Bin("mass", "$m_{\\mu\\mu}$ [GeV]", 60, 2, 4),
                    hist.Bin("charge", "$q$ [e]", 50, -0.1, 0.1),
                ),
                "segment": hist.Hist(
                    "Segment",
                    hist.Bin("mu_id", "$\\mu_id$", 8, -2, 2),
                    hist.Bin("dxdz", "$dxdz$", 50, -1, 1),
                    hist.Bin("chisq", "$\\chi^2$", 20, 0, 100),
                    hist.Bin("nHits", "Number of hits", 8, 0, 8),
                ),
                "segment_muon": hist.Hist(
                    "Segment_muon",
                    hist.Bin("chisq", "$\\chi^2$", 20, 0, 100),
                    hist.Bin("mu_id", "$\\mu_id$", 8, -2, 2),
                    hist.Bin("pt", "$p_{T}$ [GeV]", 50, 0, 25),
                    hist.Bin("dxdz", "$dxdz$", 50, -0.25, 0.25),
                    hist.Bin("nHits", "Number of hits", 8, 0, 8),
                ),
                "segment_slice_dxdz": hist.Hist(
                    "Muons",
                    hist.Cat("pt_slice", "slice based on pt"),
                    hist.Bin("slice_data", "$dxdz$", 50, -1, 1),
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

        """
        Now, you'll need to unzip the variable, this stores the data into
        the histograms we defined earlier.

        variable = ak.zip(
            {
                "leaf": location_in_root_file,
            },
        )
        """

        muons = ak.zip(
            {
                "pt": events.muon_pt,
                "eta": events.muon_eta,
                "phi": events.muon_phi,
                "mass": ak.Array([0.105658375 for _ in range(len(events.muon_pt))]),
                "charge": events.muon_q,
            },
            with_name="PtEtaPhiMCandidate",
        )
        segment = ak.zip(
            {
                "mu_id": events.segment_mu_id,
                "dxdz": events.segment_dxdz,
                "chisq": events.segment_chisq,
                "nHits": events.segment_nHits,
            },
        )

        """
        Finally, we must assign the histograms to the output to return
        to template_executor.py for plotting.

        output["variable"].fill(
            leaf=ak.flatten(variable.leaf),
            )

        """

        output["segment"].fill(
            mu_id=ak.flatten(segment.mu_id),
            dxdz=ak.flatten(segment.dxdz),
            chisq=ak.flatten(segment.chisq),
            nHits=ak.flatten(segment.nHits),
        )

        segments_w_muon = segment[segment.mu_id != -1]

        segment_associated_muons = muons[segments_w_muon.mu_id]

        segment_slice_2 = segments_w_muon[
            (segment_associated_muons.pt > 2) & (segment_associated_muons.pt < 5)
        ]
        segment_slice_3 = segments_w_muon[
            (segment_associated_muons.pt > 5) & (segment_associated_muons.pt < 10)
        ]
        segment_slice_4 = segments_w_muon[segment_associated_muons.pt > 10]

        output["segment_muon"].fill(
            mu_id=ak.flatten(segments_w_muon.mu_id),
            chisq=ak.flatten(segments_w_muon.chisq),
            pt=ak.flatten(segment_associated_muons.pt),
            dxdz=ak.mean(abs(segments_w_muon.dxdz)),
            nHits=ak.flatten(segments_w_muon.nHits),
        )

        output["segment_slice_dxdz"].fill(
            pt_slice="2 GeV < $p_{T}$ < 5 GeV",
            slice_data=ak.flatten(segment_slice_2.dxdz),
        )
        output["segment_slice_dxdz"].fill(
            pt_slice="5 GeV < $p_{T}$ < 10 GeV",
            slice_data=ak.flatten(segment_slice_3.dxdz),
        )
        output["segment_slice_dxdz"].fill(
            pt_slice="$p_{T}$ > 10 GeV", slice_data=ak.flatten(segment_slice_4.dxdz)
        )

        return output

    def postprocess(self, accumulator):
        """Return our total."""
        return accumulator
