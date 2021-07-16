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
        self._accumulator = processor.dict_accumulator(
            {
                "allevents": processor.defaultdict_accumulator(float),
                "muons": hist.Hist(
                    "Muons",  # <- things we are counting
                    hist.Bin("pt", "$p_{T}$ [GeV]", 50, 0, 200),
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

        muons = ak.zip(
            {
                "pt": events.muon_pt,
                "eta": events.muon_eta,
                "phi": events.muon_phi,
                "mass": ak.Array([0.105658375 for _ in range(len(events.muon_pt))]),
                "charge": events.muon_q,
            }
        )

        muons = muons[muons.pt > 10]  # select muons with pT > 10 GeV

        output["muons"].fill(pt=ak.flatten(muons.pt))

        return output

    def postprocess(self, accumulator):
        """Return our total."""
        return accumulator
