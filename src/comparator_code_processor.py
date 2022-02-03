"""Processor to create histograms and generate plots from fed comparator code."""
import awkward as ak
from coffea import hist, processor

from coffea.nanoevents.methods import candidate

ak.behavior.update(candidate.behavior)


class ComparatorCodeProcessor(processor.ProcessorABC):
    """Runs the analysis."""

    def __init__(self):
        dataset_axis = hist.Cat("pcc", "Pattern-Comparator Code Combination")

        """Initialize."""
        self._accumulator = processor.dict_accumulator(
            {
                "allevents": processor.defaultdict_accumulator(float),
                "events": hist.Hist(
                    "Events",
                    dataset_axis,
                    hist.Bin("nMuons", "Number of muons", 6, 0, 6),
                ),
                "LUT": hist.Hist(
                    "LUT",
                    dataset_axis,
                    hist.Bin("position", "$position$", 20, -1, 0),
                    hist.Bin("slope", "$slope$", 20, -0.5, 0.5),
                    hist.Bin("pt", "$pt$", 50, 0, 50),
                    hist.Bin("multiplicity", "$multiplicity$", 3, 1, 4),
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

        lut = ak.zip(
            {
                "position": events.position,
                "slope": events.slope,
                "pt": events.pt,
                "multiplicity": events.multiplicity,
            },
        )

        output["LUT"].fill(
            pcc=dataset,
            position=lut.position,
            slope=lut.slope,
            pt=lut.pt,
            multiplicity=lut.multiplicity,
        )
        return output

    def postprocess(self, accumulator):
        """Return our total."""
        return accumulator
