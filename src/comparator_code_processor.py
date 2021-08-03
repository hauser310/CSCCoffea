"""Ethan processor to create histograms and generate plots."""
import awkward as ak
from coffea import hist, processor

# register our candidate behaviors
from coffea.nanoevents.methods import candidate

ak.behavior.update(candidate.behavior)


class ComparatorCodeProcessor(processor.ProcessorABC):
    """Runs the analysis."""

    def __init__(self):
        dataset_axis = hist.Cat("dataset", "Primary dataset")

        """Initialize."""
        """First, you need to define a multi-dimensional histogram to hold
                the data. Follow the form.
                "tree": hist.Hist(
                    "Thing we're counting",
                    hist.Bin("leaf", "$units$", #number of bins, #min value, #max value),
                    ),"""
        self._accumulator = processor.dict_accumulator(
            {
                "allevents": processor.defaultdict_accumulator(float),
                "events": hist.Hist(
                    "Events",
                    dataset_axis,
                    hist.Bin("nMuons", "Number of muons", 6, 0, 6),
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

        """Now, you'll need to unzip the variable, this stores the data into
        the histograms we defined earlier.
        variable = ak.zip(
            {
                "leaf": location_in_root_file,
            },
        )"""

        """Finally, we must assign the histograms to the output to return
        to template_executor.py for plotting.
        output["variable"].fill(
            leaf=ak.flatten(variable.leaf),
            )"""

        return output

    def postprocess(self, accumulator):
        """Return our total."""
        return accumulator
