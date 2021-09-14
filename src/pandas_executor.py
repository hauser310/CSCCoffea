"""Executor that processes pandas dataframe and generates ratio and ROC plots from analysis."""
import matplotlib.pyplot as plt
import pandas as pd
from coffea import hist
from cycler import cycler

"""increase resolution of output .png files"""
plt.figure(dpi=400)

"""create a dataframe from .csv file"""
df = pd.read_csv(
    "/afs/cern.ch/user/e/ezweig/CSCUCLA/CSCPatterns/outputs/LUTBuilder_TEMPLATE.csv",
    header=None,
    names=["key_pattern", "key_code", "foundSegment", "entry_layers", "entry_chi2"],
)

"""generate two histograms: entire dataset, and smaller one that's sliced"""
clct = hist.Hist(
    "CLCTs",
    hist.Cat("hist_type", "$hist_type$"),
    hist.Bin("key_pattern", "$key pattern$", 6, 50, 110),
    hist.Bin("key_code", "$key code$", 64, 0, 4096),
    hist.Bin("foundSegment", "$foundSegment$", 2, 0, 2),
    hist.Bin("entry_layers", "$number\\ of\\ layers$", 7, 0, 7),
    hist.Bin("entry_chi2", "$\\chi^2$", 20, 0, 10),
)

reduced_clct = hist.Hist(
    "CLCTs",
    hist.Cat("hist_type", "$hist_type$"),
    hist.Bin("key_pattern", "$key pattern$", 6, 50, 110),
    hist.Bin("key_code", "$key code$", 64, 0, 4096),
    hist.Bin("foundSegment", "$foundSegment$", 2, 0, 2),
    hist.Bin("entry_layers", "$number\\ of\\ layers$", 7, 0, 7),
    hist.Bin("entry_chi2", "$\\chi^2$", 20, 0, 10),
)

df2 = df[df["foundSegment"] == 1]

"""Fill the histogram with data from dataframes."""
clct.fill(
    hist_type="all",
    key_pattern=df["key_pattern"].to_numpy(),
    key_code=df["key_code"].to_numpy(),
    foundSegment=df["foundSegment"].to_numpy(),
    entry_layers=df["entry_layers"].to_numpy(),
    entry_chi2=df["entry_chi2"].to_numpy(),
)
reduced_clct.fill(
    hist_type="has segment",
    key_pattern=df2["key_pattern"].to_numpy(),
    key_code=df2["key_code"].to_numpy(),
    foundSegment=df2["foundSegment"].to_numpy(),
    entry_layers=df2["entry_layers"].to_numpy(),
    entry_chi2=df2["entry_chi2"].to_numpy(),
)

"""make a nice ratio plot, adjusting some font sizes"""
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)
fig, (ax, rax) = plt.subplots(
    nrows=2, ncols=1, figsize=(7, 7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
)
fig.subplots_adjust(hspace=0.07)

colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"]
ax.set_prop_cycle(cycler(color=colors))

fill_opts = {"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8}

data_err_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.0,
    "color": "k",
    "elinewidth": 1,
}

"""plot the MC first"""
hist.plot1d(
    clct.project("key_pattern"),
    ax=ax,
    clear=False,
    stack=True,
    line_opts=None,
    fill_opts=fill_opts,
)

"""now the pseudodata, setting clear=False to avoid overwriting the previous plot"""
hist.plot1d(
    reduced_clct.project("key_pattern"),
    ax=ax,
    clear=False,
    fill_opts=fill_opts,
    error_opts=data_err_opts,
)

ax.autoscale(axis="x", tight=True)
ax.set_ylim(100, None)
ax.set_xlabel(None)
ax.set_yscale("log")
leg = ax.legend(["all", "has segment"])

"""now we build the ratio plot"""
hist.plotratio(
    num=reduced_clct.project("key_pattern"),
    denom=clct.project("key_pattern"),
    ax=rax,
    error_opts=data_err_opts,
    denom_fill_opts={},
    guide_opts={},
    unc="num",
)
rax.set_ylabel("Ratio")
rax.set_ylim(0.001, 1)
rax.set_yscale("log")

plt.savefig("pandas/pandas_coffea_key_pattern.png")

"""want to find both false positive rate and true positive rate for a classification"""

true_pos_rate_layers = []
false_pos_rate_layers = []
accuracy_layers = []

true_pos_rate_chi2 = []
false_pos_rate_chi2 = []
accuracy_chi2 = []

true_pos_rate_pattern = []
false_pos_rate_pattern = []
accuracy_pattern = []


def get_positives(data):
    return data[data.foundSegment is True].size


def get_negatives(data):
    return data[data.foundSegment is False].size


def get_true_positives(data, classifier):
    """Find the total positives that also are selected by our classifier."""
    return data[data["foundSegment"] & classifier(data)].size


def get_true_negatives(data, classifier):
    """Find the total negatives that also are not selected by our classifier."""
    return data[(data["foundSegment"] == 0) & classifier(data)].size


def layer_classifier(data, layer_threshold):
    """Return an array mask passing our selection."""
    return data["entry_layers"] > layer_threshold


def not_layer_classifier(data, layer_threshold):
    """Return an array mask failing our selection."""
    return data["entry_layers"] <= layer_threshold


def chi2_classifier(data, chi2_threshold):
    """Return an array mask passing our selection."""
    return data["entry_chi2"] < chi2_threshold


def not_chi2_classifier(data, chi2_threshold):
    """Return an array mask failing our selection."""
    return data["entry_chi2"] >= chi2_threshold


def pattern_classifier(data, pattern_threshold):
    """Return an array mask passing our selection."""
    return data["key_pattern"] > pattern_threshold


def not_pattern_classifier(data, pattern_threshold):
    """Return an array mask failing our selection."""
    return data["key_pattern"] <= pattern_threshold


for layer in range(0, 7):
    # print(f"Layer: {layer}")
    def classifier(data):
        return layer_classifier(data, layer)

    def not_classifier(data):
        return not_layer_classifier(data, layer)

    # print(f"Classifier: {classifier(df)}")
    # print(f"!Classifier: {not_classifier(df)}")

    # print(f"Negatives: {get_negatives(df)}")
    # print(f"Positives: {get_positives(df)}")
    # print(f"True Positives: {get_true_positives(df,classifier)}")
    # print(f"True Negatives: {get_true_negatives(df,not_classifier)}")
    pos = get_positives(df)
    neg = get_negatives(df)
    true_pos = get_true_positives(df, classifier)
    true_neg = get_true_negatives(df, not_classifier)
    false_neg = pos - true_pos
    false_pos = neg - true_neg
    tpr = (
        0.0 if true_pos + false_neg == 0.0 else float(true_pos / (true_pos + false_neg))
    )
    fpr = (
        0.0
        if false_pos + true_neg == 0.0
        else float(false_pos / (false_pos + true_neg))
    )
    true_pos_rate_layers.append(tpr)
    false_pos_rate_layers.append(fpr)
    accuracy_layers.append(
        (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    )

for chi2 in range(0, 10):
    # print(f"Chi2: {chi2}")
    def classifier(data):
        return chi2_classifier(data, chi2)

    def not_classifier(data):
        return not_chi2_classifier(data, chi2)

    # print(f"Classifier: {classifier(df)}")
    # print(f"!Classifier: {not_classifier(df)}")

    # print(f"Negatives: {get_negatives(df)}")
    # print(f"Positives: {get_positives(df)}")
    # print(f"True Positives: {get_true_positives(df,classifier)}")
    # print(f"True Negatives: {get_true_negatives(df,not_classifier)}")
    pos = get_positives(df)
    neg = get_negatives(df)
    true_pos = get_true_positives(df, classifier)
    true_neg = get_true_negatives(df, not_classifier)
    false_neg = pos - true_pos
    false_pos = neg - true_neg
    tpr = (
        0.0 if true_pos + false_neg == 0.0 else float(true_pos / (true_pos + false_neg))
    )
    fpr = (
        0.0
        if false_pos + true_neg == 0.0
        else float(false_pos / (false_pos + true_neg))
    )
    true_pos_rate_chi2.append(tpr)
    false_pos_rate_chi2.append(fpr)
    accuracy_chi2.append(
        (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    )

for pattern in range(0, 4):
    # print(f"PatternID: {55 + 20*pattern}")
    def classifier(data):
        return pattern_classifier(data, pattern)

    def not_classifier(data):
        return not_pattern_classifier(
            data, pattern
        )  # print(f"Classifier: {classifier(df)}")

    # print(f"!Classifier: {not_classifier(df)}")

    # print(f"Negatives: {get_negatives(df)}")
    # print(f"Positives: {get_positives(df)}")
    # print(f"True Positives: {get_true_positives(df,classifier)}")
    # print(f"True Negatives: {get_true_negatives(df,not_classifier)}")
    pos = get_positives(df)
    neg = get_negatives(df)
    true_pos = get_true_positives(df, classifier)
    true_neg = get_true_negatives(df, not_classifier)
    false_neg = pos - true_pos
    false_pos = neg - true_neg
    tpr = (
        0.0 if true_pos + false_neg == 0.0 else float(true_pos / (true_pos + false_neg))
    )
    fpr = (
        0.0
        if false_pos + true_neg == 0.0
        else float(false_pos / (false_pos + true_neg))
    )
    true_pos_rate_pattern.append(tpr)
    false_pos_rate_pattern.append(fpr)
    accuracy_pattern.append(
        (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    )

# print(accuracy_layers)
# print(accuracy_chi2)
# print(accuracy_pattern)
fig, ax = plt.subplots()
plt.axes(aspect="equal")
lims = [0.0, 1.0]
plt.plot(lims, lims, color="black", linestyle="--")

ax = plt.plot(false_pos_rate_layers, true_pos_rate_layers)
ax = plt.plot(false_pos_rate_chi2, true_pos_rate_chi2)
ax = plt.plot(false_pos_rate_pattern, true_pos_rate_pattern)
plt.legend(["", "num of layers", "$\\chi^2$", "patternID"])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("pandas/pandas_ROC.png")
