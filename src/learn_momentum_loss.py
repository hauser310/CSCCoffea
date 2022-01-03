"""Learn muon momentum loss using energy deposits in the detector."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import numba
from helpers import landau, OUTPUT_DIR
import keras_tuner as kt
from coffea import hist
from tensorflow.keras.layers.experimental import preprocessing
from matplotlib import colors
import scipy.stats as st
from models import neural_network
import scipy
import pickle
import logging


def gauss(x, H, A, x0, sigma):
    """Gaussian function."""
    return H + A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def gauss_fit(x, y):
    """Fit a Gaussian to a sett of x/y points."""
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = scipy.optimize.curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt


MIN = 0
MAX = 4000

# warning: reading untrusted pickled files is *not* safe

# load the landau fit parameters
with open(OUTPUT_DIR + "landau_fit_parameters.pkl", "rb") as f:
    popt = pickle.load(f)

dataset = pd.read_pickle(OUTPUT_DIR + "brem_dataset.pkl")

train_dataset = dataset.sample(frac=0.8, random_state=1)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

# get the momentum (not used for training)
train_p = train_features.pop("p")
test_p = test_features.pop("p")

# get the momentum loss (training labels)
train_labels = train_features.pop("dp")
test_labels = test_features.pop("dp")

# normalize the features such that it has mean 0, std 1
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

tune = False
if tune:

    def neural_network_with_normalizer(hp):
        return neural_network(normalizer, hp)

    # scan over possible hyper-parameters to find the best ones
    tuner = kt.Hyperband(
        neural_network_with_normalizer,
        objective="val_loss",
        max_epochs=20,
        factor=3,
        directory=OUTPUT_DIR,
        project_name="hyperparameter_scan",
    )

    tuner.search(train_features, train_labels, epochs=10, validation_split=0.2)
    model = tuner.get_best_models()[0]
else:
    model = neural_network(normalizer=normalizer)


model.fit(
    train_features,
    train_labels,
    epochs=20,
    # Calculate validation results on 20% of the training data
    validation_split=0.2,
)
predicts = model.predict(test_features)


fig, ax = plt.subplots()
fig.clear()
lims = [0.0, 100.0]
plt.hist2d(
    test_labels.values,
    ak.flatten(predicts),
    bins=(50, 50),
    range=[lims, lims],
    cmap=plt.cm.jet,
    norm=colors.LogNorm(),
)
plt.xlabel("True $\\Delta p$ [GeV/c]")
plt.ylabel("Predicted $\\Delta p$ [GeV/c]")
plt.ylim(lims)
plt.xlim(lims)
plt.plot(lims, lims, color="black", linestyle="--")
for suffix in (".png", ".pdf"):
    plt.savefig(OUTPUT_DIR + "predicted_vs_true_dp" + suffix)

p_axis = np.array([p for p in range(MIN + 10, MAX, 10)])


fig, ax = plt.subplots()

res_hist = hist.Hist(
    "Muons",
    hist.Bin("res_norm", "$p_{reco}-p_{gen}/\\sigma_{reco}$", 100, -3, 3),
    hist.Bin("p", "True $p$ [GeV]", 50, MIN, MAX),
    hist.Bin("pred_p", "Predicted $p$ [GeV]", 50, MIN, MAX),
)

# Calculate -2 ln L
in_count = 0
count = 0
for pred_dp, true_p in zip(predicts, test_p):
    if not count % 1000:
        logging.info(f"Completed: {count}")

    # Define a pdf for p given this momentum loss
    class landau_p_pdf(st.rv_continuous):
        def _pdf(self, x):
            # norm = scipy.integrate.quad(lambda z: landau((pred_dp, z), *popt), 0, 1e5)
            # return landau((pred_dp, x), *popt) / norm
            return landau((pred_dp, x), *popt)

    cv = landau_p_pdf(a=MIN, b=MAX, name="landau_p_pdf")

    ydata = []
    for p in p_axis:
        ydata.append(-2 * cv.logpdf(p))
    ydata = np.array(ydata)
    ydata -= min(ydata)

    @numba.njit
    def find_bounds(ydata, xdata, ypoint=1):
        """
        Find where they array (assumed concave with one minimum) reaches the given number.

        ydata: array of y points
        xdata: array of x points
        ypoint: look for where the ydata is equal to this point
                for -2ln(L), 1 corresponds to 68%, 3.84 corresponds to 95%

        pred_p is the location in x of the min y
        """
        pred_p = xdata[np.argmin(ydata)]
        low = high = -999.0
        for i, point in enumerate(ydata):
            if low == high and point <= ypoint:
                low = xdata[i]
            if low != high and point > ypoint:
                high = xdata[i - 1]
                break
        if high == -999.0:
            high = xdata[-1]
        return pred_p, low, high

    pred_p, low, high = find_bounds(ydata, p_axis, 1.0)  # 68%

    res_hist.fill(res_norm=(pred_p - true_p) / (high - low), pred_p=pred_p, p=true_p)

    if count < 10:
        fig.clear(True)
        fig, ax = plt.subplots()
        ax.plot(p_axis, np.array(ydata))
        plt.xlabel("$p$ [GeV]")
        plt.ylabel("$-2\\Delta ln\\mathcal{L}$")
        ll_ybounds = [-1, 10]
        plt.ylim(ll_ybounds)
        plt.plot([true_p, true_p], ll_ybounds, color="black")
        plt.plot([pred_p, pred_p], ll_ybounds, color="black", linestyle="--", alpha=0.5)
        plt.plot([low, low], ll_ybounds, color="red", linestyle="--")
        plt.plot([high, high], ll_ybounds, color="red", linestyle="--")
        ax.legend(
            ["Landau fit", "True momentum", "Predicted momentum", "$68\\%$ interval"]
        )
        for suffix in (".png", ".pdf"):
            plt.savefig(OUTPUT_DIR + "momentum_pdf_" + str(count) + suffix)

    count += 1
    if true_p <= high and true_p >= low:
        in_count += 1
logging.info(f"Fraction in interval: {1.*in_count/(count)}")

fig.clear()
ax = hist.plot1d(res_hist.project("res_norm"), overflow="all")
ax.get_legend().remove()
for suffix in (".png", ".pdf"):
    plt.savefig(OUTPUT_DIR + "momentum_res_norm" + suffix)

fig.clear(True)
ax = hist.plot2d(
    res_hist.project("p", "pred_p"),
    xaxis="p",
    patch_opts={"norm": colors.LogNorm()},
)
p_lims = [MIN, MAX]
plt.plot(p_lims, p_lims, color="black", linestyle="--")
for suffix in (".png", ".pdf"):
    plt.savefig(OUTPUT_DIR + "muon_pred_vs_true_p" + suffix)


x_points = np.array(
    res_hist.project("res_norm").to_boost().axes.centers.tolist()
).flatten()
y_points = res_hist.project("res_norm").to_boost().values()
H, A, x0, sigma = gauss_fit(x_points, y_points)
FWHM = 2.35482 * sigma

logging.info("The offset of the gaussian baseline is", H)
logging.info("The center of the gaussian fit is", x0)
logging.info("The sigma of the gaussian fit is", sigma)
logging.info("The maximum intensity of the gaussian fit is", H + A)
logging.info("The Amplitude of the gaussian fit is", A)
logging.info("The FWHM of the gaussian fit is", FWHM)
