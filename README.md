Exploratory repository which attempts to use [coffea](https://coffeateam.github.io/coffea/index.html) as an alternative means of processing and analyzing [Cathode Strip Chamber](https://cms.cern/detector/detecting-muons/cathode-strip-chambers) (CSC) data.

The repository contains:
- Signal-to-background analysis Cathode Local Charged Tracks (CLCTs) using [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curves
- Algorithms used to estimate muon momentum via the energy they lose along their flight path

## Setup

On an [LXPLUS](https://information-technology.web.cern.ch/book/lxplus-service/lxplus-guide) node, first clone the git repository with

```bash
git clone https://github.com/williamnash/CSCCoffea.git
cd CSCCoffea
```
Then setup the virtual environment

```bash
source setup.sh
```

After the initial installation, one must reactivate the virtual environment again before running the code

## Run

Analysis is broken up into `processors`, which hold the logic which operates on each event, and `executors`, which distributes the computation and ultimately collects it. Once the setup is complete (and with an active virtual environment) one can run a script in the `src/` folder via

```bash
python template_executor.py
```

this script will produce an output histogram (png) holding information related to a sample dataset.


## Coffea usage

The primary idea is that `for` loops in python are slow (due to type-checking overhead on each iteration), so using [awkward array](https://awkward-array.readthedocs.io/en/latest/) (based on numpy) affords loops that can iterate ~ 100x faster than they would otherwise.

This is done using combinations of masking, slicing, and broadcasting, which affords a diverse selection of operations that can all be done without a single for loop. This also has the added benefit of both not having to keep track of indices when iterating through arrays, as well as code readability.

For instance a mask can be created with

```python
import numpy as np
a = np.array([1,2,3,4,5,6])
mask = a < 3
print(mask)
```
Which gives
```bash
array([ True,  True, False, False, False, False])
```
These masks can also be used to select for elements in the array (particularly useful for us)

```python
print(a[mask])
```

Which gives

```bash
[1 2]
```
