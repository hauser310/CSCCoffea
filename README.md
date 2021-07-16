Exploratory repository which attempts to use [coffea](https://coffeateam.github.io/coffea/index.html) as an alternative means of processing and analyzing CSC data

##



## Setup

Starting your Coffea-based scripts with the `setup.sh` script will automatically handle the setup and (de)activation of a Coffea-ready virtual environment.

Recommended usage:
```
source setup.sh
```

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
