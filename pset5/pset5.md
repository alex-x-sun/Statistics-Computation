# Problem Set 5

```Python
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
import statistics as stat
import seaborn as sns
import random
import math
from datetime import datetime

os.chdir('e:/MIT4/statistics-Computation/pset5')
```
## 5.1  BPP Data Analysis
```python
data = pd.read_csv('data/PriceStats_CPI.csv')

data.head()

data['date'] = pd.to_datetime(data['date'])

def diff_month(d1, d2):
    return (d1.year - d2.year)*12 + d1.month - d2.month


ini_date = data['date'][0]


data['month_index'] = data.apply(lambda row: diff_month(row.date, ini_date), axis=1)

data.head()
agg_dict = {'CPI': 'first', 'BPP': 'mean'}
data_monthly = data.groupby('month_index').agg(agg_dict)




```
### (a)
First, we will try to predict the monthly CPI without using the BER or PriceStats. Fit an AR model to the CPI data (take therst CPI value of each month as that month's CPI, you may or may not want to work in log scale in order to make the model comparable to models you t in part (c)) and report the mean squared prediction error for 1 month ahead forecasts. Which order model gives the best predictions?

```python
data =
```

### (b)
How might you calculate monthly inflation rates from the CPI data and your 1 month ahead predictions? How about from PriceStats data? And BER data? (What dates would you use? Or would you use an average of many dates?) Overlay your estimates of monthly inflation rates (there should be 4 lines, one for each dataset, plus the predictions) over time (months from September 2013 onward).

```python

```

### (c)
Next, we will include external regressors to try to improve the predictions. Include as external regressors monthly average PriceStats data and BER data tot a new AR model to the CPI. Report your prediction error. Try instead using PriceStats data and BER data from the last day of each month as your external regressors. Fit another AR model. Which model performs better in prediction? (Hint: Again, in order to match the units of your predictors and responses, you'll want to either work on a log scale or work with inflation rates. Please justify your choices in which values you decide to work with.)

```python

```

### (d)
Try to improve your model from part (c). What is the smallest prediction error you can obtain? You might consider including MA terms, adding a seasonal AR term, or adding multiple daily values (or values from dierent months) of PriceStats and BER data as external regressors.

```python

```

### (e)
Consider the MA(1) model, X_t = W_t + theta*W_t-1, where {W_t} ~ WN(0, sigma^2). Find the autocovariance function of {X_t}.

```python

```

### (f)
Consider the AR(1) model, X_t = phi*X_t-1 + W_t, where {W_t} ~ WN(0, sigma^2). Suppose |phi| < 1. Find the autocovariance function of {X_t}.

```python

```

## 5.2  The Mauna Loa CO2 Concentration

### (a)
Fit the data to a linear model.Plot the data and thefit.

```python

```

### (b)
Fit the data to a quadratic model. Plot the data and the fit.

```python

```

### (c)
Which fit (F1 or F2) is better in capturing the trend in the data? Explain.

```python

```

### (d)
Consider F2(t). We will now extract the periodic component which appears in the data. Average the residual Ci - F2(t_i) over each month. Namely, collect all the data for Jan (resp.Feb, Mar, etc) and average them to get one data point for Jan (resp. Feb, Mar, etc). The collection of those points can be interpolated to form a periodic signal P_i. Plot P_i.

```python

```

### (e)
Plot thet F2(t_i) + P_i. What can we conclude on the variation of the CO2 concentration since 1958?

```python

```
