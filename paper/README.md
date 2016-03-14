#Data
```bash
>>> import numpy as np
>>> perfs = np.load("Feb_mean_perf_salience_delay_allcues.npy")
```
The stimulus set is the key of to access respective data. The stimuli are...
```bash
"[3, 0]" , "[3, 1]" , "[3, 2]" , "[2, 0]" , "[2, 1]" , "[1, 0]"
```

For salience data...
```bash
>>> perfs["[3, 0]"]["salience"][0].shape
(36, 50)
>>> perfs["[3, 1]"]["salience"][0].shape
(36, 50)
```
For each of the following 36 relative salience difference values, 50 trials are run.
```bash
saliences = np.arange(36)*7.0/100
```

For delay data...
```bash
>>> perfs["[3, 0]"]["delay"][0].shape
(41, 50)
>>> perfs["[3, 1]"]["delay"][0].shape
(41, 50)
```

For each of the following 41 delays, 50 trials are run.
```
delays = np.arange(41)
```
