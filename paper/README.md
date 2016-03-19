# [Separate] Salience, Delay 
```bash
>>> import numpy as np
>>> perfs = np.load("Feb_mean_perf_salience_delay_allcues.npy")
```
The stimulus set is the key to access respective data. The stimuli are...
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

# [Combined] Salience & Delay 
```bash
>>> import numpy as np
>>> perfs = np.load("icann_2D_perf_salience_delay_allcues.npy")
```

The stimulus set is the key to access respective data. The stimuli are...
```bash
"[3, 0]" , "[3, 1]" , "[3, 2]" , "[2, 0]" , "[2, 1]" , "[1, 0]"
```

For each of the following 21 relative salience values and 21 delay values (ranging from 0 to 40 units), 50 trials are run.
```bash
saliences = np.arange(21)*2*7.0/100
delays = np.arange(21)*2
```

```bash
>>> perfs.dtype
dtype([('[1, 0]', '<f8', (50,)), ('[2, 0]', '<f8', (50,)), ('[3, 0]', '<f8', (50,)), ('[2, 1]', '<f8', (50,)), ('[3, 1]', '<f8', (50,)), ('[3, 2]', '<f8', (50,))])
>>> perfs.shape
(21, 21)
```

For a given pair of (salience, delay) combination and the stimuli set, the mea performance over 50 trials could be found...
```bash
>>> perfs[0,0]["[3, 0]"].mean()
1.0
>>> perfs[5,5]["[3, 0]"].mean()
0.963
```
