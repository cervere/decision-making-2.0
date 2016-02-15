## Introduction

This is a reference implementation for the following models:

* M. Guthrie, A Leblois, A. Garenne, T. Boraud, "Interaction between cognitive
  and motor cortico-basal ganglia loops during decision making: a computational
  study", Journal of Neurophysiology, 2013. 

* Meropi, Topalidou, and Nicolas P. Rougier. "[Re] Interaction between cognitive and motor cortico-basal ganglia loops during decision making: a computational study." ReScience 1.1 (2015).


## Installation

It requires python, numpy, cython and matplotlib:

```bash
$ pip install numpy
$ pip install cython
$ pip install matplotlib
```

To compile the model, just type:

```bash
$ python setup_basic.py build_ext --inplace
```

Then you can run a single trial:

```bash
$ python basic/experiments/single-trial.py
```

