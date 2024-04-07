# Causal Discovery with Deductive Reasoning: One Less Problem

This python package is for the paper titled Causal Discovery with Deductive Reasoning: One Less Problem submission submitted to UAI 2024 (under review.)

Upon the requests by reviewers, we modified many parts of our codebase implementing additional functionalities during the short discussion period.
Regardless, we organized python3 files into folders such as `scripts`, `experiments`, and `algos` (algorithms) where
`scripts` folder is separated from module `cddd` in source folder `src`.
Due to some code changes, we do not exactly reproduce the results
since some of the random seeds are changed. But there will be no problem observing trends similar to the tables and figures reported in the paper.

## Preparation

At this moment, we only support Unix-based systems (we hard-coded path separators as `/`).
Please use `python` 3.11 since we did not test other versions.
You may install python packages necessary to
run experiments by first checking `requirements.txt` (you may ignore version constraints for the packages listed in the requirements.txt.)

`pip3 install -r requirements.txt`

## Install the Package:

Now, move to the source directory in `CD_DD` and install the module with pip.
We will assume that you downloaded `CD_DD` in your home directory.

`cd ~/CD_DD/src`

`pip3 install .`

## Running Scripts

To run the algorithms, you will need to 1) generate data, 2) conduct experiments, and 3) draw results.
Those files in scripts folder, `scripts`

`cd ~/CD_DD/scripts`

### Generating Data:

`sample_data.py` script will create two directories `Ground_truth` and `Sampled_datasets` under `~/CD_DD/data`
and generate DAGs and corresponding samples used in the paper.   
Please change `WORKING_DIR` variable in the script properly, e.g., `WORKING_DIR = '/home/johndoe/CD_DD'`
(Note that the current code does not expand tilde `~` to an actual home directory. Please specify `WORK_DIR` based on an absolute path.)

`python3 sample_data.py`

It will take about 2 minutes.
Note that there will be warning messages about probabilities not being sum to 1.0. You can ignore those warnings.
Also it may take (less than) 1GB of storage.

### Conducting Experiments:

`conduct_experiments.py` will run multiple experiments simultaneously.
If you don't need to reproduce results similar to the paper, you may select some of the datasets and parameters for algorithms.
`~/CD_DD/results` will be created and csv files will be generated. You can also control parallelism by changing values for `n_jobs` (the number of CPU cores to utilize.)
Again, please change `WORKING_DIR` variable in the script properly, e.g., `WORKING_DIR = '/home/johndoe/CD_DD'`

`python3 conduct_experiments.py`

Note that full experiments may take about 4 hours utilizing 128 CPUs.

### Drawing Plots:

`draw_results.py` script will draw plots (and save them to `results` folder) based on the csv files saved in `results` after `conduct_experiments.py`.
Note that the code may not output plots properly if some of experiments are not conducted. Sorry for the inconvenience.

`python3 draw_results.py`


