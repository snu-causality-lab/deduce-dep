# Causal Discovery with Deductive Reasoning : One Less Problem

To implement additional functionalities upon the requests by reviewers,
we modified our codebase a bit in hurry during the discussion period.
Hence, the code maybe a bit untidy. Still, we organized python3 files into folders such as `scripts`, `experiments`, and `algos` (algorithms).
`scripts` folder is separated from module `cddd` in source folder `src`.

## Prerequisite

At this moment, we only support Unix-based systems (we hard-coded a path separator as `/`).
Make sure that you are using intended `pip` and `python`, by checking `which pip` and `which python`
Please use `python` 3.11 or higher. You may install packages necessary to run all the experiments by checking `requirements.txt` (you may ignore version constraints for the packages listed in the requirements.txt.)

`pip3 install -r requirements.txt`

## Install:

move to the source directory in `CD_DD` and install the module with pip.
Assuming you are at a directory enclosing `CD_DD`.

`cd CD_DD/src`

`pip3 install .`

### Running Scripts

To run the algorithms, you will need to 1) generate data, 2) conduct experiments, and 3) draw results.
Assuming you are at `CD_DD` folder, first change directory to `scripts`

`cd scripts`

## Generating Data:

`sample_data.py` script will create two directories `Ground_truth` and `Sampled_datasets` under `CD_DD/data`
and generate DAGs and corresponding samples used in the paper.   
Please change `WORKING_DIR` variable in the script properly, e.g., `WORKING_DIR = '/home/johndoe/CD_DD'`

`python3 sample_data.py`

It will take about less than 2 minutes. Note that there will be warning messages about probabilities not being sum to 1.0. You can ignore those warnings.  

## Conducting Experiments:

`conduct_experiments.py` will run multiple experiments simultaneously.
If you don't need to reproduce results similar to the paper, you may select some datasets and parameters for algorithms.
`CD_DD\results` will be created and csv files will be generated. You can also control parallelism by changing values for `n_jobs` (the number of CPU cores to utilize.)
Again, please change `WORKING_DIR` variable in the script properly, e.g., `WORKING_DIR = '/home/johndoe/CD_DD'`

`python3 conduct_experiments.py`

## Drawing Plots:

`draw_results.py` script will draw plots (and save them to `results` folder) based on the csv files in `results`.
Note that the code may not output plots properly if all the necessary experiments are not conducted. Sorry for the inconvenience.

`python3 draw_results.py`


