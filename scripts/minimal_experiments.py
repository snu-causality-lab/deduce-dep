from pathlib import Path

from cddd.experiments.PC_stable_experiment import pc_stable_experiment

if __name__ == '__main__':
    WORKING_DIR = '/home/johndoe/CD_DD'
    results_dir = f'{WORKING_DIR}/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # about ~30 seconds,
    # alarm network with size 200
    # with CIT based on g-squared test
    # K = 0, alpha = 0.05
    # averaged over 10 repetitions
    pc_stable_experiment('alarm', 'G2', WORKING_DIR, 200, 10, 0, 0.05)
    # please check
    # ~/CD_DD/results/pc_stable_result_0.05_0.csv
