import itertools
import multiprocessing as mp
from pathlib import Path

from joblib import Parallel, delayed

from cddd.experiments.FN_experiment import fn_experiment
from cddd.experiments.PC_experiment import pc_experiment
from cddd.experiments.PC_stable_experiment import pc_stable_experiment
from cddd.experiments.Complete_PC_stable_experiment import complete_pc_stable_experiment
from cddd.experiments.cond_experiment import cond_experiment
from cddd.experiments.new_FN_experiment import new_fn_experiment

if __name__ == '__main__':
    ############# just for checking working! #####################
    EXPERIMENTAL_RUN = True

    WORKING_DIR = '/Users/gimjonghwan/Desktop/CD_DD'
    # '/Users/sanghacklee/Dropbox/python_projs/CD_DD'
    results_dir = f'{WORKING_DIR}/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if EXPERIMENTAL_RUN:
        BNs = ['Linear_10_15']
        CITs = ['KCI']

        # BNs = ['ER_10_12', 'alarm']
        # CITs = ['G2', 'G2']
        nums_vars = (10,)
        edge_ratios = (1.2,)
        dataset_sizes = (200,)
        num_sampling = 3

    else:
        # full experimentation
        BNs = ['ER_10_12', 'ER_10_15', 'ER_10_20', 'ER_20_24', 'ER_20_30', 'ER_20_40', 'ER_30_36', 'ER_30_45', 'ER_30_60',
               'alarm', 'insurance', 'sachs', 'asia', 'child', 'water']
        CITs = ['G2'] * len(BNs)
        nums_vars = (10, 20, 30)
        edge_ratios = (1.2, 1.5, 2)
        dataset_sizes = (200, 500, 1000, 2000)
        num_sampling = 30

    # Parallel(n_jobs=mp.cpu_count())(delayed(cond_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    # Parallel(n_jobs=mp.cpu_count())(delayed(pc_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    # Parallel(n_jobs=mp.cpu_count())(delayed(pc_stable_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    # Parallel(n_jobs=mp.cpu_count())(delayed(new_fn_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    Parallel(n_jobs=mp.cpu_count())(
        itertools.chain(
            (delayed(cond_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(pc_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(pc_stable_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(complete_pc_stable_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(new_fn_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs))
        )
    )
    fn_experiment(WORKING_DIR, nums_vars, edge_ratios, dataset_sizes)
