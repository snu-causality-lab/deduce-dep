import itertools
import multiprocessing as mp
from pathlib import Path

from joblib import Parallel, delayed

from cddd.experiments.PC_stable_experiment import pc_stable_experiment

if __name__ == '__main__':
    ############# just for checking working! #####################
    EXPERIMENTAL_RUN = True

    WORKING_DIR = '/home/jonghwan/CDDD'
    # WORKING_DIR = '/Users/gimjonghwan/Desktop/CD_DD'
    # WORKING_DIR = '/Users/sanghacklee/Dropbox/python_projs/CD_DD'
    results_dir = f'{WORKING_DIR}/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if EXPERIMENTAL_RUN:
        BNs = ['Linear_20_36', 'Linear_20_64', 'Linear_20_36_sf', 'Linear_20_64_sf']
        # BNs = ['Linear_20_36']
        CITs = ['ParCorr'] * len(BNs)

        # BNs = ['alarm', 'sachs', 'insurance']
        # CITs = ['G2'] * 3
        # nums_vars = (20,)
        # edge_ratios = (1.2, 1.5, 2)
        # dataset_sizes = (200, 500, 1000)
        Ks = [0, 1, 2]
        Alphas = [0.05, 0.01]
        dataset_sizes = (30, 50, 100, 150)
        num_sampling = 10

    else:
        # full experimentation
        BNs = ['ER_10_12', 'ER_10_15', 'ER_10_20', 'ER_20_24', 'ER_20_30', 'ER_20_40', 'ER_30_36', 'ERcd_30_45', 'ER_30_60',
               'alarm', 'insurance', 'sachs', 'asia', 'child', 'water']
        CITs = ['G2'] * len(BNs)
        nums_vars = (10, 20, 30)
        edge_ratios = (1.2, 1.5, 2)
        dataset_sizes = (200, 500, 1000, 2000)
        num_sampling = 30

    # n_jobs = 1
    n_jobs = mp.cpu_count()
    # Parallel(n_jobs=mp.cpu_count())(delayed(cond_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    # Parallel(n_jobs=mp.cpu_count())(delayed(pc_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    # Parallel(n_jobs=mp.cpu_count())(delayed(pc_stable_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    # Parallel(n_jobs=mp.cpu_count())(delayed(new_fn_experiment)(BN, WORKING_DIR, dataset_sizes, num_sampling) for BN in BNs)
    Parallel(n_jobs=n_jobs)(
        itertools.chain(
            # (delayed(cond_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            # (delayed(pc_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(pc_stable_experiment)(BN, cit, WORKING_DIR, size_of_sampled_dataset, num_sampling, K, Alpha) for (BN, cit), K, Alpha, size_of_sampled_dataset in itertools.product(list(zip(BNs, CITs)), Ks, Alphas, dataset_sizes)),
            # (delayed(complete_pc_stable_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            # (delayed(new_fn_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs))
        )
    )
    # fn_experiment(WORKING_DIR, nums_vars, edge_ratios, dataset_sizes)
