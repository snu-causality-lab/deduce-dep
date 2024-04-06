import itertools
import multiprocessing as mp
from pathlib import Path

from joblib import Parallel, delayed

from cddd.experiments.Complete_PC_stable_experiment import complete_pc_stable_experiment
from cddd.experiments.FN_experiment import fn_experiment
from cddd.experiments.PC_stable_experiment import pc_stable_experiment
from cddd.experiments.cached_PC_stable_experiment import cached_pc_stable_experiment
from cddd.experiments.cond_experiment import cond_experiment
from cddd.experiments.new_FN_experiment import new_fn_experiment

if __name__ == '__main__':
    ############# just for checking working! #####################
    EXPERIMENTAL_RUN = True

    n_jobs = mp.cpu_count()
    WORKING_DIR = '/home/johndoe/CD_DD'
    results_dir = f'{WORKING_DIR}/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if EXPERIMENTAL_RUN:
        BNs = ['Linear_20_36', 'Linear_20_64', 'Linear_20_36_sf', 'Linear_20_64_sf']
        # we must specify type of CITs,
        # `ParCorr` for partial correlation,
        # `G2` for discrete data,
        # `KCI` for kernel conditional independence test (using RBF kernel with median heuristic).
        CITs = ['ParCorr'] * len(BNs)

        Ks = [0, 1]
        Alphas = [0.05, 0.01]
        dataset_sizes = (100,)
        num_sampling = 3
        reliability_criteria = ['no', 'deductive_reasoning']

    else:
        # full (+additional) experimentation (this may not include tests reported during the rebuttal)
        BNs = ['ER_10_12', 'ER_10_15', 'ER_10_20', 'ER_20_24', 'ER_20_30', 'ER_20_40', 'ER_30_36', 'ERcd_30_45', 'ER_30_60',
               'alarm', 'insurance', 'sachs', 'asia', 'child', 'water']
        CITs = ['G2'] * len(BNs)
        nums_vars = (10, 20, 30)
        edge_ratios = (1.2, 1.5, 2)
        dataset_sizes = (200, 500, 1000, 2000)
        num_sampling = 30

    Parallel(n_jobs=n_jobs)(
        itertools.chain(
            (delayed(cond_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(pc_stable_experiment)(BN, cit, WORKING_DIR, size_of_sampled_dataset, num_sampling, K, Alpha) for (BN, cit), K, Alpha, size_of_sampled_dataset in itertools.product(list(zip(BNs, CITs)), Ks, Alphas, dataset_sizes)),
            (delayed(complete_pc_stable_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs)),
            (delayed(new_fn_experiment)(BN, cit, WORKING_DIR, dataset_sizes, num_sampling) for BN, cit in zip(BNs, CITs))
        )
    )
    cached_pc_stable_experiment(WORKING_DIR, BNs, CITs, Ks, Alphas, dataset_sizes, num_sampling, reliability_criteria, n_jobs=n_jobs)
    fn_experiment(WORKING_DIR, nums_vars, edge_ratios, dataset_sizes)
