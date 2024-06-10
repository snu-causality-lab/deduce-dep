import itertools
import multiprocessing as mp
import os
from pathlib import Path

from joblib import Parallel, delayed

from cddd.experiments.Complete_PC_stable_experiment import complete_pc_stable_experiment
from cddd.experiments.cached_PC_stable_experiment import cached_pc_stable_experiment
from cddd.experiments.new_FN_experiment import new_fn_experiment
from cddd.experiments.FN_experiment import fn_experiment
from cddd.experiments.performance_experiment import performance_experiment

if __name__ == '__main__':
    n_jobs = mp.cpu_count()
    WORKING_DIR = os.path.expanduser('~/CD_DD')
    results_dir = f'{WORKING_DIR}/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Experimental settings for correction experiment
    nums_vars = (10, 20, 30)
    times_vars = (1.2, 1.5, 2)
    dataset_sizes_for_corr = (200, 500, 1000)
    num_sampling_for_corr = 50

    # Experimental settings for performance experiment
    BNs = ['alarm', 'insurance', 'sachs', 'water', 'asia', 'child']
    Algos = ['HITON-PC', 'PC']
    CITs = ['G2'] * len(BNs)
    dataset_sizes_for_perf = (200, 500, 1000, 2000)
    num_sampling_for_perf = 30
    reliability_criteria = ['no', 'deductive_reasoning']
    Ks = [1,] #[0, 1, 2]

    # Common setting for both experiments
    Alphas = [0.05, 0.01]


    Parallel(n_jobs=n_jobs)(
        itertools.chain(
            # (delayed(fn_experiment)(WORKING_DIR, num_vars, time_vars, sampling_number, alpha, dataset_size)
            #  for num_vars, time_vars, alpha, dataset_size, sampling_number
            #  in itertools.product(nums_vars, times_vars, Alphas, dataset_sizes_for_corr, list(range(num_sampling_for_corr)))),

            # (delayed(new_fn_experiment)(BN, alpha, K, cit, WORKING_DIR, dataset_size, sample_id)
            #  for (BN, cit), alpha, K, dataset_size, sample_id
            #  in itertools.product(list(zip(BNs, CITs)), Alphas, Ks, dataset_sizes_for_corr, list(range(1, num_sampling_for_corr + 1)))),

            # (delayed(performance_experiment)(BN, dataset_size, num_sampling_for_perf, algo, cit, alpha, WORKING_DIR, reliability_criteria, K)
            # for (BN, cit), algo, alpha, K, dataset_size in itertools.product(list(zip(BNs, CITs)), Algos, Alphas, Ks, dataset_sizes_for_perf)),

            (delayed(complete_pc_stable_experiment)(BN, K, alpha, cit, WORKING_DIR, dataset_size, num_sampling_for_perf)
            for (BN, cit), alpha, K, dataset_size in itertools.product(list(zip(BNs, CITs)), Alphas, Ks, dataset_sizes_for_perf)),
        )
    )
