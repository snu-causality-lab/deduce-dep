import itertools
import multiprocessing as mp
import os
from pathlib import Path

from joblib import Parallel, delayed

from cddd.experiments.Complete_PC_stable_experiment import complete_pc_stable_experiment
from cddd.experiments.PC_stable_experiment import pc_stable_experiment
from cddd.experiments.cached_PC_stable_experiment import cached_pc_stable_experiment
from cddd.experiments.cond_experiment import cond_experiment
from cddd.experiments.new_FN_experiment import new_fn_experiment
from cddd.experiments.performance_experiment import performance_experiment

if __name__ == '__main__':
    n_jobs = mp.cpu_count()
    WORKING_DIR = os.path.expanduser('~/CD_DD')
    results_dir = f'{WORKING_DIR}/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Experimental settings for correction experiment
    # nums_vars = (10, 20, 30)
    # edge_ratios = (1.2, 1.5, 2)


    # full (+additional) experimentation (this may not include tests reported during the rebuttal)
    BNs = ['synthetic_ER_10_12', 'synthetic_ER_10_15', 'synthetic_ER_10_20',
            'synthetic_ER_20_24', 'synthetic_ER_20_30', 'synthetic_ER_20_40',
            'synthetic_ER_30_36', 'synthetic_ER_30_45', 'synthetic_ER_30_60']

    # BNs_perf = ['alarm', 'insurance', 'sachs', 'asia', 'child', 'water']

    # BNs = ['sachs']  # ['alarm', 'insurance', 'sachs']
    dataset_sizes = (200, 500, 1000, 2000)
    num_sampling = 30

    Algos = ['HITON-PC', 'PC']
    CITs = ['G2'] * len(BNs)
    Alphas = [0.05, 0.01]

    reliability_criteria = ['no', 'deductive_reasoning']
    Ks = [0, 1, 2]

    Parallel(n_jobs=n_jobs)(
        itertools.chain(
            (delayed(new_fn_experiment)(BN, alpha, K, cit, WORKING_DIR, dataset_size, sample_id)
             for (BN, cit), alpha, K, dataset_size, sample_id
             in itertools.product(list(zip(BNs, CITs)), Alphas, Ks, dataset_sizes, list(range(1, num_sampling + 1)))),

            # (delayed(performance_experiment)(BN, dataset_size, num_sampling, algo, cit, alpha, WORKING_DIR, reliability_criteria, K)
            # for (BN, cit), algo, alpha, K, dataset_size in itertools.product(list(zip(BNs, CITs)), Algos, Alphas, Ks, dataset_sizes)),

            # (delayed(cached_pc_stable_experiment)(Alphas, BN, Ks, ci_tester_name, reliability_criteria, sample_id,
            #                                       size_of_sampled_dataset, WORKING_DIR) for
            #  (BN, ci_tester_name), size_of_sampled_dataset, sample_id in itertools.product(list(zip(BNs, CITs)), dataset_sizes, list(range(1, num_sampling + 1)))),

            # (delayed(complete_pc_stable_experiment)(BN, K, alpha, cit, WORKING_DIR, dataset_size, num_sampling)
            #  for (BN, cit), alpha, K, dataset_size in itertools.product(list(zip(BNs, CITs)), Alphas, Ks, dataset_sizes)),

        )
    )
