import itertools
import math
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = (
#     r'\usepackage{helvet}'  # helvetica font
#     r'\usepackage{sansmath}'  # math-font matching  helvetica
#     r'\sansmath'  # actually tell tex to use it!
# )


def correction_draw(results_dir, K):
    REPORTING_CONFIDENCE_INTERVAL = True

    df_fn_005 = pd.read_csv(results_dir + f'/new_fn_result_0.05_{K}.csv')
    df_fn_001 = pd.read_csv(results_dir + f'/new_fn_result_0.01_{K}.csv')
    df_fn_001['alpha'] = 0.01
    df_fn_005['alpha'] = 0.05
    df = pd.concat([df_fn_001, df_fn_005], ignore_index=True)

    colors = {
        (False, 0.01): plt.cm.Paired(4),
        (False, 0.05): plt.cm.Paired(5),
        (True, 0.05): plt.cm.Paired(1),
        (True, 0.01): plt.cm.Paired(0)
    }

    crits = {
        'is_deductive_reasoning': [False, True],
        'alpha': [0.01, 0.05]
    }
    # Es = {
    #     10: [12, 15, 20],
    #     20: [24, 30, 40],
    #     30: [36, 45, 60]
    # }

    BNs = ['ER_10_12', 'ER_10_15', 'ER_10_20',
           'ER_20_24', 'ER_20_30', 'ER_20_40',
           'ER_30_36', 'ER_30_45', 'ER_30_60']
    BNs += ['alarm', 'asia', 'child', 'insurance', 'sachs', 'water']

    crit_dicts = [{crit_k: crit_v for crit_k, crit_v in zip(crits.keys(), crit_vs)}
                  for crit_vs in itertools.product(*[crits[k] for k in crits.keys()])]

    for metric in ('F1', 'Precision', 'Recall'):
        # fig = plt.figure(figsize=(6.5, 5.25))
        fig = plt.figure(figsize=(9, 9))
        grid = gridspec.GridSpec(5, 3, figure=fig)
        plt.subplots_adjust(wspace=0, hspace=1)
        x = []
        ax_idx = -1
        for BN in BNs:
            ax_idx += 1
            ax = plt.subplot(grid[ax_idx])
            # select_df = df[(df['num_vars'] == n_V) & (df['num_edges'] == n_E)]
            select_df = df[df['BN'] == BN]
            if not len(select_df):
                print(f'skip {BN}')
                continue
            shift = 0
            # max_y = 0
            for crit_dict in crit_dicts:
                df_crit = select_df
                for crit_k, crit_v in crit_dict.items():
                    df_crit = df_crit[df[crit_k] == crit_v]

                x = np.array([200, 500, 1000, 2000])
                gb_y = df_crit.groupby('data_set_size')[metric].mean()
                gb_counts = df_crit.groupby('data_set_size')[metric].count()
                y = [gb_y[_] if _ in gb_y else 0 for _ in x]  # just in case we didn't experiment.... for some reasons
                gb_yerr = df_crit.groupby('data_set_size')[metric].std()
                if REPORTING_CONFIDENCE_INTERVAL:
                    yerr = [(gb_yerr[_] if _ in gb_yerr else 0) * 1.96 / math.sqrt(gb_counts[_] if _ in gb_counts else 1) for _ in x]
                else:
                    yerr = [gb_yerr[_] if _ in gb_yerr else 0 for _ in x]
                color_key = tuple(crit_dict.values())
                plt.errorbar(x + shift, y, yerr, fmt='-o', color=colors[color_key], ecolor=colors[color_key],
                             capsize=1.5, markersize=1.5, lw=1)
                # shift += 20

            ax.set_xticks(x)
            ax.set_title(BN)
            if ax_idx >= 4:
                ax.set_xlabel('size of dataset')
            if ax_idx == 2:
                custom_lines = [Line2D([0], [0], color=plt.cm.Paired(5), lw=1),
                                Line2D([0], [0], color=plt.cm.Paired(1), lw=1)]

                fig.gca().legend(custom_lines,
                                 ['as-is', 'w/ DD'],
                                 loc=4 if metric in {'F1', 'Precision', 'Recall'} else 2, frameon=False)

            ax.set_ylim(0, 1.05)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            start, end = ax.get_ylim()
            ytick_candidates = np.arange(start, min(end, 1.05), 0.2)
            ax.yaxis.set_ticks(ytick_candidates)

            if ax_idx <= 5:
                ax.set_xlabel(None)
            if ax_idx % 3:
                ax.set_ylabel(None)
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(metric)
            if ax_idx < 6:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(x, rotation=45)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(.8)

        sns.despine(fig)
        fig.tight_layout()
        if REPORTING_CONFIDENCE_INTERVAL:
            fig.savefig(results_dir + f'/new_fn_{metric}_CI95_{K}.pdf', bbox_inches='tight', pad_inches=0.02)
        else:
            fig.savefig(results_dir + f'/new_fn_{metric}_stdev_{K}.pdf', bbox_inches='tight', pad_inches=0.02)


def data_preprocessing(df_algo):
    df_algo['reliability_criterion'] = df_algo['reliability_criterion'].apply(
        lambda x: 'as-is' if x == 'no' else 'with DD')
    df_alarm = df_algo[df_algo['BN'] == 'alarm']
    df_sachs = df_algo[df_algo['BN'] == 'sachs']
    df_insurance = df_algo[df_algo['BN'] == 'insurance']
    return df_alarm, df_insurance, df_sachs


def drop_duplicates(results_dir):
    alphas = [0.01, 0.05]
    Ks = [0, 1, 2]
    for alpha, K in itertools.product(alphas, Ks):
        df = pd.read_csv(results_dir + f'/HITON-PC_result_{alpha}_{K}.csv')
        df = df.drop_duplicates(['BN', 'size_of_sampled_dataset', 'reliability_criterion'], keep='last')
        df.to_csv(results_dir + f'/HITON-PC_result_{alpha}_{K}.csv')

        df = pd.read_csv(results_dir + f'/PC_result_{alpha}_{K}.csv')
        df = df.drop_duplicates(['BN', 'size_of_sampled_dataset', 'reliability_criterion'], keep='last')
        df.to_csv(results_dir + f'/PC_result_{alpha}_{K}.csv')


def create_all_algo(results_dir, K):
    df_ht_005 = pd.read_csv(results_dir + f'/HITON-PC_result_0.05_{K}.csv')
    df_ht_001 = pd.read_csv(results_dir + f'/HITON-PC_result_0.01_{K}.csv')
    df_pc_005 = pd.read_csv(results_dir + f'/PC_result_0.05_{K}.csv')
    df_pc_001 = pd.read_csv(results_dir + f'/PC_result_0.01_{K}.csv')
    dfs = [df_ht_005, df_ht_001, df_pc_005, df_pc_001]

    df_ht_005['algo'] = 'HITON-PC'
    df_ht_001['algo'] = 'HITON-PC'
    df_pc_005['algo'] = 'PC'
    df_pc_001['algo'] = 'PC'

    df_ht_005['alpha'] = 0.05
    df_ht_001['alpha'] = 0.01
    df_pc_005['alpha'] = 0.05
    df_pc_001['alpha'] = 0.01

    # common_columns = set(dfs[0].columns) & set(dfs[1].columns) & set(dfs[2].columns) & set(dfs[3].columns)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(results_dir + f'/all_algo_{K}.csv')


def draw_for_perf_experiments(results_dir, K, repeated=10):
    REPORTING_CONFIDENCE_INTERVAL = True
    if REPORTING_CONFIDENCE_INTERVAL:
        print(f'REPORTING_CONFIDENCE_INTERVAL!! not STD with REPEATED={repeated}!!!!')
    plt.rc('legend', fontsize=5)
    sns.set(style="ticks", rc={"lines.linewidth": 0.8})

    # df = pd.read_csv(WORK_DIR + 'all_algo.csv')
    df = pd.read_csv(results_dir + f'/all_algo_{K}.csv')
    algos = ['HITON-PC', 'PC']
    networks = ['ER_10_12', 'ER_10_15', 'ER_10_20', 'ER_20_24', 'ER_20_30', 'ER_20_40', 'ER_30_36', 'ER_30_45', 'ER_30_60']
    networks += ['alarm', 'sachs', 'insurance', 'asia', 'child', 'water']
    alphas = [0.01, 0.05]
    # criteria = {'algo': algos, 'BN': networks, 'alpha': alphas}
    mapping = {'HITON-PC': 'HITON', 'PC': 'PC', 'alarm': 'Alarm', 'sachs': 'Sachs', 'insurance': 'Insurance',
               'asia': 'Asia', 'child': 'Child', 'water': 'Water',
               'CI_number': '\# CI Tests'}

    colors = {
        ('no', 0.01): plt.cm.Paired(4),
        ('no', 0.05): plt.cm.Paired(5),
        ('deductive_reasoning', 0.05): plt.cm.Paired(1),
        ('deductive_reasoning', 0.01): plt.cm.Paired(0)
    }

    crits = {
        'reliability_criterion': ['no', 'deductive_reasoning'],
        'alpha': [0.01, 0.05]
    }

    crit_dicts = [{crit_k: crit_v for crit_k, crit_v in zip(crits.keys(), crit_vs)}
                  for crit_vs in itertools.product(*[crits[k] for k in crits.keys()])]

    for metric in ('F1', 'Precision', 'Recall', 'Time', 'CI_number'):
        fig = plt.figure(figsize=(12, 15))
        grid = gridspec.GridSpec(6, 5, figure=fig)
        plt.subplots_adjust(wspace=1, hspace=1)
        x = []
        for ax_idx, (network, algo) in enumerate(itertools.product(networks, algos)):
            print(network, algo)
            ax = plt.subplot(grid[ax_idx])
            select_df = df[(df['algo'] == algo) & (df['BN'] == network)]
            shift = 0
            max_y = 0
            for crit_dict in crit_dicts:
                df_crit = select_df
                for crit_k, crit_v in crit_dict.items():
                    df_crit = df_crit[df[crit_k] == crit_v]

                if len(df_crit) == 0:
                    continue

                df_crit = df_crit.sort_values(['size_of_sampled_dataset'])

                x = df_crit['size_of_sampled_dataset']
                y = df_crit[metric]

                max_y = max(max_y, max(y))
                if REPORTING_CONFIDENCE_INTERVAL:
                    yerr = df_crit[metric + '_std'] * 1.96 / math.sqrt(repeated)
                else:
                    yerr = df_crit[metric + '_std']

                color_key = tuple(crit_dict.values())
                plt.errorbar(x + shift, y, yerr, fmt='-o', color=colors[color_key], ecolor=colors[color_key],
                             capsize=1.5, markersize=1.5, lw=1)
                shift += 20

            ax.set_xticks(x)
            ax.set_title(f'{mapping[network] if network in mapping else network} ({mapping[algo]})')
            if ax_idx >= 25:
                ax.set_xlabel('size of dataset')
            if ax_idx == 29:
                custom_lines = [Line2D([0], [0], color=plt.cm.Paired(5), lw=1),
                                Line2D([0], [0], color=plt.cm.Paired(1), lw=1)]

                fig.gca().legend(custom_lines,
                                 ['as-is', 'w/ DD'],
                                 loc=4 if metric in {'F1', 'Precision', 'Recall'} else 2, frameon=False)

            if ax_idx <= 24:
                ax.set_xlabel(None)
            if ax_idx % 5:
                ax.set_ylabel(None)
            else:
                if metric in mapping:
                    ax.set_ylabel(mapping[metric])
                else:
                    ax.set_ylabel(metric)

            if ax_idx < 25:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(x, rotation=45)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(.8)

            if metric not in {'F1', 'Precision', 'Recall'}:
                start, end = ax.get_ylim()
                ax.set_ylim(max(0, start), min(end, max_y * 1.25))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                start, end = ax.get_ylim()
                ytick_candidates = np.arange(start, min(end, 1.0), 0.1)
                if len(ytick_candidates) >= 6:
                    ytick_candidates = np.arange(start, min(end, 1.0), 0.2)
                ax.yaxis.set_ticks(ytick_candidates)

        sns.despine(fig)
        fig.tight_layout()

        if REPORTING_CONFIDENCE_INTERVAL:
            print('REPORTING_CONFIDENCE_INTERVAL!! not STD')
            fig.savefig(results_dir + f'/all_working_{metric}_CI95_{K}.pdf', bbox_inches='tight', pad_inches=0.02)
        else:
            fig.savefig(results_dir + f'/all_working_{metric}_stdev_{K}.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    WORKING_DIR = os.path.expanduser('~/CD_DD')
    results_dir = f'{WORKING_DIR}/results'

    # draw for performance experiment
    # drop_duplicates(results_dir)
    for K in [0, 1, 2]:
        create_all_algo(results_dir, K)
        # TODO previous version used 10 samples, now 30, please change repeated accordingly (for correct confidence intervals)
        draw_for_perf_experiments(results_dir, K, repeated=30)

    # draw for correction experiment
    # for K in [0, 1, 2]:
    #     correction_draw(results_dir, K)
