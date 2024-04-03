from functools import reduce

import pandas as pd


def conditional_sorted(data, X, Y, S, perturb_portion):
    if S:
        rslt_df = data.sort_values(by=S)
        manipulated_data = pd.DataFrame()
        for _, val in rslt_df.groupby(S):
            val = val.sample(frac=1, random_state=42)
            num_of_perturbed = round(perturb_portion * val.shape[0])

            new_col_X_sort = list(val.iloc[num_of_perturbed:, X].sort_values())
            val.iloc[num_of_perturbed:, X] = new_col_X_sort
            new_col_Y_sort = list(val.iloc[num_of_perturbed:, Y].sort_values())
            val.iloc[num_of_perturbed:, Y] = new_col_Y_sort
            manipulated_data = pd.concat([manipulated_data, val], ignore_index=True)
    else:
        manipulated_data = data.sample(frac=1, random_state=42)
        num_of_perturbed = round(perturb_portion * manipulated_data.shape[0])

        new_col_X_sort = list(manipulated_data.iloc[num_of_perturbed:, X].sort_values())
        manipulated_data.iloc[num_of_perturbed:, X] = new_col_X_sort
        new_col_Y_sort = list(manipulated_data.iloc[num_of_perturbed:, Y].sort_values())
        manipulated_data.iloc[num_of_perturbed:, Y] = new_col_Y_sort

    return manipulated_data


def get_num_of_parameters(data, target, x, z):
    num_of_domain_target = len(data[target].unique())
    num_of_domain_x = len(data[x].unique())
    if z:
        levels_of_domain_z = list(map(lambda x_: len(data[x_].unique()), z))
        num_of_domain_z = reduce(lambda x_, y_: x_ * y_, levels_of_domain_z)
    else:
        num_of_domain_z = 1

    num_of_parameters = num_of_domain_x * num_of_domain_target * num_of_domain_z
    return num_of_parameters
