import random
from massspecgym.tools.murcko_hist import are_sub_hists


def split_by_murcko_histograms(df_us, df_gb, val_fraction=0.10, test_fraction=0.10, k=3, d=4, seed=None):
    """
    Split the dataset into training, validation, and test sets based on Murcko histograms.
    If seed is provided, performs random splitting; otherwise, uses deterministic splitting.

    Parameters:
    - df_us: DataFrame of unique SMILES with Murcko histograms.
    - df_gb: DataFrame grouped by MurckoHistStr.
    - val_fraction: Fraction of molecules to include in the validation set (from the training set).
    - test_fraction: Fraction of molecules to include in the test set (from the entire dataset).
    - k: Parameter for are_sub_hists function.
    - d: Parameter for are_sub_hists function.
    - seed: Random seed for reproducibility. If None, deterministic splitting is used.

    Returns:
    - smiles_to_fold: Dictionary mapping SMILES to 'train', 'val', or 'test'.
    """
    if seed is not None:
        random.seed(seed)
    median_i = len(df_gb) // 2
    total_mols = len(df_us)
    cum_test_mols = 0
    test_mols_frac = test_fraction
    test_idx, train_idx = [], []

    if seed is not None:
        indices = list(range(0, median_i + 1))
        random.shuffle(indices)
    else:
        indices = range(median_i, -1, -1)

    for i in indices:
        current_hist = df_gb.iloc[i]['MurckoHist']

        is_test_subhist = any(
            are_sub_hists(current_hist, df_gb.iloc[j]['MurckoHist'], k=k, d=d)
            for j in test_idx
        )

        if is_test_subhist:
            train_idx.append(i)
        else:
            test_fraction_current = cum_test_mols / total_mols
            if test_fraction_current <= test_mols_frac:
                cum_test_mols += df_gb.iloc[i]['count']
                test_idx.append(i)
            else:
                train_idx.append(i)

    if seed is not None:
        remaining_indices = [i for i in range(median_i + 1, len(df_gb))]
        random.shuffle(remaining_indices)
        train_idx.extend(remaining_indices)
    else:
        train_idx.extend(range(median_i + 1, len(df_gb)))

    assert len(train_idx) + len(test_idx) == len(df_gb)

    smiles_to_fold = {}
    for i, row in df_gb.iterrows():
        if i in test_idx:
            fold = 'test'
        else:
            fold = 'train'
        for smiles in row['smiles_list']:
            smiles_to_fold[smiles] = fold

    if val_fraction > 0:

        train_smiles = [smiles for smiles, fold in smiles_to_fold.items() if fold == 'train']
        df_us_train = df_us[df_us['smiles'].isin(train_smiles)].copy()

        df_gb_train = df_us_train.groupby('MurckoHistStr').agg(
            count=('smiles', 'count'),
            smiles_list=('smiles', list)
        ).reset_index()
        df_gb_train['MurckoHist'] = df_gb_train['MurckoHistStr'].apply(eval)
        df_gb_train = df_gb_train.sort_values('count', ascending=False).reset_index(drop=True)

        median_i_train = len(df_gb_train) // 2
        total_train_mols = len(df_us_train)
        cum_val_mols = 0
        val_mols_frac = val_fraction
        val_idx_train, train_idx_train = [], []

        if seed is not None:
            indices_train = list(range(0, median_i_train + 1))
            random.shuffle(indices_train)
        else:
            indices_train = range(median_i_train, -1, -1)

        for i in indices_train:
            current_hist = df_gb_train.iloc[i]['MurckoHist']

            is_val_subhist = any(
                are_sub_hists(current_hist, df_gb_train.iloc[j]['MurckoHist'], k=k, d=d)
                for j in val_idx_train
            )

            if is_val_subhist:
                train_idx_train.append(i)
            else:
                val_fraction_current = cum_val_mols / total_train_mols
                if val_fraction_current <= val_mols_frac:
                    cum_val_mols += df_gb_train.iloc[i]['count']
                    val_idx_train.append(i)
                else:
                    train_idx_train.append(i)

        if seed is not None:
            remaining_indices_train = [i for i in range(median_i_train + 1, len(df_gb_train))]
            random.shuffle(remaining_indices_train)
            train_idx_train.extend(remaining_indices_train)
        else:
            train_idx_train.extend(range(median_i_train + 1, len(df_gb_train)))

        assert len(train_idx_train) + len(val_idx_train) == len(df_gb_train)

        for i, row in df_gb_train.iterrows():
            if i in val_idx_train:
                fold = 'val'
            else:
                fold = 'train'
            for smiles in row['smiles_list']:
                smiles_to_fold[smiles] = fold
    return smiles_to_fold