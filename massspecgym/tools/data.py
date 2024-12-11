import random
from tqdm.notebook import tqdm
import pandas as pd
import json
import time
import os
from matchms import Spectrum
from rdkit import Chem
import typing as T

from massspecgym.tools.murcko_hist import are_sub_hists, murcko_hist

def compute_murcko_histograms(df):
    """
    Compute Murcko histograms for each unique SMILES in the DataFrame.

    Parameters:
    - df: pandas DataFrame with at least a 'smiles' column.

    Returns:
    - df_us: DataFrame of unique SMILES with their Murcko histograms.
    """
    if 'smiles' not in df.columns:
        raise ValueError("SMILES column is missing in DataFrame.")

    df_us = df.drop_duplicates(subset=['smiles']).copy()

    print("Computing Murcko histograms...")
    tqdm.pandas()
    df_us['MurckoHist'] = df_us['smiles'].progress_apply(
        lambda x: murcko_hist(Chem.MolFromSmiles(x))
    )
    df_us['MurckoHistStr'] = df_us['MurckoHist'].astype(str)
    return df_us

def group_by_murcko_histograms(df_us):
    """
    Group molecules by their Murcko histograms.

    Parameters:
    - df_us: DataFrame of unique SMILES with their Murcko histograms.

    Returns:
    - df_gb: DataFrame grouped by MurckoHistStr.
    """
    df_gb = df_us.groupby('MurckoHistStr').agg(
        count=('smiles', 'count'),
        smiles_list=('smiles', list)
    ).reset_index()

    df_gb['MurckoHist'] = df_gb['MurckoHistStr'].apply(eval)

    df_gb = df_gb.sort_values('count', ascending=False).reset_index(drop=True)
    return df_gb

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

def canonicalize_smiles_in_spectra(
    spectra_list: T.List[Spectrum],
    standardize_smiles_func: T.Callable[[T.Union[str, T.List[str]]], T.Union[str, T.List[str]]],
    save_path: str,
    batch_size: int = 500,
    max_retries: int = 10,
    delay_between_retries: int = 60  # in seconds
) -> T.Dict[str, str]:
    """
    Canonicalize SMILES in a list of spectrum objects.

    Args:
        spectra_list: List of Spectrum objects.
        standardize_smiles_func: Function to standardize SMILES.
        save_path: Path to save the progress dictionary.
        batch_size: Number of SMILES to process before saving progress.
        max_retries: Maximum number of retries if the database is not reachable.
        delay_between_retries: Delay between retries in seconds.

    Returns:
        Dictionary mapping original SMILES to canonical SMILES.
    """
    # Extract all unique SMILES from spectra
    all_smiles = set()
    for spectrum in spectra_list:
        smiles = spectrum.metadata.get('smiles')
        if smiles:
            all_smiles.add(smiles)

    print(f"Total unique SMILES to process: {len(all_smiles)}")

    # Check if there is a saved progress file
    if os.path.exists(save_path):
        print(f"Resuming from saved progress at {save_path}")
        with open(save_path, 'r') as f:
            smiles_dict = json.load(f)
    else:
        smiles_dict = {}

    processed_smiles = set(smiles_dict.keys())
    remaining_smiles = all_smiles - processed_smiles
    print(f"SMILES remaining to process: {len(remaining_smiles)}")

    smiles_list = list(remaining_smiles)
    total_smiles = len(smiles_list)

    # For saving progress
    batch_counter = 0
    for idx, original_smiles in enumerate(smiles_list, 1):
        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
                # Canonicalize the SMILES
                canonical_smiles = standardize_smiles_func(original_smiles)
                if not canonical_smiles:
                    print(f"Warning: Invalid SMILES '{original_smiles}'. Skipping.")
                    canonical_smiles = None
                else:
                    # If the function returns a list, extract the first element
                    if isinstance(canonical_smiles, list):
                        canonical_smiles = canonical_smiles[0]
                smiles_dict[original_smiles] = canonical_smiles
                success = True
            except Exception as e:
                retries += 1
                print(f"Error processing SMILES '{original_smiles}': {e}")
                if retries < max_retries:
                    print(f"Retrying in {delay_between_retries} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(delay_between_retries)
                else:
                    print(f"Max retries reached for SMILES '{original_smiles}'. Skipping.")
                    smiles_dict[original_smiles] = None  # Mark as failed

        batch_counter += 1
        # Save progress every batch_size SMILES or at the end
        if batch_counter >= batch_size or idx == total_smiles:
            print(f"Processing progress: {idx}/{total_smiles} SMILES")
            with open(save_path, 'w') as f:
                json.dump(smiles_dict, f)
            print(f"Progress saved to {save_path}")
            batch_counter = 0

    print("Canonicalization completed.")
    return smiles_dict

def handle_problematic_smiles(problematic_smiles, standardize_smiles_func):
    """
    Takes a list of problematic SMILES strings, validates and canonicalizes them using RDKit,
    then sends the RDKit-canonicalized SMILES to PubChem for standardization.
    Returns a dictionary mapping the original problematic SMILES to the PubChem-canonicalized SMILES.
    Args:
        problematic_smiles: List of SMILES strings that could not be canonicalized initially.
        standardize_smiles_func: Function to standardize SMILES.

    Returns:
        Dictionary mapping original problematic SMILES to PubChem-canonicalized SMILES.
    """

    unique_problematic_smiles = set(problematic_smiles)
    mapping_dict = {}

    for smi in unique_problematic_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rdkit_canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            try:
                pubchem_canonical_smi = standardize_smiles_func(rdkit_canonical_smi)
                if isinstance(pubchem_canonical_smi, list):
                    pubchem_canonical_smi = pubchem_canonical_smi[0]
                mapping_dict[smi] = pubchem_canonical_smi
                print(f"Canonicalized '{smi}' to '{pubchem_canonical_smi}'")
            except Exception as e:
                print(f"Error standardizing SMILES '{smi}' after RDKit canonicalization: {e}")
                mapping_dict[smi] = None
        else:
            print(f"Cannot canonicalize invalid SMILES '{smi}'.")
            mapping_dict[smi] = None

    return mapping_dict
