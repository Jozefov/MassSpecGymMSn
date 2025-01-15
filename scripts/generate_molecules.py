#!/usr/bin/env python3

import os
import logging
import random
import typing as T
import json
import multiprocessing
import ctypes
from multiprocessing import Pool, cpu_count, Array
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from matchms.importing import load_from_mgf
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcExactMolWt

# If you have your own local `massspecgym.utils`, you can import it:
# import massspecgym.utils as utils

# Set a random seed to ensure reproducibility
random.seed(0)

# Enable tqdm in Pandas
tqdm.pandas()

# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def dedup(seq):
    """Deduplicate list while preserving order (https://stackoverflow.com/a/480227)"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def init(shared_smiles, shared_formula, shared_mass, shared_inchi_key_2D):
    """
    Initializer function for the worker processes.
    We store references to shared arrays in global variables.
    """
    global df_smiles, df_formula, df_mass, df_inchi_key_2d
    df_smiles = shared_smiles
    df_formula = shared_formula
    df_mass = shared_mass
    df_inchi_key_2d = shared_inchi_key_2D

def get_candidates_single(args):
    """
    Worker function that takes a tuple of (index, smiles, cand_type, max_cands)
    and returns a list of candidate SMILES.
    """
    index, smiles, cand_type, max_cands = args
    candidates_single = []

    # Add the query SMILES as one of the candidates first
    if not candidates_single:
        candidates_single.append(smiles)

    # If we've already reached max_cands, just return
    if len(candidates_single) == max_cands:
        return candidates_single

    # Build an RDKit Mol and calculate the InChI key 2D component
    mol = Chem.MolFromSmiles(smiles)
    inchi_key_2D = Chem.MolToInchiKey(mol).split("-")[0]

    # Depending on candidate type, filter from the global shared arrays
    if cand_type == 'formula':
        formula = CalcMolFormula(mol)
        new_cands = [
            df_smiles[i] for i in range(len(df_smiles))
            if df_formula[i] == formula and df_inchi_key_2d[i] != inchi_key_2D
        ]
    elif cand_type == 'mass':
        mass = CalcExactMolWt(mol)
        mass_eps = mass * 1e-6 * 10  # 10 ppm
        new_cands = [
            df_smiles[i] for i in range(len(df_smiles))
            if abs(df_mass[i] - mass) < mass_eps and df_inchi_key_2d[i] != inchi_key_2D
        ]
    else:
        raise ValueError(f'Unknown candidates type {cand_type}.')

    # Shuffle new candidates (so if we prune, it's somewhat random)
    random.shuffle(new_cands)

    # Deduplicate and prune to max_cands
    candidates_single.extend(new_cands)
    candidates_single = dedup(candidates_single)[:max_cands]

    return candidates_single

def get_candidates_parallel(
    query_smiles: T.Iterable[str],
    df: pd.DataFrame,
    cand_type: str = 'formula',
    max_cands: int = 256,
    df_candidates: T.Optional[pd.DataFrame] = None,
    max_workers: int = cpu_count() - 2
) -> pd.DataFrame:
    """
    For each SMILES in `query_smiles`, find similar SMILES in the dataframe `df`
    in parallel, either by formula or mass.
    """
    logging.info('Starting get_candidates_parallel')

    # If no DataFrame (df_candidates) was provided to fill, create a new one
    if df_candidates is None:
        df_candidates = pd.DataFrame({'smiles': query_smiles, 'cands': [[] for _ in query_smiles]})
        logging.info('Initialized df_candidates with empty candidate lists')

    # Convert df columns to numpy arrays
    df_smiles_array = df['smiles'].to_numpy()
    df_formula_array = df['formula'].fillna('').to_numpy()
    df_mass_array = df['mass'].to_numpy()
    df_inchi_key_2d_array = df['inchi_key_2D'].fillna('').to_numpy()

    # Create shared arrays (so we don't copy large df for each process)
    shared_smiles = Array(ctypes.c_wchar_p, df_smiles_array, lock=False)
    shared_formula = Array(ctypes.c_wchar_p, df_formula_array, lock=False)
    shared_mass = Array(ctypes.c_double, df_mass_array, lock=False)
    shared_inchi_key_2D = Array(ctypes.c_wchar_p, df_inchi_key_2d_array, lock=False)

    logging.info(f'Setting up shared arrays with {max_workers} workers')

    with Pool(
        processes=max_workers,
        initializer=init,
        initargs=(shared_smiles, shared_formula, shared_mass, shared_inchi_key_2D)
    ) as pool:
        # Prepare arguments for each SMILES
        args = [
            (index, row['smiles'], cand_type, max_cands)
            for index, row in df_candidates.iterrows()
        ]

        results = []
        for result in tqdm(pool.imap(get_candidates_single, args), total=len(args)):
            results.append(result)

        # Store results in the 'cands' column
        df_candidates['cands'] = results

    logging.info('Finished multiprocessing pool')
    logging.info('Completed get_candidates_parallel')
    return df_candidates

def main():
    # This is the main function that does all the data loading, processing, and saving.

    # For demonstration, you can set your own paths here
    df_1M_path = '/Users/macbook/CODE/Majer:MassSpecGym/data/candidates_generation/MassSpecGym_retrieval_molecules_1M.tsv'
    df_4M_path = '/Users/macbook/CODE/Majer:MassSpecGym/data/candidates_generation/MassSpecGym_retrieval_molecules_4M.tsv'
    df_pubchem_path = '/Users/macbook/CODE/Majer:MassSpecGym/data/candidates_generation/MassSpecGym_retrieval_molecules_pubchem_118M.tsv'
    mgf_path = '/Users/macbook/CODE/Majer:MassSpecGym/data/MSn/20241211_msn_library_pos_all_lib_MSn.mgf'

    # Read the data
    df_1M = pd.read_csv(df_1M_path, sep='\t')
    df_4M = pd.read_csv(df_4M_path, sep='\t')
    df_pubchem = pd.read_csv(df_pubchem_path, sep='\t')

    # Rename columns to unify them
    df_1M = df_1M.rename(columns={'weight': 'mass'})
    df_4M = df_4M.rename(columns={'weight': 'mass'})

    # Load the MGF spectra
    spectra = list(load_from_mgf(mgf_path))

    # Extract SMILES from the spectra
    smiles_series = pd.Series([s.metadata['smiles'] for s in spectra]).drop_duplicates()
    logging.info(f"Number of MSn spectra: {len(spectra)}; number of unique SMILES: {len(smiles_series)}")

    # Run candidate retrieval in stages

    # 1) First with the 1M dataset
    candidates = get_candidates_parallel(
        query_smiles=smiles_series,
        df=df_1M,
        cand_type='mass',
        max_cands=256,
        max_workers=4
    )

    # Quick histogram
    plt.figure()
    candidates['cands'].apply(len).hist(bins=100)
    plt.title('Candidates from 1M dataset')
    plt.savefig('hist_1M_candidates.png')
    plt.close()

    # 2) Enrich with the 4M dataset
    candidates = get_candidates_parallel(
        query_smiles=smiles_series,
        df=df_4M,
        cand_type='mass',
        df_candidates=candidates,
        max_cands=256,
        max_workers=4
    )

    plt.figure()
    candidates['cands'].apply(len).hist(bins=100)
    plt.title('Candidates after adding 4M dataset')
    plt.savefig('hist_4M_candidates.png')
    plt.close()

    # Save intermediate results
    candidates.to_pickle('MSn_cands_4M_mass.pkl')

    # 3) Enrich with the PubChem dataset
    candidates = get_candidates_parallel(
        query_smiles=smiles_series,
        df=df_pubchem,
        cand_type='mass',
        df_candidates=candidates,
        max_cands=256,
        max_workers=4
    )

    plt.figure()
    candidates['cands'].apply(len).hist(bins=100)
    plt.title('Candidates after adding PubChem dataset')
    plt.savefig('hist_pubchem_candidates.png')
    plt.close()

    # Finally, save the combined results
    candidates.to_pickle('MSn_cands_pubchem_118M_mass.pkl')

    # Save a JSON version if desired
    with open('MassSpecGymMSn_retrieval_candidates_mass.json', 'w') as json_file:
        candidates_dict = dict(zip(candidates['smiles'], candidates['cands']))
        json.dump(candidates_dict, json_file)

    logging.info("All done. Results saved to pickle and JSON!")

if __name__ == "__main__":
    # On macOS Apple Silicon, you might consider explicitly setting the start method:
    # multiprocessing.set_start_method("spawn")

    main()