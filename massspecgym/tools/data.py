from tqdm.notebook import tqdm
import pandas as pd
import json
import time
import os
from collections import deque, defaultdict
from math import comb
import numpy as np

import random
from matchms import Spectrum
from rdkit import Chem
import typing as T
from typing import List, Dict, Tuple, Optional
from massspecgym import utils

from massspecgym.tools.murcko_hist import are_sub_hists, murcko_hist
from massspecgym.tools.metrics import dreams_embedding_similarity, compute_cosine_greedy_score, compute_cosine_hungarian_score

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


def compute_root_mol_freq(metadata: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Takes a dataframe 'metadata' and the name of a column containing an INCHI-like string.
    It will extract the first 14 characters from that column to form an 'inchi_key' column.
    """
    # Ensure the column exists
    if col_name not in metadata.columns:
        print("Warning: Column '{}' not found in metadata. Computing from smiles.".format(col_name))
        metadata["inchi_key"] = metadata["smiles"].apply(utils.smiles_to_inchi_key)
    else:
        # Extract the first 14 characters from the given column to form inchikey_aux
        metadata["inchi_key"] = metadata[col_name].apply(lambda x: x[:14] if isinstance(x, str) else "")

    # Filter root rows (ms_level=2)
    root_rows = metadata[metadata["ms_level"] == str(2)]

    # Check for multiple root rows with same identifier
    if "identifier" in root_rows.columns:
        dup_identifiers = root_rows[root_rows.duplicated("identifier", keep=False)]
        if not dup_identifiers.empty:
            print("Warning: Multiple rows with the same identifier at ms_level=2 detected.")
            print(dup_identifiers[["identifier"]])

    # Compute mol_freq from root rows only
    mol_freq_map = root_rows.groupby("inchi_key").size().to_dict()

    # Map this mol_freq back to all rows by inchikey_aux
    metadata["mol_freq"] = metadata["inchi_key"].map(mol_freq_map).fillna(0).astype(float)

    return metadata


def get_spectrum(node) -> Optional[Spectrum]:
    """Return the matchms.Spectrum from a node."""
    if node is None or node.spectrum is None:
        return None
    return node.spectrum

def get_embedding_for_node(node, embeddings_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Return embedding from a dictionary that maps identifier->embedding."""
    if node is None or node.spectrum is None:
        return None
    identifier = node.spectrum.get('identifier', None)
    if identifier is None:
        return None
    return embeddings_dict.get(identifier, None)

def get_ms_level(node) -> Optional[int]:
    """Return the ms_level from node's spectrum metadata if present."""
    if node is None or node.spectrum is None:
        return None
    ms_level_str = node.spectrum.get('ms_level', None)
    if ms_level_str is not None:
        return int(ms_level_str)
    return None

def compute_ancestor_descendant_similarity(tree,
                                           use_embedding: bool=False,
                                           sim_fn=None,
                                           embeddings_dict: Optional[Dict[str, np.ndarray]]=None,
                                           tolerance: float=0.1) -> List[float]:
    """
    For each parent->child in the tree, compute similarity.
    If use_embedding=False, sim_fn is (specA,specB)->(score,matched).
    If None => default CosineGreedy(tol=0.1).
    If use_embedding=True, sim_fn is (embA,embB)->float or None => dreams_embedding_similarity.
    """

    scores = []
    queue = deque([tree.root])
    while queue:
        parent = queue.popleft()
        for child in parent.children.values():
            if not use_embedding:
                specA = get_spectrum(parent)
                specB = get_spectrum(child)
                if sim_fn is None:
                    (sc, _) = compute_cosine_greedy_score(specA, specB, tolerance=tolerance)
                else:
                    (sc, _) = sim_fn(specA, specB)
                scores.append(sc)
            else:
                embA = get_embedding_for_node(parent, embeddings_dict)
                embB = get_embedding_for_node(child, embeddings_dict)
                if sim_fn is None:
                    sc = dreams_embedding_similarity(embA, embB)
                else:
                    sc = sim_fn(embA, embB)
                scores.append(sc)
            queue.append(child)
    return scores

def compute_sibling_similarity(tree,
                               use_embedding: bool=False,
                               sim_fn=None,
                               embeddings_dict: Optional[Dict[str, np.ndarray]]=None,
                               tolerance: float=0.1) -> List[float]:
    """
    For each parent, compute pairwise similarity among sibling nodes.
    Return a list of the resulting scores.
    """

    scores = []
    queue = deque([tree.root])
    while queue:
        parent = queue.popleft()
        siblings = list(parent.children.values())
        for i in range(len(siblings)):
            for j in range(i+1, len(siblings)):
                nodeA = siblings[i]
                nodeB = siblings[j]
                if not use_embedding:
                    specA = get_spectrum(nodeA)
                    specB = get_spectrum(nodeB)
                    if sim_fn is None:
                        (sc, _) = compute_cosine_greedy_score(specA, specB, tolerance=tolerance)
                    else:
                        (sc, _) = sim_fn(specA, specB)
                    scores.append(sc)
                else:
                    embA = get_embedding_for_node(nodeA, embeddings_dict)
                    embB = get_embedding_for_node(nodeB, embeddings_dict)
                    if sim_fn is None:
                        sc = dreams_embedding_similarity(embA, embB)
                    else:
                        sc = sim_fn(embA, embB)
                    scores.append(sc)

        for child in siblings:
            queue.append(child)

    return scores

def random_node_pairs(
    msn_dataset,
    num_pairs: int=1000,
    use_embedding: bool=False,
    sim_fn=None,
    embeddings_dict: Optional[Dict[str, np.ndarray]]=None,
    tolerance: float=0.1
) -> List[float]:
    """
    Gather all nodes from all trees, sample random pairs, compute similarity.
    Return list of scores.
    """
    from collections import deque
    all_nodes = []
    for tree in msn_dataset.trees:
        queue = deque([tree.root])
        while queue:
            n = queue.popleft()
            all_nodes.append(n)
            for c in n.children.values():
                queue.append(c)

    if len(all_nodes) < 2:
        return []

    scores = []
    for _ in range(num_pairs):
        A = random.choice(all_nodes)
        B = random.choice(all_nodes)
        if A is B:
            continue

        if not use_embedding:
            specA = get_spectrum(A)
            specB = get_spectrum(B)
            if sim_fn is None:
                (sc, _) = compute_cosine_greedy_score(specA, specB, tolerance=tolerance)
            else:
                (sc, _) = sim_fn(specA, specB)
            scores.append(sc)
        else:
            embA = get_embedding_for_node(A, embeddings_dict)
            embB = get_embedding_for_node(B, embeddings_dict)
            if sim_fn is None:
                sc = dreams_embedding_similarity(embA, embB)
            else:
                sc = sim_fn(embA, embB)
            scores.append(sc)

    return scores

def compute_pairwise_similarity_by_mslevel(
    tree,
    use_embedding: bool = False,
    sim_fn = None,
    embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
    tolerance: float = 0.1,
    descendant_mode: bool = False
) -> Dict[Tuple[int,int], List[float]]:
    """
    For each pair of distinct nodes in the tree, compute similarity and
    group the result by (msLevelA, msLevelB).

    Parameters
    ----------
    tree : your Tree object (with .root and .children).
    use_embedding : bool
        If False, interpret `sim_fn` as (specA, specB)->(score, matched_peaks).
        If None => default CosineGreedy(tol=0.1).
        If True, interpret `sim_fn` as (embA, embB)->float; if None => default
        dreams_embedding_similarity.
    sim_fn : function or None
        The function to compute similarity for a pair of nodes. If None and
        use_embedding=False => CosineGreedy. If None and use_embedding=True =>
        dreams_embedding_similarity. Otherwise, user-provided.
    embeddings_dict : dict or None
        If use_embedding=True, we look up each node's embedding with
        node.spectrum.get('identifier') => embeddings_dict[identifier].
    tolerance : float
        Tolerance for spectral-based Cosine if `sim_fn` is None or
        if your custom function relies on it.
    descendant_mode : bool
        - If False (default),:
            *all distinct node pairs* are compared and appended to
            level_sims[(msA, msB)].
        - If True, we do the "descendant" approach:
            For pairs (A,B):
              1) If msA == msB, we still do pairwise among them (like old approach).
              2) If msA < msB, we only do the pair if node B is in A's subtree
                 (i.e. B is a descendant of A).
              3) If msA > msB, we skip it entirely.

    Returns
    -------
    level_sims : Dict[(msA, msB), List[float]]
        A dict mapping (msLevelA, msLevelB) -> a list of similarity scores.
    """

    # BFS gather all nodes
    nodes = []
    queue = deque([tree.root])
    while queue:
        n = queue.popleft()
        nodes.append(n)
        for c in n.children.values():
            queue.append(c)

    # Precompute set of all descendant node objects
    #    so we can do O(1) membership checks for "is B a descendant of A?"
    descendants_of = {}

    def _compute_descendants(n0) -> set:
        # BFS from n0
        dset = set()
        qq = deque([n0])
        while qq:
            current = qq.popleft()
            for child in current.children.values():
                dset.add(child)
                qq.append(child)
        return dset

    for n in nodes:
        descendants_of[n] = _compute_descendants(n)

    # define a helper to compute node-to-node similarity
    def _node_sim(nA, nB):
        if not use_embedding:
            specA = get_spectrum(nA)
            specB = get_spectrum(nB)
            if sim_fn is None:
                # default CosineGreedy
                (score, _) = compute_cosine_greedy_score(specA, specB, tolerance=tolerance)
            else:
                (score, _) = sim_fn(specA, specB)
            return score
        else:
            embA = get_embedding_for_node(nA, embeddings_dict)
            embB = get_embedding_for_node(nB, embeddings_dict)
            if sim_fn is None:
                return dreams_embedding_similarity(embA, embB)
            else:
                return sim_fn(embA, embB)

    level_sims = defaultdict(list)
    num_nodes = len(nodes)

    # We'll do a double loop i<j to avoid duplication for same-level pairs.
    for i in range(num_nodes):
        nA = nodes[i]
        msA = get_ms_level(nA)
        if msA is None:
            continue

        for j in range(i+1, num_nodes):
            nB = nodes[j]
            msB = get_ms_level(nB)
            if msB is None:
                continue

            if not descendant_mode:
                # all distinct pairs
                score = _node_sim(nA, nB)
                level_sims[(msA, msB)].append(score)
            else:
                # only same-level or A->descendant
                if msA == msB:
                    # same-level => do pair
                    score = _node_sim(nA, nB)
                    level_sims[(msA, msB)].append(score)
                elif msA < msB:
                    # only do if nB in subtree of nA
                    if nB in descendants_of[nA]:
                        score = _node_sim(nA, nB)
                        level_sims[(msA, msB)].append(score)
                    else:
                        # skip, different branch
                        pass
                else:
                    # msA > msB => skip
                    pass

    return dict(level_sims)

def compute_same_level_similarity_limited(
    msn_dataset,
    target_level: int = 2,
    use_embedding: bool = False,
    sim_fn=None,
    embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
    tolerance: float = 0.1,
    max_pairs: int = 5000
) -> List[float]:
    """
    Collect *all* nodes from ALL trees with ms_level == target_level,
    then compute pairwise similarity among them (excluding node with itself).

    If the number of total pairs is small enough (< max_pairs), do them all.
    Otherwise, randomly sample up to max_pairs pairs.

    If use_embedding=False, we interpret sim_fn as (specA, specB)->(score, matched_peaks).
      If sim_fn is None => default CosineGreedy(tol=0.1).
    If use_embedding=True, we interpret sim_fn as (embA, embB)->float.
      If sim_fn is None => default dreams_embedding_similarity.

    Returns
    -------
    List of similarity scores (float).
    """

    # 1) Gather nodes at target_level
    nodes_at_level = []
    for tree in msn_dataset.trees:
        queue = deque([tree.root])
        while queue:
            node = queue.popleft()
            if node.spectrum is not None:
                lvl_str = node.spectrum.get("ms_level", None)
                if lvl_str is not None and int(lvl_str) == target_level:
                    nodes_at_level.append(node)
            for child in node.children.values():
                queue.append(child)

    n = len(nodes_at_level)
    if n < 2:
        return []

    # total number of pairs
    total_pairs = comb(n, 2)  # n*(n-1)//2

    # Decide if we do all pairs or random sampling
    if total_pairs <= max_pairs:
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((nodes_at_level[i], nodes_at_level[j]))
    else:
        pairs = []
        seen = set()
        attempts = 0
        while len(pairs) < max_pairs and attempts < max_pairs*10:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i < j:
                if (i,j) not in seen:
                    seen.add((i,j))
                    pairs.append((nodes_at_level[i], nodes_at_level[j]))
            elif j < i:
                if (j,i) not in seen:
                    seen.add((j,i))
                    pairs.append((nodes_at_level[j], nodes_at_level[i]))
            attempts += 1

    # For each pair, compute similarity
    sims = []
    for (nodeA, nodeB) in pairs:
        if not use_embedding:
            specA = get_spectrum(nodeA)
            specB = get_spectrum(nodeB)
            if sim_fn is None:
                (score, _) = compute_cosine_greedy_score(specA, specB, tolerance=tolerance)
            else:
                (score, _) = sim_fn(specA, specB)
            sims.append(score)
        else:
            embA = get_embedding_for_node(nodeA, embeddings_dict)
            embB = get_embedding_for_node(nodeB, embeddings_dict)
            if sim_fn is None:
                sc = dreams_embedding_similarity(embA, embB)
            else:
                sc = sim_fn(embA, embB)
            sims.append(sc)

    return sims