from collections import defaultdict, deque
import typing as T
from typing import List
import numpy as np
import json
from rdkit import Chem
import pandas as pd
from collections import defaultdict, Counter
from massspecgym.data.datasets import MSnDataset
import heapq
from scipy import stats
from matchms.importing import load_from_mgf

def find_max_deviation(deviations: T.List[T.Tuple[str, float, float, float]]) -> T.Optional[T.Dict]:
    """
    Finds and returns the tuple with the maximum 'deviation' value in the deviations list.

    Parameters:
        deviations (List[Tuple[str, float, float, float]]):
            A list of tuples, each containing:
                (identifier, desired_value, actual_value, deviation)

    Returns:
        Optional[Dict]:
            A dictionary with keys 'identifier', 'desired_value', 'actual_value', 'deviation'.
            Returns `None` if the input list is empty.
    """
    if not deviations:
        print("The deviations list is empty. No maximum deviation found.")
        return None

    # Initialize variables to track the maximum deviation
    max_deviation_tuple = None
    max_deviation_value = -float('inf')  # Start with negative infinity

    for deviation in deviations:
        identifier, desired_value, actual_value, dev = deviation
        if dev > max_deviation_value:
            max_deviation_value = dev
            max_deviation_tuple = deviation

    if max_deviation_tuple is not None:
        print(f"Maximum deviation found: {max_deviation_value}")
        return {
            'identifier': max_deviation_tuple[0],
            'desired_value': max_deviation_tuple[1],
            'actual_value': max_deviation_tuple[2],
            'deviation': max_deviation_tuple[3]
        }
    else:
        print("No valid deviations found.")
        return None


def analyze_trees( trees, mgf_file_path, spectype = 'ALL_ENERGIES', deviations = None, top_n = 10):
    """
    Analyzes a list of trees and an MGF file to determine molecule-level, tree-level, and spectra-level statistics about missing spectra.

    Parameters:
        trees (List[Tree]): List of Tree instances. Each Tree must have a 'smiles' attribute.
        mgf_file_path (str): Path to the original MGF file.
        spectype (str, optional): The spectype to filter spectra in the MGF file. Defaults to 'ALL_ENERGIES'.
        deviations (List[Dict], optional): List of deviation dictionaries to find the maximum deviation.
        top_n (int, optional): Number of top molecules and trees to return based on missing spectra. Defaults to 10.

    Returns:
        dict: A dictionary containing molecule-level, tree-level, spectra-level, and deviation statistics.
    """
    # ------------------- Molecule-Level Statistics ------------------- #
    smiles_to_trees = defaultdict(list)
    skipped_trees = 0

    for tree in trees:
        smiles = tree.root.spectrum.get('smiles')

        if not smiles:
            print("Warning: Tree with missing SMILES. Skipping this tree.")
            skipped_trees += 1
            continue

        # Group trees by their SMILES strings
        smiles_to_trees[smiles].append(tree)

    total_unique_molecules = len(smiles_to_trees)
    unique_molecules_with_missing_spectra = 0
    unique_molecules_all_trees_missing = 0
    unique_molecules_with_complete_trees = 0
    molecule_missing_counts = {}

    for smiles, group_trees in smiles_to_trees.items():
        trees_missing = 0
        trees_complete = 0

        for tree in group_trees:
            # Initialize a queue for BFS traversal
            queue = deque([tree.root])
            missing_count = 0

            while queue:
                node = queue.popleft()
                if node.spectrum is None:
                    missing_count += 1
                for child in node.children.values():
                    queue.append(child)

            if missing_count > 0:
                trees_missing += 1
            else:
                trees_complete += 1

        if trees_missing > 0:
            unique_molecules_with_missing_spectra += 1
            molecule_missing_counts[smiles] = trees_missing
        if trees_complete > 0:
            unique_molecules_with_complete_trees += 1
        if trees_missing == len(group_trees):
            unique_molecules_all_trees_missing += 1

    # Identify top_n molecules with the most missing trees
    top_missing_molecules = heapq.nlargest(top_n, molecule_missing_counts.items(), key=lambda x: x[1])

    # --------------------- Tree-Level Statistics --------------------- #
    trees_with_missing = 0
    total_missing_nodes = 0
    tree_missing_counts = []
    trees_with_inchi_mismatch = 0
    trees_inchi_mismatch_details = []


    for tree in trees:
        # Initialize a queue for BFS traversal
        queue = deque([tree.root])
        missing_count = 0
        missing_nodes = []
        root_inchi = tree.root.spectrum.get('inchi') if tree.root.spectrum else None
        root_identifier = tree.root.spectrum.get('identifier') if tree.root.spectrum else None
        inchi_mismatch = False
        mismatched_nodes = []

        while queue:
            node = queue.popleft()
            if node.spectrum is None:
                missing_count += 1
                missing_nodes.append(node)
            else:
                node_inchi = node.spectrum.get('inchi')
                if node_inchi != root_inchi:
                    inchi_mismatch = True
                    mismatched_nodes.append({
                        'node': node,
                        'inchi': node_inchi,
                        'identifier': node.spectrum.get('identifier') if node.spectrum else 'Unknown'
                    })

            for child in node.children.values():
                queue.append(child)

        if missing_count > 0:
            trees_with_missing += 1
            total_missing_nodes += missing_count

        tree_missing_counts.append((tree, missing_count, missing_nodes))

        if inchi_mismatch:
            trees_with_inchi_mismatch += 1
            trees_inchi_mismatch_details.append({
                'tree': tree,
                'root_identifier': root_identifier,
                'root_inchi': root_inchi,
                'mismatched_nodes': mismatched_nodes
            })

    # Identify top_n trees with the most missing nodes
    top_missing_trees = sorted(tree_missing_counts, key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------- Spectra-Level Statistics ------------------- #
    spectra_in_mgf = set()
    total_spectra_in_mgf = 0

    try:
        spectra = list(load_from_mgf(mgf_file_path))
        for spec in spectra:
            if spec.get('spectype') == spectype:
                identifier = spec.get('identifier')
                if identifier:
                    spectra_in_mgf.add(identifier)
                    total_spectra_in_mgf += 1
    except FileNotFoundError:
        print(f"Error: MGF file not found at path '{mgf_file_path}'.")
    except Exception as e:
        print(f"Error reading MGF file: {e}")

    # Traverse trees and collect spectra identifiers from nodes with spectrum not None
    spectra_in_trees = set()
    all_identifiers_in_trees = []
    duplicate_identifiers = set()
    identifier_counts = defaultdict(int)

    for tree in trees:
        queue = deque([tree.root])
        while queue:
            node = queue.popleft()
            if node.spectrum is not None:
                identifier = node.spectrum.get('identifier')
                if identifier:
                    all_identifiers_in_trees.append(identifier)
                    identifier_counts[identifier] += 1
                    if identifier_counts[identifier] > 1:
                        duplicate_identifiers.add(identifier)
            for child in node.children.values():
                queue.append(child)

    spectra_in_trees = set(all_identifiers_in_trees)
    total_spectra_in_trees = len(all_identifiers_in_trees)
    unique_spectra_in_trees = len(spectra_in_trees)

    # Check for missing and extra spectra
    missing_spectra = spectra_in_mgf - spectra_in_trees
    extra_spectra = spectra_in_trees - spectra_in_mgf

    # Check if all spectra in trees are unique
    all_unique_in_trees = len(all_identifiers_in_trees) == unique_spectra_in_trees

    # ------------------- Deviation Analysis ------------------- #
    if deviations is None:
        # Aggregate deviations from all trees
        deviations = []
        for tree in trees:
            deviations.extend(tree.deviations)

    # Find the maximum deviation if deviations are present
    max_deviation = find_max_deviation(deviations) if deviations else None


    # ---------------------------- Reporting --------------------------- #
    # Molecule-Level Reporting
    print(f"--- Molecule-Level Statistics ---")
    print(f"Total number of unique molecules: {total_unique_molecules}")
    print(f"Number of unique molecules with at least one tree missing spectra nodes: {unique_molecules_with_missing_spectra}")
    print(f"Number of unique molecules where all trees have missing spectra nodes: {unique_molecules_all_trees_missing}")
    print(f"Number of unique molecules with at least one complete tree: {unique_molecules_with_complete_trees}")
    print(f"\nTop {top_n} molecules with the most trees missing spectra:")
    for idx, (smiles, missing_trees) in enumerate(top_missing_molecules, 1):
        total_trees = len(smiles_to_trees[smiles])
        print(f"{idx}. SMILES: {smiles}, Missing Trees: {missing_trees} out of {total_trees} trees")

    # Tree-Level Reporting
    print(f"\n--- Tree-Level Statistics ---")
    print(f"Total number of trees: {len(trees)}")
    print(f"Number of trees containing nodes with spectrum=None: {trees_with_missing}")
    print(f"Total number of nodes missing spectra across all trees: {total_missing_nodes}")
    print(f"Number of trees with INCHI mismatch: {trees_with_inchi_mismatch}")

    print(f"\nTop {top_n} trees with the most missing spectra:")
    for idx, (tree, count, missing_nodes) in enumerate(top_missing_trees, 1):
        total_nodes = tree.get_total_nodes_count()
        print(f"{idx}. Tree with root IDENTIFIER '{tree.root.spectrum.get('identifier')}': {count} missing spectra out of {total_nodes} nodes.")
        if missing_nodes:
            missing_nodes_mz = [node.value for node in missing_nodes]
            print(f"    Nodes missing spectra (m/z values): {missing_nodes_mz}")

    if trees_with_inchi_mismatch > 0:
        print(f"\nTrees with INCHI mismatch between root and nodes:")
        for idx, mismatch_info in enumerate(trees_inchi_mismatch_details, 1):
            root_identifier = mismatch_info['tree'].root.spectrum.get('identifier')
            root_inchi = mismatch_info['root_inchi']
            print(f"{idx}. Tree with root IDENTIFIER '{root_identifier}' and INCHI '{root_inchi}':")
            for node_info in mismatch_info['mismatched_nodes']:
                node_inchi = node_info['inchi']
                node_identifier = node_info['identifier']
                print(f"   - Node with identifier '{node_identifier}' has INCHI '{node_inchi}'")
    else:
        print(f"\nFound 0 trees with INCHI mismatch between root and nodes.")

    # Spectra-Level Reporting
    print(f"\n--- Spectra-Level Statistics ---")
    print(f"Total number of spectra in MGF with SPECTYPE='{spectype}': {total_spectra_in_mgf}")
    print(f"Total number of spectra present in trees (nodes with spectrum not None): {total_spectra_in_trees}")
    print(f"Number of unique spectra in trees: {unique_spectra_in_trees}")

    if missing_spectra:
        print(f"Number of spectra in MGF not present in trees: {len(missing_spectra)}")
    else:
        print("No spectra from MGF are missing in trees.")

    if extra_spectra:
        print(f"Number of spectra in trees not present in MGF: {len(extra_spectra)}")
    else:
        print("No extra spectra found in trees that are not in MGF.")

    if not all_unique_in_trees:
        print("Warning: There are duplicate spectra in trees based on 'IDENTIFIER'.")
        print(f"Number of duplicate 'IDENTIFIER's: {len(duplicate_identifiers)}")
        print("Duplicate 'IDENTIFIER's:")
        for identifier in duplicate_identifiers:
            print(f"  {identifier}")
    else:
        print("All spectra in trees are unique based on 'IDENTIFIER'.")

    # Deviation Statistics Reporting
    if max_deviation:
        print(f"\n--- Deviation Statistics ---")
        print(f"Maximum deviation found at tree: {max_deviation['identifier']}")
        print(f"Maximum deviation found: {max_deviation['deviation']}")
        print(f"Desired Value: {max_deviation['desired_value']}")
        print(f"Actual Value: {max_deviation['actual_value']}")
    else:
        print("\nNo deviations recorded.")

    # Additional Reporting
    print(f"\nNumber of trees skipped due to missing SMILES: {skipped_trees}")

    # ---------------------------- Return ------------------------------ #
    return {
        'molecule_level': {
            'total_unique_molecules': total_unique_molecules,
            'unique_molecules_with_missing_spectra': unique_molecules_with_missing_spectra,
            'unique_molecules_all_trees_missing': unique_molecules_all_trees_missing,
            'unique_molecules_with_complete_trees': unique_molecules_with_complete_trees,
            'top_n_molecules_missing_trees': top_missing_molecules
        },
        'tree_level': {
            'total_trees': len(trees),
            'trees_with_missing_spectra': trees_with_missing,
            'total_missing_nodes': total_missing_nodes,
            'trees_with_inchi_mismatch': trees_with_inchi_mismatch,
            'trees_inchi_mismatch_details': trees_inchi_mismatch_details,
            'top_n_trees_missing_spectra': top_missing_trees
        },
        'spectra_level': {
            'total_spectra_in_mgf': total_spectra_in_mgf,
            'total_spectra_in_trees': total_spectra_in_trees,
            'unique_spectra_in_trees': unique_spectra_in_trees,
            'missing_spectra_in_trees': len(missing_spectra),
            'extra_spectra_in_trees': len(extra_spectra),
            'all_spectra_unique_in_trees': all_unique_in_trees,
            'duplicate_identifiers': list(duplicate_identifiers) if not all_unique_in_trees else []
        },
        'deviation_statistics': {
            'max_deviation': max_deviation if deviations is not None else None
        },
        'additional': {
            'skipped_trees_due_to_missing_smiles': skipped_trees
        }
    }


def compute_fold_statistics(msn_dataset: MSnDataset, smiles_to_fold: T.Dict[str, str]) -> T.Dict[str, T.Any]:
    """
    Compute and print statistics per fold for the given MSnDataset and smiles_to_fold mapping.
    Returns the statistics as a dictionary.

    Parameters:
    - msn_dataset: An instance of MSnDataset.
    - smiles_to_fold: A dictionary mapping SMILES strings to fold names ('train', 'val', 'test').

    Returns:
    - fold_stats: A dictionary containing statistics per fold.
    """

    fold_trees = defaultdict(list)
    unmatched_trees = []

    for tree in msn_dataset.trees:
        smiles = tree.root.spectrum.get('smiles')
        fold = smiles_to_fold.get(smiles)
        if fold:
            fold_trees[fold].append(tree)
        else:
            unmatched_trees.append(tree)

    total_trees = len(msn_dataset.trees)
    matched_trees = total_trees - len(unmatched_trees)
    if matched_trees != total_trees:
        print(f"Warning: {len(unmatched_trees)} out of {total_trees} trees were not matched to any fold.")
    else:
        print("All trees have been matched to folds.")

    fold_stats = {}

    # For each fold, compute statistics
    for fold, trees in fold_trees.items():
        depths = []
        branching_factors = []
        precursor_mzs = []
        retention_times = []
        total_nodes_counts = []
        adducts = []

        for tree in trees:
            # Depth and branching factor
            depth = tree.get_depth()
            depths.append(depth)

            branching_factor = tree.get_branching_factor()
            branching_factors.append(branching_factor)

            # Precursor m/z and retention time
            precursor_mz = tree.root.spectrum.get('precursor_mz')
            if precursor_mz is not None:
                precursor_mzs.append(float(precursor_mz))

            retention_time = tree.root.spectrum.get('retention_time')
            if retention_time is not None:
                retention_times.append(float(retention_time))

            # Total nodes count
            total_nodes = tree.get_total_nodes_count()
            total_nodes_counts.append(total_nodes)

            # Adduct
            adduct = tree.root.spectrum.get('adduct')
            if adduct:
                adducts.append(adduct)

        avg_depth = np.mean(depths) if depths else 0
        avg_branching_factor = np.mean(branching_factors) if branching_factors else 0
        avg_precursor_mz = np.mean(precursor_mzs) if precursor_mzs else 0
        avg_retention_time = np.mean(retention_times) if retention_times else 0
        avg_total_nodes = np.mean(total_nodes_counts) if total_nodes_counts else 0

        adduct_counts = Counter(adducts)
        total_adducts = sum(adduct_counts.values())
        adduct_relative_abundance = {adduct: count / total_adducts for adduct, count in adduct_counts.items()}

        sorted_adducts = adduct_counts.most_common()

        print(f"Statistics for fold '{fold}':")
        print(f"  Number of trees: {len(trees)}")
        print(f"  Average depth: {avg_depth:.2f}")
        print(f"  Average branching factor: {avg_branching_factor:.2f}")
        print(f"  Average precursor m/z: {avg_precursor_mz:.2f}")
        print(f"  Average retention time: {avg_retention_time:.2f}")
        print(f"  Average total nodes per tree: {avg_total_nodes:.2f}")
        print("  Relative abundances of adducts:")
        for adduct, count in sorted_adducts:
            rel_abundance = adduct_relative_abundance[adduct]
            print(f"    {adduct}: {rel_abundance * 100:.2f}% ({count} occurrences)")
        print()

        fold_stats[fold] = {
            'num_trees': len(trees),
            'avg_depth': avg_depth,
            'avg_branching_factor': avg_branching_factor,
            'avg_precursor_mz': avg_precursor_mz,
            'avg_retention_time': avg_retention_time,
            'avg_total_nodes': avg_total_nodes,
            'adduct_relative_abundance': adduct_relative_abundance,
            'adduct_counts': adduct_counts
        }

    return fold_stats

def analyze_canonical_smiles(data, mode='spectra'):
    """
    Processes SMILES strings from either filtered spectra or JSON data,
    extracting original and canonical SMILES sets, and computing relevant statistics.

    Parameters:
    - data (list or dict):
        - If mode='spectra', a list of Spectrum objects.
        - If mode='json', a dictionary with SMILES strings as keys.
    - mode (str): Mode of processing. Either 'spectra' or 'json'.

    Returns:
    - original_smiles_set (set): SMILES strings as extracted from the input.
    - canonical_smiles_set (set): Canonicalized SMILES strings.
    - invalid_smiles (set): SMILES strings that could not be canonicalized.
    """

    print("==== RDKit Canonical SMILES ====")
    # Initialize sets to store SMILES
    original_smiles_set = set()
    canonical_smiles_set = set()
    invalid_smiles = set()

    # Validate mode
    if mode not in ['spectra', 'json']:
        raise ValueError("Invalid mode. Choose 'spectra' or 'json'.")

    if mode == 'spectra':
        for spectrum in data:
            # Extract SMILES from spectrum metadata
            smiles = spectrum.metadata.get("SMILES") or spectrum.metadata.get("smiles")
            if smiles:
                original_smiles_set.add(smiles)
                # Canonicalize SMILES using RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    canonical_smiles_set.add(canonical_smiles)
                else:
                    invalid_smiles.add(smiles)
    elif mode == 'json':
        if not isinstance(data, dict):
            raise TypeError("For mode 'json', data must be a dictionary with SMILES as keys.")
        for smiles in data.keys():
            original_smiles_set.add(smiles)
            # Canonicalize SMILES using RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles_set.add(canonical_smiles)
            else:
                invalid_smiles.add(smiles)

    # Compute statistics
    total_smiles = len(original_smiles_set)
    unique_original = len(original_smiles_set)
    unique_canonical = len(canonical_smiles_set)
    num_invalid = len(invalid_smiles)
    # Intersection counts how many original SMILES are already canonical
    intersection_count = len(original_smiles_set.intersection(canonical_smiles_set))

    print("=== SMILES Processing Statistics ===")
    print(f"Mode: {mode.upper()}")
    print(f"Total SMILES extracted: {total_smiles}")
    print(f"Unique original SMILES: {unique_original}")
    print(f"Unique canonical SMILES: {unique_canonical}")
    print(f"Number of invalid SMILES: {num_invalid}")
    print(f"Number of SMILES unchanged after RDKit canonicalization: {intersection_count}")
    print("====================================\n")

    return original_smiles_set, canonical_smiles_set, invalid_smiles


def compare_canonical_smiles(mgf_path: str, json_path: str) -> None:
    """
    Compares canonical SMILES from PubChem (JSON) with those generated by RDKit.

    Parameters:
    - mgf_path (str): Path to the MGF file containing spectra.
    - json_path (str): Path to the JSON file containing PubChem canonical SMILES.

    Returns:
    - None: Prints the comparison statistics and discrepancies.
    """
    print("To make sense of analyzis is recommended to use version of dataset without canonizations, date before 29/09")
    # Load spectra from MGF file
    print("Loading spectra from MGF file...")
    spectra = list(load_from_mgf(mgf_path))
    print(f"Loaded {len(spectra)} spectra.\n")

    # Load PubChem canonical SMILES from JSON
    print("Loading PubChem canonical SMILES from JSON...")
    try:
        with open(json_path, 'r') as f:
            smiles_dict = json.load(f)
        print(f"Loaded canonical SMILES dictionary with {len(smiles_dict)} entries.\n")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}. Exiting function.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}. Exiting function.")
        return

    # Extract unique SMILES from spectra
    print("Extracting unique SMILES from spectra...")
    smiles_in_spectra = set()
    for spectrum in spectra:
        smiles = spectrum.metadata.get('smiles') or spectrum.metadata.get('SMILES')
        if smiles:
            smiles_in_spectra.add(smiles)
    print(f"Total unique SMILES in spectra: {len(smiles_in_spectra)}")
    print(f"Total SMILES in canonicalization dict (PubChem): {len(smiles_dict)}\n")

    # Identify problematic SMILES not in PubChem dictionary
    problematic_smiles = smiles_in_spectra - set(smiles_dict.keys())
    print(f"Found {len(problematic_smiles)} problematic SMILES not in the canonicalization dict.\n")

    # Analyze canonical SMILES with RDKit
    print("Analyzing canonical SMILES with RDKit...")
    original_set_rdkit, canonical_set_rdkit, invalid_smiles_set_rdkit = analyze_canonical_smiles(spectra,
                                                                                                 mode='spectra')
    print(f"Total unique RDKit canonical SMILES: {len(canonical_set_rdkit)}")
    print(f"Total invalid SMILES by RDKit: {len(invalid_smiles_set_rdkit)}\n")

    # Extract PubChem canonical SMILES from the JSON dictionary
    canonical_smiles_set_pubchem = set(smiles_dict.values())
    print(f"Total unique PubChem canonical SMILES: {len(canonical_smiles_set_pubchem)}")

    # Compare PubChem and RDKit canonical SMILES sets
    print("\nComparing PubChem and RDKit canonical SMILES sets...")
    if canonical_smiles_set_pubchem == canonical_set_rdkit:
        print("\n PubChem and RDKit canonical SMILES sets are identical.\n")
    else:
        print("\n PubChem and RDKit canonical SMILES sets differ.\n")

        # SMILES in PubChem but not in RDKit
        pubchem_not_rdkit = canonical_smiles_set_pubchem - canonical_set_rdkit
        print(f" SMILES in PubChem but not in RDKit: {len(pubchem_not_rdkit)}")

        # SMILES in RDKit but not in PubChem
        rdkit_not_pubchem = canonical_set_rdkit - canonical_smiles_set_pubchem
        print(f"\n SMILES in RDKit but not in PubChem: {len(rdkit_not_pubchem)}")

        # Overlapping SMILES
        overlap = canonical_smiles_set_pubchem.intersection(canonical_set_rdkit)
        print(f"\n Number of overlapping canonical SMILES: {len(overlap)}\n")

    # Create mapping: Original SMILES -> PubChem canonical SMILES
    print("Mapping original SMILES to PubChem canonical SMILES...")
    original_to_pubchem = {smi: smiles_dict[smi] for smi in smiles_in_spectra if smi in smiles_dict}
    for smi in smiles_in_spectra - set(smiles_dict.keys()):
        original_to_pubchem[smi] = smi  # Keep original if not in PubChem dict
    print("Mapping complete.\n")

    # Create mapping: Original SMILES -> RDKit canonical SMILES
    print("Mapping original SMILES to RDKit canonical SMILES...")
    original_to_rdkit = {}
    for smi in smiles_in_spectra:
        if smi in invalid_smiles_set_rdkit:
            original_to_rdkit[smi] = smi  # Keep original if RDKit failed
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rdkit_canonical = Chem.MolToSmiles(mol, canonical=True)
                original_to_rdkit[smi] = rdkit_canonical
            else:
                original_to_rdkit[smi] = smi  # Fallback, though already in invalid set
    print("Mapping complete.\n")

    # Compare the two mappings and compute statistics
    print("Comparing PubChem and RDKit canonical SMILES per original SMILES...\n")
    total_smiles = len(smiles_in_spectra)
    both_success = 0
    both_failed = 0
    pubchem_failed = 0
    rdkit_failed = 0
    differing_canon = 0
    differing_smiles = []

    for smi in smiles_in_spectra:
        pubchem_canonical = original_to_pubchem.get(smi, smi)
        rdkit_canonical = original_to_rdkit.get(smi, smi)

        if pubchem_canonical != smi and rdkit_canonical != smi:
            # Both succeeded in canonicalization
            both_success += 1
            if pubchem_canonical != rdkit_canonical:
                differing_canon += 1
                differing_smiles.append(smi)
        elif pubchem_canonical == smi and rdkit_canonical != smi:
            # PubChem failed, RDKit succeeded
            pubchem_failed += 1
        elif pubchem_canonical != smi and rdkit_canonical == smi:
            # RDKit failed, PubChem succeeded
            rdkit_failed += 1
        else:
            # Both failed
            both_failed += 1

    # Print statistics
    print("=== Detailed Comparison Statistics ===")
    print(f"Total SMILES processed: {total_smiles}")
    print(f"Successfully canonicalized by both PubChem and RDKit: {both_success}")
    print(f"Successfully canonicalized by both but differing: {differing_canon}")
    print(f"Canonicalization different by PubChem but same by RDKit: {pubchem_failed}")
    print(f"Canonicalization same by RDKit but different by PubChem: {rdkit_failed}")
    print(f"Canonicalization same as original by both PubChem and RDKit: {both_failed}")
    print(f"Total differing canonical SMILES: {differing_canon}\n")

    i = 0
    if differing_canon > 0:
        print("=== SMILES with Differing Canonical Forms ===")
        for smi in differing_smiles:
            i += 1
            pubchem_canonical = original_to_pubchem[smi]
            rdkit_canonical = original_to_rdkit[smi]
            print(f"Original SMILES: {smi}")
            print(f"PubChem canonical: {pubchem_canonical}")
            print(f"RDKit canonical: {rdkit_canonical}\n")
            if i == 5:
                break
    else:
        print("No differing canonical SMILES found between PubChem and RDKit.\n")

    print("=== Comparison Complete ===")


def compare_distributions(distA: List[float], distB: List[float], test_type: str = "ttest") -> float:
    """
    Compare two distributions (lists of floats) with a chosen test, e.g. 'ttest' or 'ranksum'.
    Return the p-value.

    - "ttest": a standard two-sample t-test
    - "ranksum": a nonparametric Wilcoxon rank-sum test
    """
    arrA = np.array(distA)
    arrB = np.array(distB)
    arrA = arrA[~np.isnan(arrA)]
    arrB = arrB[~np.isnan(arrB)]
    if len(arrA) < 2 or len(arrB) < 2:
        return float('nan')

    if test_type.lower() == "ttest":
        _, pval = stats.ttest_ind(arrA, arrB, equal_var=False, nan_policy='omit')
        return pval
    elif test_type.lower() == "ranksum":
        _, pval = stats.ranksums(arrA, arrB)
        return pval
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


