from collections import defaultdict, deque
import typing as T
import heapq
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

    for tree in trees:
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
            trees_with_missing += 1
            total_missing_nodes += missing_count
            tree_missing_counts.append((tree, missing_count))

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
    print(f"\nTop {top_n} trees with the most missing spectra:")
    for idx, (tree, count) in enumerate(top_missing_trees, 1):
        total_nodes = tree.get_total_nodes_count()
        print(f"{idx}. Tree with root SMILES '{tree.root.spectrum.get('smiles')}': {count} missing spectra out of {total_nodes} nodes.")

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
