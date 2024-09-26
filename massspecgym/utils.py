import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import pandas as pd
import typing as T
import pulp
import os
from collections import defaultdict, deque
import heapq
from matchms.importing import load_from_mgf
import uuid
import networkx as nx
import massspecgym.data.datasets as msgym_datasets
from sklearn.model_selection import GroupKFold
from torch_geometric.utils import to_networkx
from itertools import combinations
from pathlib import Path
from myopic_mces.myopic_mces import MCES
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs, Draw
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Scaffolds import MurckoScaffold
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from standardizeUtils.standardizeUtils import (
    standardize_structure_with_pubchem,
    standardize_structure_list_with_pubchem,
)



def load_massspecgym():
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df['mzs'] = df['mzs'].apply(parse_spec_array)
    df['intensities'] = df['intensities'].apply(parse_spec_array)
    return df


def pad_spectrum(
    spec: np.ndarray, max_n_peaks: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad a spectrum to a fixed number of peaks by appending zeros to the end of the spectrum.
    
    Args:
        spec (np.ndarray): Spectrum to pad represented as numpy array of shape (n_peaks, 2).
        max_n_peaks (int): Maximum number of peaks in the padded spectrum.
        pad_value (float, optional): Value to use for padding.
    """
    n_peaks = spec.shape[0]
    if n_peaks > max_n_peaks:
        raise ValueError(
            f"Number of peaks in the spectrum ({n_peaks}) is greater than the maximum number of peaks."
        )
    else:
        return np.pad(
            spec,
            ((0, max_n_peaks - n_peaks), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2, to_np=True):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
        to_np (bool, optional): Convert the fingerprint to numpy array.
    """

    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp


def tanimoto_morgan_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(morgan_fp(mol1, to_np=False), morgan_fp(mol2, to_np=False))


def standardize_smiles(smiles: T.Union[str, T.List[str]]) -> T.Union[str, T.List[str]]:
    """
    Standardize SMILES representation of a molecule using PubChem standardization.
    """
    if isinstance(smiles, str):
        return standardize_structure_with_pubchem(smiles, 'smiles')
    elif isinstance(smiles, list):
        return standardize_structure_list_with_pubchem(smiles, 'smiles')
    else:
        raise ValueError("Input should be a SMILES tring or a list of SMILES strings.")


def mol_to_inchi_key(mol: Chem.Mol, twod: bool = True) -> str:
    """
    Convert a molecule to InChI Key representation.
    
    Args:
        mol (Chem.Mol): RDKit molecule object.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    inchi_key = Chem.MolToInchiKey(mol)
    if twod:
        inchi_key = inchi_key.split("-")[0]
    return inchi_key


def smiles_to_inchi_key(mol: str, twod: bool = True) -> str:
    """
    Convert a SMILES molecule to InChI Key representation.
    
    Args:
        mol (str): SMILES string.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    mol = Chem.MolFromSmiles(mol)
    return mol_to_inchi_key(mol, twod)


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    
    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
        filename="data/" + file_name,
        repo_type="dataset",
    )


def init_plotting(figsize=(6, 2), font_scale=1.0, style="whitegrid"):
    # Set default figure size
    plt.show()  # Does not work without this line for some reason
    sns.set_theme(rc={"figure.figsize": figsize})
    mpl.rcParams['svg.fonttype'] = 'none'
    # Set default style and font scale
    sns.set_style(style)
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(["#009473", "#D94F70", "#5A5B9F", "#F0C05A", "#7BC4C4", "#FF6F61"])


def get_smiles_bpe_tokenizer() -> ByteLevelBPETokenizer:
    """
    Return a Byte-level BPE tokenizer trained on the SMILES strings from the
    `MassSpecGym_test_fold_MCES2_disjoint_molecules_4M.tsv` dataset.
    TODO: refactor to a well-organized class.
    """
    # Initialize the tokenizer
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    smiles_tokenizer = ByteLevelBPETokenizer()
    smiles = pd.read_csv(hugging_face_download(
        "molecules/MassSpecGym_test_fold_MCES2_disjoint_molecules_4M.tsv"
    ), sep="\t")["smiles"]
    smiles_tokenizer.train_from_iterator(smiles, special_tokens=special_tokens)

    # Enable padding
    smiles_tokenizer.enable_padding(direction='right', pad_token="<pad>")

    # Add template processing to include start and end of sequence tokens
    smiles_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", smiles_tokenizer.token_to_id("<s>")),
            ("</s>", smiles_tokenizer.token_to_id("</s>")),
        ],
    )
    return smiles_tokenizer


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


def plot_spectrum(spec, hue=None, xlim=None, ylim=None, mirror_spec=None, highl_idx=None,
                  figsize=(6, 2), colors=None, save_pth=None):

    if colors is not None:
        assert len(colors) >= 3
    else:
        colors = ['blue', 'green', 'red']

    # Normalize input spectrum
    def norm_spec(spec):
        assert len(spec.shape) == 2
        if spec.shape[0] != 2:
            spec = spec.T
        mzs, ins = spec[0], spec[1]
        return mzs, ins / max(ins) * 100
    mzs, ins = norm_spec(spec)

    # Initialize plotting
    init_plotting(figsize=figsize)
    fig, ax = plt.subplots(1, 1)

    # Setup color palette
    if hue is not None:
        norm = matplotlib.colors.Normalize(vmin=min(hue), vmax=max(hue), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
        plt.colorbar(mapper, ax=ax)

    # Plot spectrum
    for i in range(len(mzs)):
        if hue is not None:
            color = mcolors.to_hex(mapper.to_rgba(hue[i]))
        else:
            color = colors[0]
        plt.plot([mzs[i], mzs[i]], [0, ins[i]], color=color, marker='o', markevery=(1, 2), mfc='white', zorder=2)

    # Plot mirror spectrum
    if mirror_spec is not None:
        mzs_m, ins_m = norm_spec(mirror_spec)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            label = str(round(-x)) if x < 0 else str(round(x))
            return label

        for i in range(len(mzs_m)):
            plt.plot([mzs_m[i], mzs_m[i]], [0, -ins_m[i]], color=colors[2], marker='o', markevery=(1, 2), mfc='white',
                     zorder=1)
        ax.yaxis.set_major_formatter(major_formatter)

    # Setup axes
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, max(mzs) + 10)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('m/z')
    plt.ylabel('Intensity [%]')

    if save_pth is not None:
        raise NotImplementedError()


def show_mols(mols, legends='new_indices', smiles_in=False, svg=False, sort_by_legend=False, max_mols=500,
              legend_float_decimals=4, mols_per_row=6, save_pth: T.Optional[Path] = None):
    """
    Returns svg image representing a grid of skeletal structures of the given molecules. Copy-pasted
     from https://github.com/pluskal-lab/DreaMS/blob/main/dreams/utils/mols.py

    :param mols: list of rdkit molecules
    :param smiles_in: True - SMILES inputs, False - RDKit mols
    :param legends: list of labels for each molecule, length must be equal to the length of mols
    :param svg: True - return svg image, False - return png image
    :param sort_by_legend: True - sort molecules by legend values
    :param max_mols: maximum number of molecules to show
    :param legend_float_decimals: number of decimal places to show for float legends
    :param mols_per_row: number of molecules per row to show
    :param save_pth: path to save the .svg image to
    """
    if smiles_in:
        mols = [Chem.MolFromSmiles(e) for e in mols]

    if legends == 'new_indices':
        legends = list(range(len(mols)))
    elif legends == 'masses':
        legends = [ExactMolWt(m) for m in mols]
    elif callable(legends):
        legends = [legends(e) for e in mols]

    if sort_by_legend:
        idx = np.argsort(legends).tolist()
        legends = [legends[i] for i in idx]
        mols = [mols[i] for i in idx]

    legends = [f'{l:.{legend_float_decimals}f}' if isinstance(l, float) else str(l) for l in legends]

    img = Draw.MolsToGridImage(mols, maxMols=max_mols, legends=legends, molsPerRow=min(max_mols, mols_per_row),
                         useSVG=svg, returnPNG=False)

    if save_pth:
        with open(save_pth, 'w') as f:
            f.write(img.data)

    return img

class MyopicMCESNew:
    """
    A new version of MyopicMCES class designed to handle the updated MCES function parameters.
    """
    def __init__(
        self,
        ind: int = 0,  # Index for parallel processing
        solver: str = 'PULP_CBC_CMD',  # Default solver to use
        threshold: int = 15,  # MCES threshold for distance calculations
        always_stronger_bound: bool = True,  # Use the second stronger bound by default
        no_ilp_threshold: bool = False,  # Option to always return the exact distance, ignoring the threshold
        solver_options: dict = None,  # Additional solver options
        catch_errors: bool = False  # Option to catch errors during MCES computation
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        self.no_ilp_threshold = no_ilp_threshold
        self.catch_errors = catch_errors
        if solver_options is None:
            solver_options = {'msg': 0}  # Make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        try:
            retval = MCES(
                smiles1=smiles_1,  # Correct parameter name
                smiles2=smiles_2,  # Correct parameter name
                threshold=self.threshold,
                i=self.ind,
                solver=self.solver,
                solver_options=self.solver_options,
                no_ilp_threshold=self.no_ilp_threshold,
                always_stronger_bound=self.always_stronger_bound,
                catch_errors=self.catch_errors
            )
            # Extract the relevant output (distance)
            dist = retval[1]
            return dist
        except Exception as e:
            if self.catch_errors:
                print(f"Error calculating MCES for SMILES {smiles_1} and {smiles_2}: {e}")
                return float('inf')  # Return a high distance on error if catching errors
            else:
                raise e  # Reraise the error if not catching


class MyopicMCES():
    def __init__(
        self,
        ind: int = 0,  # dummy index
        solver: str = pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
        threshold: int = 15,  # MCES threshold
        always_stronger_bound: bool = True, # "False" makes computations a lot faster, but leads to overall higher MCES values
        solver_options: dict = None
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options
        )
        dist = retval[1]
        return dist


# def visualize_MSn_tree(tree, figsize=(12, 8)):
#     def add_path_to_graph(graph, path):
#         path_nodes = []
#         for i, label in enumerate(path):
#             label = f"{label:.3f}"
#             parent_nodes = "_".join(path_nodes)
#             node_name = f"{parent_nodes}_{label}" if parent_nodes else label
#             graph.add_node(node_name, label=label)
#             if path_nodes:
#                 graph.add_edge(path_nodes[-1], node_name)
#             path_nodes.append(node_name)
#
#     G = nx.DiGraph()
#     for path in tree.paths:
#         add_path_to_graph(G, path)
#
#     pos = nx.bfs_layout(G, list(G.nodes)[0])
#     labels = {node: G.nodes[node]['label'] for node in G.nodes}
#     plt.figure(figsize=figsize)
#     nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000,
#             node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
#     plt.show()


def smiles_to_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles


def train_val_test_split(all_smiles, scaffolds, n_splits=10):
    X = all_smiles
    y = np.random.rand(len(all_smiles))  # dummy target variable

    group_kfold = GroupKFold(n_splits=n_splits)

    train = []
    validation = []
    test = []

    # calculate the number of splits for each set based on the ratios
    n_test_splits = int(n_splits * 0.1)
    n_val_splits = int(n_splits * 0.1)
    n_train_splits = n_splits - n_test_splits - n_val_splits

    # ensure that both validation and test splits aren't empty
    if n_test_splits == 0:
        n_test_splits = 1
        n_train_splits -= 1

    if n_val_splits == 0:
        n_val_splits = 1
        n_train_splits -= 1

    # perform GroupKFold split
    for fold_num, (_, test_index) in enumerate(group_kfold.split(X, y, groups=np.array(scaffolds))):
        if fold_num < n_test_splits:
            test.extend(test_index)
        elif fold_num < n_test_splits + n_val_splits:
            validation.extend(test_index)
        else:
            train.extend(test_index)
    
    return train, validation, test


def create_split_file(msn_dataset, train_idxs, val_idxs, test_idxs, filepath):
    if os.path.exists(filepath):
        print(f"split tsv file already exists at {filepath}")
        return

    split_df = pd.DataFrame(columns=["identifier", "fold"])
    all_indexes = [("train", train_idxs), 
                   ("val", val_idxs),
                    ("test", test_idxs)]

    rows = []

    # create the dataframe row by row
    for fold, split_idxs in all_indexes:
        for idx in split_idxs:
            msn_id = msn_dataset.identifiers[idx]
            rows.append({"identifier": msn_id, "fold": fold})

    # concatenate the rows into the dataframe
    split_df = pd.concat([split_df, pd.DataFrame(rows)], ignore_index=True)
    split_df.to_csv(filepath, sep='\t', index=False)
    print(f"split tsv file was created successfully at {filepath}")


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


def assign_positions(G, root):
    """
    Assigns positions to nodes in a hierarchical layout.

    Parameters:
        G (nx.DiGraph): The tree graph.
        root (node): The root node of the tree.

    Returns:
        dict: A dictionary mapping nodes to positions.
    """
    pos = {}
    x = 0

    def assign_x(node, depth):
        nonlocal x
        children = list(G.successors(node))
        if not children:
            pos[node] = (x, -depth)
            x += 1
        else:
            for child in children:
                assign_x(child, depth + 1)
            child_x = [pos[child][0] for child in children]
            pos[node] = (sum(child_x) / len(child_x), -depth)

    assign_x(root, 0)
    return pos


def is_tree(G: nx.DiGraph) -> bool:
    """
    Checks if a given directed graph is a tree.

    Parameters:
        G (nx.DiGraph): The directed graph to check.

    Returns:
        bool: True if G is a tree, False otherwise.
    """
    # A directed tree must be a directed acyclic graph (DAG) with exactly one root
    if not nx.is_directed_acyclic_graph(G):
        print("Graph is not a Directed Acyclic Graph (DAG).")
        return False

    # Identify potential roots (nodes with in_degree 0)
    roots = [n for n, d in G.in_degree() if d == 0]
    if len(roots) != 1:
        print(f"Graph has {len(roots)} roots; a tree must have exactly one root.")
        return False

    # Check if all nodes are reachable from the root
    root = roots[0]
    descendants = nx.descendants(G, root)
    if len(descendants) + 1 != len(G.nodes()):
        print("Not all nodes are reachable from the root.")
        return False

    return True


def visualize_tree(tree, save_path=None, figsize=(50, 40), dpi=300):
    """
    Visualizes the tree with hierarchical layout.
    Nodes with spectrum=None are colored red, others are colored blue.
    For large trees, it can save the visualization as a high-resolution image.

    Parameters:
        tree (Tree): The tree to visualize.
        save_path (str, optional): Path to save the image. If None, displays the plot.
        figsize (tuple, optional): Size of the matplotlib figure.
        dpi (int, optional): Resolution of the saved image.
    """
    G = nx.DiGraph()
    queue = deque()
    queue.append(tree.root)
    visited = set()

    while queue:
        node = queue.popleft()
        if id(node) in visited:
            continue
        visited.add(id(node))
        G.add_node(id(node), value=node.value, spectrum=node.spectrum)
        for child in node.children.values():
            G.add_edge(id(node), id(child))
            queue.append(child)

    try:
        if not is_tree(G):
            print("Warning: The graph is not a tree. Visualization may not be accurate.")
    except Exception as e:
        print(f"Error checking if graph is a tree: {e}")

    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['spectrum'] is None:
            node_colors.append('red')
        else:
            node_colors.append('blue')

    label_counts = defaultdict(int)
    labels = {}
    for node_id, attrs in G.nodes(data=True):
        label = f"{attrs['value']:.3f}"
        count = label_counts[label]
        unique_label = f"{count}.{label}" if count > 0 else label
        labels[node_id] = unique_label
        label_counts[label] += 1

    try:
        root_node = id(tree.root)
        pos = assign_positions(G, root_node)
    except ValueError as ve:
        print(f"Error in hierarchical layout: {ve}")
        print("Falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        print(f"Unexpected error in hierarchical layout: {e}")
        pos = nx.spring_layout(G, seed=42)

    if save_path:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')

    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Spectrum Present'),
        Patch(facecolor='red', edgecolor='black', label='Spectrum Missing')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title("Tree Visualization with Missing Spectra Nodes Highlighted")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Tree visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


