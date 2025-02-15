import pandas as pd
import ast
import json
import typing as T
import numpy as np
import torch
from collections import deque
import matchms
from numpy.f2py.auxfuncs import throw_error

import massspecgym.utils as utils
from pathlib import Path
from rdkit import Chem
from typing import Optional, Union, Dict, List, Tuple
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from matchms.importing import load_from_mgf
from massspecgym.featurize import SpectrumFeaturizer
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey
from massspecgym.tools.data import compute_root_mol_freq

import time
import functools

def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Only print if time exceeds threshold (e.g., 0.001 seconds)
        if elapsed > 0.001:
            print(f"[PROFILE] {func.__name__} took {elapsed:.6f} seconds")
        return result
    return wrapper

class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is
    responsible for loading the data from disk and applying transformation steps to the spectra and
    molecules.
    """

    def __init__(
        self,
        spec_transform: T.Optional[T.Union[SpecTransform, T.Dict[str, SpecTransform]]] = None,
        mol_transform: T.Optional[T.Union[MolTransform, T.Dict[str, MolTransform]]] = None,
        pth: T.Optional[Path] = None,
        return_mol_freq: bool = True,
        return_identifier: bool = True,
        dtype: T.Type = torch.float32
    ):
        """
        Args:
            pth (Optional[Path], optional): Path to the .tsv or .mgf file containing the mass spectra.
                Default is None, in which case the MassSpecGym dataset is downloaded from HuggingFace Hub.
        """
        self.pth = pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
        self.return_mol_freq = return_mol_freq

        if self.pth is None:
            self.pth = utils.hugging_face_download("MassSpecGym.tsv")

        if isinstance(self.pth, str):
            self.pth = Path(self.pth)

        if self.pth.suffix == ".tsv":
            self.metadata = pd.read_csv(self.pth, sep="\t")
            self.spectra = self.metadata.apply(
                lambda row: matchms.Spectrum(
                    mz=np.array([float(m) for m in row["mzs"].split(",")]),
                    intensities=np.array(
                        [float(i) for i in row["intensities"].split(",")]
                    ),
                    metadata={"precursor_mz": row["precursor_mz"]},
                ),
                axis=1,
            )
            self.metadata = self.metadata.drop(columns=["mzs", "intensities"])
        elif self.pth.suffix == ".mgf":
            self.spectra = list(load_from_mgf(str(self.pth)))
            self.metadata = pd.DataFrame([s.metadata for s in self.spectra])
        else:
            raise ValueError(f"{self.pth.suffix} file format not supported.")

        if self.return_mol_freq:
            if "inchikey" not in self.metadata.columns:
                self.metadata["inchikey"] = self.metadata["smiles"].apply(utils.smiles_to_inchi_key)
            self.metadata["mol_freq"] = self.metadata.groupby("inchikey")["inchikey"].transform("count")

        self.return_identifier = return_identifier
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.spectra)

    @profile_function
    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra[i]
        # spec = (
        #     self.spec_transform(spec)
        #     if transform_spec and self.spec_transform
        #     else spec
        # )
        # spec = torch.as_tensor(spec, dtype=self.dtype)

        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]
        # mol = self.mol_transform(mol) if transform_mol and self.mol_transform else mol
        # if isinstance(mol, np.ndarray):
        #     mol = torch.as_tensor(mol, dtype=self.dtype)

        # Apply all transformations to the spectrum
        item = {}
        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec

        # Apply all transformations to the molecule
        if transform_mol and self.mol_transform:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)
        else:
            item["mol"] = mol

        # Add other metadata to the item
        item.update({
            k: metadata[k] for k in ["precursor_mz", "adduct"]
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

        # TODO: this should be refactored
        for k, v in item.items():
            if not isinstance(v, str):
                item[k] = torch.as_tensor(v, dtype=self.dtype)

        return item

    @staticmethod
    @staticmethod
    @profile_function
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        """
        Custom collate function to handle the outputs of __getitem__.
        """
        return default_collate(batch)


class RetrievalDataset(MassSpecDataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures, with additional
    candidates of molecules for retrieval based on spectral similarity.
    """

    def __init__(
        self,
        mol_label_transform: MolTransform = MolToInChIKey(),
        candidates_pth: T.Optional[T.Union[Path, str]] = None,
        **kwargs,
    ):
        """
        Args:
            mol_label_transform (MolTransform, optional): Transformation to apply to the candidate molecules.
                Defaults to `MolToInChIKey()`.
            candidates_pth (Optional[Union[Path, str]], optional): Path to the .json file containing the candidates for
                retrieval. Defaults to None, in which case the candidates for standard `molecular retrieval` challenge
                are downloaded from HuggingFace Hub. If set to `bonus`, the candidates based on molecular formulas
                for the `bonus chemical formulae challenge` are downloaded instead.
        """
        super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform

        # Download candidates from HuggigFace Hub if not a path to exisiting file is passed
        if self.candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_mass.json"
            )
        elif self.candidates_pth == 'bonus':
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_formula.json"
            )
        elif isinstance(self.candidates_pth, str):
            if Path(self.candidates_pth).is_file():
                self.candidates_pth = Path(self.candidates_pth)
            else:
                self.candidates_pth = utils.hugging_face_download(candidates_pth)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            self.candidates = json.load(file)

    def __getitem__(self, i) -> dict:
        # Get base item (spec, mol, and metadata)
        item = super().__getitem__(i, transform_mol=False)

        # Save original SMILES
        item["smiles"] = item["mol"]

        # Retrieve candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = self.candidates[item["mol"]]

        # Save candidates_smiles
        item["candidates_smiles"] = item["candidates"]

        # Create labels
        item_label = self.mol_label_transform(item["mol"])
        item["labels"] = [
            self.mol_label_transform(c) == item_label for c in item["candidates"]
        ]
        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        # Transform query and candidates
        item["mol"] = self.mol_transform(item["mol"])
        item["candidates"] = [self.mol_transform(c) for c in item["candidates"]]

        # Convert query mol to tensor if needed
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)

        # Convert all candidates to tensors
        item["candidates"] = [
            torch.as_tensor(c, dtype=self.dtype) if isinstance(c, np.ndarray) else c
            for c in item["candidates"]
        ]

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        # Collate everything except candidates/labels/candidates_smiles using default_collate
        collated_batch = {}
        for k in batch[0].keys():
            if k not in ["candidates", "labels", "candidates_smiles"]:
                collated_batch[k] = default_collate([item[k] for item in batch])

        # Candidates: concatenate along first dimension
        collated_batch["candidates"] = torch.as_tensor(
            np.concatenate([item["candidates"] for item in batch])
        )

        # Labels: flatten into a single 1D tensor
        collated_batch["labels"] = torch.as_tensor(
            sum([item["labels"] for item in batch], start=[])
        )

        # batch_ptr: number of candidates per example
        print("I am not here")
        collated_batch["batch_ptr"] = torch.as_tensor(
            [len(item["candidates"]) for item in batch]
        )

        # candidates_smiles: just sum lists
        collated_batch["candidates_smiles"] = \
            sum([item["candidates_smiles"] for item in batch], start=[])

        return collated_batch


# TODO: Datasets for unlabeled data.


class TreeNode:
    def __init__(self, value, spectrum):
        self.value = value # m/z value of the node
        self.spectrum = spectrum # Spectrum associated with this node
        self.children = {} # Dictionary to hold child nodes

    def __repr__(self, level=0):
        ret = "  " * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__repr__(level + 1)
        return ret

    def get_child(self, child_value) -> 'TreeNode':
        key = round(child_value, 3)
        if key in self.children:
            return self.children[key]
        else:
            raise ValueError(f"Child with value {child_value} not found in children of node with value {self.value}")

    def add_child(self, child_value, spectrum=None) -> Optional['TreeNode']:

        if spectrum is not None and self.spectrum is not None:
            parent_inchi = self.spectrum.get('inchi')
            child_inchi = spectrum.get('inchi')
            if parent_inchi != child_inchi:
                print(f"InChI mismatch between parent ({self.spectrum.get('identifier')}) "
                      f"and child ({spectrum.get('identifier')}): {parent_inchi} != {child_inchi}")
                return None

        # If child spectrum is None assign spectrum, if it is not None check is same or not add save as conflict
        if child_value in self.children:
            child_node = self.children[child_value]
            if spectrum is not None:
                if child_node.spectrum is not None and child_node.spectrum != spectrum:
                    # Keep the existing spectrum, do not overwrite
                    print(f"spectrum conflict at {spectrum.get('identifier')} and"
                          f" {child_node.spectrum.get('identifier')} identifiers")
                else:
                    child_node.spectrum = spectrum
            return child_node
        else:
            # Create new child node with the spectrum (could be None)
            child_node = TreeNode(child_value, spectrum=spectrum)
            self.children[child_value] = child_node
            return child_node

    def get_child_closest(self, target_value) -> Tuple[Optional['TreeNode'], Optional[float]]:
        """
        Return the child node whose value is closest to target_value.
        Also return the deviation (absolute difference between target_value and child_node.value).
        """
        if not self.children:
            return None, None
        else:
            closest_child = None
            min_deviation = float('inf')
            for child in self.children.values():
                deviation = abs(child.value - target_value)
                if deviation < min_deviation:
                    min_deviation = deviation
                    closest_child = child
            return closest_child, min_deviation

    def get_depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children.values())

    def get_branching_factor(self) -> int:
        if not self.children:
            return 0
        return max(
            len(self.children),
            max(child.get_branching_factor() for child in self.children.values())
            )

    def get_edges(self):
        edges = []
        for child in self.children.values():
            edges.append((self.value, child.value))
            edges.extend(child.get_edges())
        return edges

    def prune_missing_spectra(self):
        """
        Recursively prune child nodes that have spectrum == None.
        If a child node has spectrum == None, remove it and its entire subtree.
        """
        children_to_remove = []
        for child_value, child_node in self.children.items():
            if child_node.spectrum is None:
                children_to_remove.append(child_value)
            else:
                child_node.prune_missing_spectra()
        # Remove the marked children
        for child_value in children_to_remove:
            del self.children[child_value]


class Tree:
    def __init__(self, root: float, spectrum=None, max_allowed_deviation: float = 0.005):
        self.root = TreeNode(root, spectrum=spectrum)
        self.max_allowed_deviation = max_allowed_deviation
        self.deviations = []

    def __repr__(self):
        return repr(self.root)

    def add_path(self, path: List[float]) -> None:
        # self.paths.append(path)
        if path[0] == self.root.value:
            path = path[1:]  # Skip the root node if it's in the path
        current_node = self.root
        for node in path:
            current_node = current_node.add_child(node)

    def add_path_with_spectrum(self, path: List[float], spectrum: matchms.Spectrum) -> None:
        if path[0] == self.root.value:
            path = path[1:]  # Skip the root node if it's in the path
        current_node = self.root
        for i, node_value in enumerate(path):
            if i == len(path) - 1:
                # Last node in the path, associate the spectrum
                # Check if the node exists
                # if node_value in current_node.children:
                #     child_node = current_node.children[node_value]
                #     current_node.add_child(node_value, spectrum=spectrum)
                #     # if child_node.spectrum is None:
                #     #     # Assign the spectrum to the existing node
                #     #     child_node.spectrum = spectrum
                #     # else:
                #     #     # Node already has a spectrum; handle conflict if necessary
                #     #     if child_node.spectrum != spectrum:
                #     #         # Record the conflict (optional)
                #     #         print(f"spectrum conflict at {child_node.spectrum.get('identifier')} and"
                #     #               f" {spectrum.get('identifier')} identifiers")
                #     #         # Keep the existing spectrum
                # else:
                #     # Create a new child node with the spectrum
                child_node = current_node.add_child(node_value, spectrum)
                current_node = child_node
            else:
                # Intermediate nodes
                if current_node.children:
                    # Find the child with the closest value
                    child_node, deviation = current_node.get_child_closest(node_value)
                    if child_node is None or deviation > self.max_allowed_deviation:
                        # Create a new node with spectrum=None
                        child_node = current_node.add_child(node_value, spectrum=None)

                    self.deviations.append((
                        self.root.spectrum.get('identifier') if self.root.spectrum else 'Unknown',
                        node_value,
                        child_node.value,
                        deviation if child_node.value != node_value else 0.0
                    ))
                    current_node = child_node
                else:
                    # No children, create a new node with spectrum=None
                    child_node = current_node.add_child(node_value, spectrum=None)
                    current_node = child_node

    def get_depth(self):
        return self.root.get_depth()

    def get_total_nodes_count(self) -> int:
        count = 0
        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            count += 1
            for child in node.children.values():
                queue.append(child)

        return count

    def get_branching_factor(self):
        return self.root.get_branching_factor()

    def get_edges(self):
        # Exclude edges starting and ending at the root node
        # Exclude self root as they do not represent any meaningful fragmentation event.
        edges = self.root.get_edges()
        edges = [(u, v) for u, v in edges if u != self.root.value or v != self.root.value]
        return edges

    def prune_missing_spectra(self):
        """
        Prune the tree by removing branches starting from nodes with spectrum == None.
        Do not prune root if is None
        """
        self.root.prune_missing_spectra()

    def cut_at_level(self, level: int):
        """
        Cuts the tree at the specified level.

        Args:
            level (int): The maximum depth to keep in the tree.
                         Level 0 corresponds to the root node.
        """
        if level < 0:
            raise ValueError("cut_tree_at_level must be non-negative")
        self._cut_node_at_level(self.root, current_level=0, max_level=level)

    def _cut_node_at_level(self, node: TreeNode, current_level: int, max_level: int):
        """
        Recursively cuts the tree at the specified level.

        Args:
            node (TreeNode): The current node being processed.
            current_level (int): The current depth in the tree.
            max_level (int): The maximum depth to retain.
        """
        if current_level >= max_level:
            node.children = {}
        else:
            for child in node.children.values():
                self._cut_node_at_level(child, current_level + 1, max_level)

    def to_pyg_data(self, featurizer: Optional[SpectrumFeaturizer] = None, hierarchical_tree: bool = False):

        if featurizer is None:
            edges = self.get_edges()

            # Extract unique node indices
            nodes_set = set(sum(edges, ()))
            node_indices = {node: idx for idx, node in enumerate(nodes_set)}

            # Prepare edge_index tensor
            edge_index = torch.tensor([[node_indices[edge[0]], node_indices[edge[1]]] for edge in edges],
                                      dtype=torch.long).t().contiguous()

            # Prepare node features tensor
            node_list = list(nodes_set)
            x = torch.tensor(node_list, dtype=torch.float).view(-1, 1)

            # Create Data object
            data = Data(x=x, edge_index=edge_index)

            return data

        # Collect nodes and edges
        nodes = []
        edges = []

        # Use a queue to traverse the tree (Breadth-First Search)
        queue = deque()
        queue.append(self.root)
        visited = set()
        while queue:
            node = queue.popleft()
            if id(node) in visited:
                continue
            visited.add(id(node))
            nodes.append(node)
            for child in node.children.values():
                edges.append((node, child))
                queue.append(child)

        # Assign indices to nodes
        node_to_index = {node: idx for idx, node in enumerate(nodes)}

        if not hierarchical_tree:
            reverse_edges = [(child, parent) for parent, child in edges]
            edges.extend(reverse_edges)

        # Build edge_index tensor
        if edges:
            edge_index = torch.tensor(
                [[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges],
                dtype=torch.long
            ).t().contiguous()
        else:
            # If no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Build node features
        node_features = []
        for node in nodes:
            feature_tensor = featurizer.featurize(node)
            node_features.append(feature_tensor)

        feature_shapes = [f.shape for f in node_features]
        if len(set(feature_shapes)) != 1:
            raise ValueError(f"Inconsistent node feature shapes: {feature_shapes}")

        # Determine if features are NumPy arrays or PyTorch tensors
        first_feature = node_features[0]
        if isinstance(first_feature, np.ndarray):
            node_features = [torch.from_numpy(f).float() for f in node_features]

        x = torch.stack(node_features)  # Shape: (num_nodes, feature_dim)

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        return data


class MSnDataset(MassSpecDataset):
    def __init__(
        self,
        pth: Optional[Path] = None,
        dtype: torch.dtype = torch.float32,
        mol_transform: T.Optional[T.Union[MolTransform, T.Dict[str, MolTransform]]] = None,
        featurizer: Optional[SpectrumFeaturizer] = None,
        max_allowed_deviation: float = 0.005,
        prune_missing_spectra: bool = True,
        hierarchical_tree: bool = False,
        return_mol_freq: bool = True,
        cut_tree_at_level: Optional[int] = None  # New parameter
    ):
        # load dataset using the parent class
        super().__init__(pth=pth, return_mol_freq=False)

        self.mol_transform = mol_transform
        self.max_allowed_deviation = max_allowed_deviation
        self.dtype = dtype
        self.metadata = self.metadata[self.metadata["spectype"] == "ALL_ENERGIES"]
        self.featurizer = featurizer
        self.return_mol_freq = return_mol_freq
        self.cut_tree_at_level = cut_tree_at_level


        # Assume the metadata includes 'inchi_aux' column
        if self.return_mol_freq:
            self.metadata = compute_root_mol_freq(self.metadata, "inchi_aux")

        # Map identifier to spectra
        self.identifier_to_spectrum = {spectrum.get('identifier'): spectrum for spectrum in self.spectra}

        # get paths from the metadata
        self.all_tree_paths = self._parse_paths_from_df(self.metadata)

        # Generate trees from paths and their corresponding SMILES, applying tree cutting
        self.trees, self.pyg_trees, self.smiles = self._generate_trees(
            self.all_tree_paths,
            prune_missing_spectra=prune_missing_spectra,
            hierarchical_tree=hierarchical_tree,
            cut_tree_at_level=self.cut_tree_at_level
        )

        # split trees to folds
        self.root_identifier_to_index = {}
        for idx, tree in enumerate(self.trees):
            root_identifier = tree.root.spectrum.get('identifier')
            self.root_identifier_to_index[root_identifier] = idx

        # Precompute molecular features
        self.mol_features = []
        for smi in self.smiles:
            mol_feature = self._compute_mol_feature(smi)
            self.mol_features.append(mol_feature)

        self.tree_depths = self._get_tree_depths(self.trees)
        self.branching_factors = self._get_branching_factors(self.trees)

    def __len__(self):
        return len(self.pyg_trees)

    def __getitem__(self, idx: int) -> dict:
        spec_tree = self.pyg_trees[idx]
        mol_feature = self.mol_features[idx]

        item = {"spec": spec_tree, "mol": mol_feature}

        # Extract additional attributes from the tree's root spectrum
        root_spectrum = self.trees[idx].root.spectrum
        if root_spectrum is not None:
            precursor_mz = root_spectrum.get('precursor_mz')

            if precursor_mz is not None:
                precursor_mz = float(precursor_mz)
            else:
                precursor_mz = float('nan')

            adduct = root_spectrum.get('adduct', "")
            identifier = root_spectrum.get('identifier', "")

            item["precursor_mz"] = precursor_mz
            item["adduct"] = adduct
            item["identifier"] = identifier

            if self.return_mol_freq:
                row = self.metadata[self.metadata["identifier"] == identifier]
                if len(row) == 1:
                    mol_freq = row["mol_freq"].values[0]
                else:
                    # If not found or multiple found, default to NaN
                    mol_freq = float('nan')
                item["mol_freq"] = mol_freq

        else:
            # If there is no root spectrum, set defaults
            item["precursor_mz"] = float('nan')
            item["adduct"] = ""
            item["identifier"] = ""
            if self.return_mol_freq:
                item["mol_freq"] = float('nan')

        # Convert all numeric fields to tensors, keep strings as is
        for k, v in item.items():
            if not isinstance(v, str) and k != 'spec':
                item[k] = torch.as_tensor(v, dtype=self.dtype)

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        """
        Custom collate function for PyG graphs + other scalar/tensor fields.
        Each element of `batch` is a dict from __getitem__ above.
        """
        # 1) Collate the "spec" PyG data
        spec_list = [item["spec"] for item in batch]
        spec_batch = Batch.from_data_list(spec_list)

        # 2) Collate the "mol"
        mol_list = [item["mol"] for item in batch]
        # If they are Tensors, stack them. If they are strings, you can just keep a list.
        if isinstance(mol_list[0], torch.Tensor):
            mol_list = torch.stack(mol_list, dim=0)

        collated_batch = {
            "spec": spec_batch,
            "mol":  mol_list
        }

        # 3) Collate the other numeric/scalar fields with default_collate
        #    (But skip the keys we already handled or which do not exist)
        skip_keys = {"spec", "mol", "candidates", "labels", "candidates_smiles"}
        for k in batch[0].keys():
            if k not in skip_keys:
                collated_batch[k] = default_collate([item[k] for item in batch])

        return collated_batch

    def _parse_paths_from_df(self, df) -> List[Tuple[str, float, List[Tuple[List[float], matchms.Spectrum]], matchms.Spectrum]]:
        all_tree_paths = []

        # First, collect all precursor_mz values and root spectra
        all_precursor_mz = []
        for ms_level, precursor_mz, smile, identifier in zip(df["ms_level"], df["precursor_mz"], df["smiles"], df["identifier"]):
            if int(ms_level) == 2:
                root_spectrum = self.identifier_to_spectrum[identifier]
                all_precursor_mz.append(precursor_mz)
                all_tree_paths.append((smile, precursor_mz, [], root_spectrum))

        idx_cur_precursors_mz = -1
        for _, row in df.iterrows():
            ms_level = int(row["ms_level"])
            if ms_level == 2:
                idx_cur_precursors_mz += 1
                continue
            else:
                cur_path_group = all_tree_paths[idx_cur_precursors_mz][2]
                root_spectrum = all_tree_paths[idx_cur_precursors_mz][3]

                msn_precursor_mzs_str = row["msn_precursor_mzs"]

                if pd.isna(msn_precursor_mzs_str) or msn_precursor_mzs_str == 'None':
                    print(f"Skip if msn_precursor_mzs at index {_} is None or NaN")
                    continue
                try:
                    msn_precursor_mzs = ast.literal_eval(msn_precursor_mzs_str)
                except Exception as e:
                    print(f"Error parsing msn_precursor_mzs: {msn_precursor_mzs_str} at index {_}. Error: {e}")
                    continue
                if not isinstance(msn_precursor_mzs, list) or len(msn_precursor_mzs) == 0:
                    print(f"Skip if msn_precursor_mzs at index {_} is not a valid list")
                    continue

                # Replace the first element of msn_precursor_mzs with the root precursor_mz
                msn_precursor_mzs[0] = all_precursor_mz[idx_cur_precursors_mz]

                # Get the spectrum for this row
                identifier = row['identifier']
                spectrum = self.identifier_to_spectrum[identifier]

                # Get InChI of root spectrum and current spectrum
                root_inchi = root_spectrum.get('inchi') if root_spectrum else None
                spectrum_inchi = spectrum.get('inchi') if spectrum else None

                if root_inchi != spectrum_inchi:
                    continue

                # Append the path and spectrum as a tuple
                cur_path_group.append((msn_precursor_mzs, spectrum))

        return all_tree_paths

    def get_all_deviations(self) -> List[Tuple[str, float, float, float]]:
        """
        Aggregates all deviations from all trees into a single list.
        """
        all_deviations = []
        for tree in self.trees:
            all_deviations.extend(tree.deviations)
        return all_deviations

    def _generate_trees(self,
                        dataset_all_tree_paths: List[
                            Tuple[str, float, List[Tuple[List[float], matchms.Spectrum]], matchms.Spectrum]],
                        prune_missing_spectra: bool = False,
                        hierarchical_tree: bool = False,
                        cut_tree_at_level: Optional[int] = None
                        ) -> Tuple[List['Tree'], List['Data'], List[str]]:
        """
        Generates Tree and PyG Data objects from the dataset paths.

        Args:
            dataset_all_tree_paths (List[Tuple[str, float, List[Tuple[List[float], matchms.Spectrum]], matchms.Spectrum]]):
                List containing tuples of (SMILES, precursor_mz, paths, root_spectrum).
            prune_missing_spectra (bool): Whether to prune trees with missing spectra.
            hierarchical_tree (bool): Whether to maintain hierarchical edges or do bidirectional.
            cut_tree_at_level (Optional[int]): Maximum depth to retain in each tree.

        Returns:
            Tuple containing lists of Tree objects, PyG Data objects, and SMILES strings.
        """

        trees = []
        smiles = []
        pyg_trees = []

        for _, root_precursor_mz, paths, root_spectrum in dataset_all_tree_paths:
            tree = Tree(root_precursor_mz, spectrum=root_spectrum,
                        max_allowed_deviation=self.max_allowed_deviation)

            for path, spectrum in paths:
                tree.add_path_with_spectrum(path, spectrum)

            if prune_missing_spectra:
                tree.prune_missing_spectra()

            # Apply tree cutting if specified
            if cut_tree_at_level is not None:
                tree.cut_at_level(cut_tree_at_level)

            pyg_tree = tree.to_pyg_data(self.featurizer, hierarchical_tree)
            pyg_trees.append(pyg_tree)
            trees.append(tree)
            smi = tree.root.spectrum.get('smiles')
            smiles.append(smi)

        return trees, pyg_trees, smiles


    def _get_tree_depths(self, trees):
        return [tree.get_depth() for tree in trees]

    def _get_branching_factors(self, trees):
        return [tree.get_branching_factor() for tree in trees]

    def _compute_mol_feature(self, smi):
        if self.mol_transform:
            mol_feature = self.mol_transform(smi)

            if isinstance(mol_feature, np.ndarray):
                mol_feature = torch.as_tensor(mol_feature, dtype=self.dtype)
            elif isinstance(mol_feature, str):
                # If the output is a string, we can keep it as is
                pass
            else:
                # Handle other data types if necessary
                pass
            return mol_feature
        else:
            # If no mol_transform is provided, return the SMILES string
            return smi

# class MSnRetrievalDataset(MSnDataset):
#     """
#     Extension of MSnDataset that also loads a dictionary of candidate molecules
#     for each item['smiles'], so we can do retrieval tasks.
#
#     The collated batch includes:
#       - 'spec' : DataBatch of PyG
#       - 'mol'  : [batch_size, fp_size] or list if string-based
#       - 'candidates' : [sum_of_candidates_in_batch, fp_size]
#       - 'labels' : 1D bool of length sum_of_candidates_in_batch
#       - 'batch_ptr': #candidates per item, shape [batch_size]
#       - 'candidates_smiles': list of length sum_of_candidates_in_batch
#     """
#
#     def __init__(
#         self,
#         mol_label_transform: MolTransform = MolToInChIKey(),
#         candidates_pth: T.Optional[T.Union[Path, str]] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#
#         self.mol_label_transform = mol_label_transform
#
#         # Load the candidate SMILES from JSON
#         if candidates_pth is None:
#             self.candidates_pth = utils.hugging_face_download(
#                 "molecules/MassSpecGym_retrieval_candidates_mass.json"
#             )
#         elif candidates_pth == 'bonus':
#             self.candidates_pth = utils.hugging_face_download(
#                 "molecules/MassSpecGym_retrieval_candidates_formula.json"
#             )
#         else:
#             self.candidates_pth = Path(candidates_pth)
#
#         with open(self.candidates_pth, "r") as file:
#             all_candidates_dict = json.load(file)
#
#         # Filter out only those indices for which we have candidate SMILES
#         valid_indices = []
#         skipped = 0
#         for idx, smi in enumerate(self.smiles):
#             if smi in all_candidates_dict:
#                 valid_indices.append(idx)
#             else:
#                 skipped += 1
#         print(f"Warning: No candidates for {skipped} SMILES. Skipping them.")
#
#         self.valid_indices = valid_indices
#         self.candidates_dict = all_candidates_dict
#
#         # Re‑map root_identifier_to_index so that only valid indices remain
#         new_map = {}
#         for new_i, old_i in enumerate(valid_indices):
#             rid = self.trees[old_i].root.spectrum.get('identifier')
#             new_map[rid] = new_i
#         self.root_identifier_to_index = new_map
#
#         print(f"Total valid indices: {len(self.valid_indices)}")
#         print(f"MSnRetrievalDataset length: {len(self)}")
#
#     def __len__(self):
#         return len(self.valid_indices)
#
#     @profile_function
#     def __getitem__(self, idx: int) -> dict:
#         # Map to the "true" index in the underlying MSnDataset
#         real_idx = self.valid_indices[idx]
#
#         # Use MSnDataset's __getitem__ to get spec, mol, etc.
#         item = super().__getitem__(real_idx)
#
#         # This is the "true" SMILES we had at that index
#         smi = self.smiles[real_idx]
#         item["smiles"] = smi
#
#         # Build the candidate list
#         candidates_smi = self.candidates_dict[smi]
#         item["candidates_smiles"] = candidates_smi
#
#         # labels: True if candidate is the same as the query
#         # We do this by matching InChIKey (or something) but here: mol_label_transform
#         item_label = self.mol_label_transform(smi)
#         item["labels"] = [
#             (self.mol_label_transform(c_smi) == item_label)
#             for c_smi in candidates_smi
#         ]
#         if not any(item["labels"]):
#             raise ValueError(f"Query molecule not in candidates for {smi}.")
#
#         # Transform the *query* molecule again if needed (like your fingerprint)
#         if self.mol_transform:
#             query_fp = self.mol_transform(smi)
#             if isinstance(query_fp, np.ndarray):
#                 query_fp = torch.as_tensor(query_fp, dtype=self.dtype)
#             item["mol"] = query_fp
#
#             # Transform each candidate
#             candidates_fp = []
#             for c_smi in candidates_smi:
#                 out = self.mol_transform(c_smi)
#                 if isinstance(out, np.ndarray):
#                     out = torch.as_tensor(out, dtype=self.dtype)
#                 candidates_fp.append(out)
#             item["candidates"] = candidates_fp
#         else:
#             # If no transform, store as plain strings or something
#             item["candidates"] = candidates_smi
#
#         return item
#     @profile_function
#     def __getitem__(self, idx: int) -> dict:
#         t0 = time.perf_counter()
#         # Map to the "true" index in the underlying MSnDataset
#         real_idx = self.valid_indices[idx]
#         t1 = time.perf_counter()
#
#         # Get base item from the parent class
#         item = super().__getitem__(real_idx)
#         t2 = time.perf_counter()
#
#         # Retrieve the true SMILES for this index
#         smi = self.smiles[real_idx]
#         t3 = time.perf_counter()
#         item["smiles"] = smi
#         t4 = time.perf_counter()
#
#         # Build the candidate list from the candidates dictionary
#         candidates_smi = self.candidates_dict[smi]
#         t5 = time.perf_counter()
#         item["candidates_smiles"] = candidates_smi
#         t6 = time.perf_counter()
#
#         # Compute query label
#         item_label = self.mol_label_transform(smi)
#         t7 = time.perf_counter()
#
#         # Build candidate labels list
#         candidate_labels = []
#         for c_smi in candidates_smi:
#             t_label0 = time.perf_counter()
#             candidate_labels.append(self.mol_label_transform(c_smi) == item_label)
#             t_label1 = time.perf_counter()
#             # Print each candidate label transform timing if desired:
#             # print(f"Candidate label transform: {(t_label1-t_label0)*1000:.2f} ms")
#         t8 = time.perf_counter()
#         item["labels"] = candidate_labels
#         if not any(item["labels"]):
#             raise ValueError(f"Query molecule {smi} not found in candidates.")
#         t9 = time.perf_counter()
#
#         # If a molecule transform is provided, transform query and candidates
#         if self.mol_transform:
#             # Transform query molecule
#             t_query0 = time.perf_counter()
#             query_fp = self.mol_transform(smi)
#             t_query1 = time.perf_counter()
#             if isinstance(query_fp, np.ndarray):
#                 query_fp = torch.as_tensor(query_fp, dtype=self.dtype)
#             t_query2 = time.perf_counter()
#             item["mol"] = query_fp
#             t_query3 = time.perf_counter()
#
#             # Transform each candidate in a loop
#             candidates_fp = []
#             t_candidates_start = time.perf_counter()
#             candidate_loop_times = []
#             for c_smi in candidates_smi:
#                 t_iter0 = time.perf_counter()
#                 out = self.mol_transform(c_smi)
#                 t_iter1 = time.perf_counter()
#                 if isinstance(out, np.ndarray):
#                     out = torch.as_tensor(out, dtype=self.dtype)
#                 t_iter2 = time.perf_counter()
#                 candidates_fp.append(out)
#                 candidate_loop_times.append((t_iter1 - t_iter0, t_iter2 - t_iter1))
#             t_candidates_end = time.perf_counter()
#             item["candidates"] = candidates_fp
#             t_candidates_total = t_candidates_end - t_candidates_start
#
#             # Print detailed candidate loop timing summary:
#             avg_transform = sum(x for x, _ in candidate_loop_times) / len(candidate_loop_times)
#             avg_tensor = sum(y for _, y in candidate_loop_times) / len(candidate_loop_times)
#             print(f"Candidate loop total: {t_candidates_total*1000:.2f} ms, "
#                   f"avg transform: {avg_transform*1000:.2f} ms, avg tensor conversion: {avg_tensor*1000:.2f} ms")
#             t_final = time.perf_counter()
#         else:
#             item["candidates"] = candidates_smi
#             t_final = time.perf_counter()
#
#         # Print overall breakdown for __getitem__
#         print(f"__getitem__ timing breakdown for idx {idx}:")
#         print(f"  Map valid index: {(t1-t0)*1000:.2f} ms")
#         print(f"  Parent __getitem__: {(t2-t1)*1000:.2f} ms")
#         print(f"  Retrieve smi: {(t3-t2)*1000:.2f} ms")
#         print(f"  Set smi in item: {(t4-t3)*1000:.2f} ms")
#         print(f"  Candidate lookup: {(t5-t4)*1000:.2f} ms")
#         print(f"  Set candidates_smiles: {(t6-t5)*1000:.2f} ms")
#         print(f"  Mol label transform (query): {(t7-t6)*1000:.2f} ms")
#         print(f"  Build candidate labels: {(t8-t7)*1000:.2f} ms")
#         print(f"  Label check: {(t9-t8)*1000:.2f} ms")
#         if self.mol_transform:
#             print(f"  Mol transform on query: {(t_query1-t_query0)*1000:.2f} ms")
#             print(f"  Tensor conversion on query: {(t_query2-t_query1)*1000:.2f} ms")
#             print(f"  Set query mol: {(t_query3-t_query2)*1000:.2f} ms")
#             print(f"  Candidate loop total: {t_candidates_total*1000:.2f} ms")
#         print(f"  Total __getitem__: {(t_final-t0)*1000:.2f} ms")
#         return item



    # @staticmethod
    # @profile_function
    # def collate_fn(batch: T.Iterable[dict]) -> dict:
    #     """
    #     Collate a batch of retrieval data:
    #       - spec -> single DataBatch
    #       - mol  -> stacked if Tensors
    #       - candidates -> stacked (2D)
    #       - labels -> 1D flatten
    #       - batch_ptr -> #candidates per item
    #       - candidates_smiles -> big list
    #     """
    #     # 1) Collate the PyG specs
    #     spec_list = [d["spec"] for d in batch]
    #     spec_batch = Batch.from_data_list(spec_list)
    #
    #     # 2) Collate the 'mol'
    #     mol_list = [d["mol"] for d in batch]
    #     # If these are Tensors, stack them
    #     if isinstance(mol_list[0], torch.Tensor):
    #         mol_list = torch.stack(mol_list, dim=0)
    #
    #     collated_batch = {
    #         "spec": spec_batch,
    #         "mol":  mol_list
    #     }
    #
    #     # 3) Collate numeric/scalar metadata with default_collate
    #     skip_keys = {"spec", "mol", "candidates", "labels", "candidates_smiles"}
    #     for k in batch[0].keys():
    #         if k not in skip_keys:
    #             collated_batch[k] = default_collate([item[k] for item in batch])
    #
    #     # 4) Flatten the candidates
    #     all_candidates = []
    #     for d in batch:
    #         # each d["candidates"] is a list of Tensors or strings
    #         # If it's a Tensor, shape might be [fp_size]
    #         # We'll gather them into all_candidates
    #         for c in d["candidates"]:
    #             if isinstance(c, torch.Tensor):
    #                 all_candidates.append(c)
    #             else:
    #                 # If string, store them as well (unusual for retrieval though)
    #                 all_candidates.append(c)
    #
    #     # If your candidates are Tensors, do a stack
    #     if isinstance(all_candidates[0], torch.Tensor):
    #         all_candidates = torch.stack(all_candidates, dim=0)
    #
    #     collated_batch["candidates"] = all_candidates
    #
    #     # 5) Flatten the labels
    #     # item["labels"] is a python list of bool for each candidate
    #     all_labels = sum([d["labels"] for d in batch], start=[])
    #     collated_batch["labels"] = torch.as_tensor(all_labels, dtype=torch.bool)
    #
    #     # 6) batch_ptr: how many candidates per item
    #     lens = [len(d["candidates"]) for d in batch]
    #     collated_batch["batch_ptr"] = torch.as_tensor(lens, dtype=torch.int)
    #
    #     # 7) Flatten candidates_smiles
    #     all_cand_smi = sum([d["candidates_smiles"] for d in batch], start=[])
    #     collated_batch["candidates_smiles"] = all_cand_smi
    #
    #     return collated_batch





from torch.utils.data.dataloader import default_collate
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
from tqdm import tqdm


from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Use file-system sharing strategy (if needed)
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

def _precompute_for_index(idx: int,
                          smiles: T.List[str],
                          candidates_dict: dict,
                          mol_transform: T.Callable,
                          mol_label_transform: T.Callable,
                          dtype: torch.dtype):
    """
    Precompute candidate transformations and labels for one valid index.
    Returns a tuple (idx, precomputed_dict).
    """
    smi = smiles[idx]
    query_label = mol_label_transform(smi)
    # Transform query molecule
    if mol_transform:
        query_fp = mol_transform(smi)
        if isinstance(query_fp, np.ndarray):
            query_fp = torch.as_tensor(query_fp, dtype=dtype)
    else:
        query_fp = smi  # fallback if no transform provided
    # Retrieve candidate SMILES.
    candidates_smi = candidates_dict[smi]
    # Transform each candidate.
    candidate_transformed = []
    if mol_transform:
        for c_smi in candidates_smi:
            out = mol_transform(c_smi)
            if isinstance(out, np.ndarray):
                out = torch.as_tensor(out, dtype=dtype)
            candidate_transformed.append(out)
    else:
        candidate_transformed = candidates_smi
    # Build candidate labels.
    candidate_labels = [mol_label_transform(c_smi) == query_label for c_smi in candidates_smi]
    if not any(candidate_labels):
        raise ValueError(f"Query molecule {smi} not found among its candidates during precomputation.")
    return idx, {
        "mol": query_fp,
        "candidates": candidate_transformed,
        "labels": candidate_labels,
        "candidates_smiles": candidates_smi
    }

def precompute_wrapper(args):
    """Module-level wrapper to unpack arguments."""
    return _precompute_for_index(*args)


# ---------------- MSnRetrievalDataset Definition ---------------- #

class MSnRetrievalDataset(MSnDataset):
    """
    Extension of MSnDataset that also loads a dictionary of candidate molecules for each item['smiles']
    so that retrieval tasks can be performed.

    For each valid index (i.e. where candidate SMILES exist), we precompute:
      - "mol": the transformed query molecule (e.g. its fingerprint)
      - "candidates": a list of transformed candidate representations
      - "labels": a list of booleans indicating whether each candidate matches the query
      - "candidates_smiles": the original candidate SMILES list

    The collated batch will include:
      - 'spec' : DataBatch of PyG graphs
      - 'mol'  : tensor of shape [batch_size, fp_size]
      - 'candidates' : tensor of shape [sum(candidates per item), fp_size]
      - 'labels' : 1D tensor of booleans for all candidates
      - 'batch_ptr': tensor of candidate counts per item
      - 'candidates_smiles': list of candidate SMILES
    """
    def __init__(
        self,
        mol_label_transform: T.Callable = MolToInChIKey(),
        candidates_pth: T.Optional[T.Union[Path, str]] = None,
        cache_path: T.Optional[T.Union[Path, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mol_label_transform = mol_label_transform

        # Load candidate SMILES from JSON.
        if candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_mass.json")
        elif candidates_pth == 'bonus':
            self.candidates_pth = utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json")
        else:
            self.candidates_pth = Path(candidates_pth)
        with open(self.candidates_pth, "r") as f:
            self.candidates_dict = json.load(f)

        # Filter valid indices: keep only indices for which candidate SMILES exist.
        valid_indices = []
        for idx, smi in enumerate(self.smiles):
            if smi in self.candidates_dict:
                valid_indices.append(idx)
            else:
                print(f"Warning: No candidates for SMILES {smi} (index {idx}); skipping.")
        self.valid_indices = valid_indices

        # Re-map root_identifier_to_index for valid indices.
        new_map = {}
        for new_i, old_i in enumerate(valid_indices):
            rid = self.trees[old_i].root.spectrum.get('identifier')
            new_map[rid] = new_i
        self.root_identifier_to_index = new_map

        print(f"Total valid indices: {len(self.valid_indices)}")
        print(f"MSnRetrievalDataset length: {len(self)}")

        # Set up cache file path.
        if cache_path is None:
            self.cache_path = Path(self.candidates_pth).with_name("msnretrieval_precomputed.pkl")
        else:
            self.cache_path = Path(cache_path)

        # Precompute candidate-side transformations.
        if self.cache_path.exists():
            print(f"Loading precomputed data from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.precomputed = pickle.load(f)
        else:
            print("Precomputing candidate-side transformations using multiprocessing with chunking...")
            self.precomputed = {}
            # Determine number of worker processes.
            allocated_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
            num_workers = max(allocated_cpus - 2, 1)
            print(f"Using {num_workers} processes for precomputation.")
            # Prepare the list of valid indices.
            all_indices = self.valid_indices
            total = len(all_indices)
            chunk_size = 1000  # Adjust this chunk size as needed.
            # Process indices in chunks.
            for i in tqdm(range(0, total, chunk_size), desc="Precomputing in chunks", total=(total + chunk_size - 1) // chunk_size):
                chunk_indices = all_indices[i:i+chunk_size]
                # Prepare argument tuples for each index in the chunk.
                args_list = [
                    (idx, self.smiles, self.candidates_dict, self.mol_transform,
                     self.mol_label_transform, self.dtype)
                    for idx in chunk_indices
                ]
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Use precompute_wrapper (no lambda) with a specified chunksize.
                    results = list(executor.map(precompute_wrapper, args_list, chunksize=10))
                for idx_result, precomputed in results:
                    self.precomputed[idx_result] = precomputed
                # Save intermediate cache to disk after each chunk.
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.precomputed, f)
            print(f"Precomputation complete for {len(self.precomputed)} items.")

    def __len__(self):
        return len(self.valid_indices)

    @profile_function
    def __getitem__(self, idx: int) -> dict:
        real_idx = self.valid_indices[idx]
        item = super().__getitem__(real_idx)
        smi = self.smiles[real_idx]
        item["smiles"] = smi
        item["candidates_smiles"] = self.candidates_dict[smi]
        pre = self.precomputed[real_idx]
        item["mol"] = pre["mol"]
        item["candidates"] = pre["candidates"]
        item["labels"] = pre["labels"]
        return item

    @staticmethod
    @profile_function
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        collated_batch = {}
        # Collate the PyG graphs.
        spec_list = [item["spec"] for item in batch]
        spec_batch = Batch.from_data_list(spec_list)
        collated_batch["spec"] = spec_batch
        # Collate query molecule representations.
        mol_list = [item["mol"] for item in batch]
        if isinstance(mol_list[0], torch.Tensor):
            mol_list = torch.stack(mol_list, dim=0)
        collated_batch["mol"] = mol_list
        # Collate any additional scalar fields.
        for k in batch[0].keys():
            if k not in {"spec", "mol", "candidates", "labels", "candidates_smiles"}:
                collated_batch[k] = default_collate([item[k] for item in batch])
        # Flatten candidate representations.
        all_candidates = []
        for item in batch:
            for cand in item["candidates"]:
                all_candidates.append(cand)
        if isinstance(all_candidates[0], torch.Tensor):
            all_candidates = torch.stack(all_candidates, dim=0)
        collated_batch["candidates"] = all_candidates
        # Flatten candidate labels.
        all_labels = sum([item["labels"] for item in batch], start=[])
        collated_batch["labels"] = torch.as_tensor(all_labels, dtype=torch.bool)
        # Build batch_ptr (number of candidates per item).
        batch_ptr = [len(item["candidates"]) for item in batch]
        collated_batch["batch_ptr"] = torch.as_tensor(batch_ptr, dtype=torch.int)
        # Concatenate candidate SMILES.
        all_cand_smiles = sum([item["candidates_smiles"] for item in batch], start=[])
        collated_batch["candidates_smiles"] = all_cand_smiles
        return collated_batch

# mp.set_sharing_strategy('file_system')
#
# def _precompute_for_index(args):
#     """
#     Precompute candidate transformations and labels for a single valid index.
#     Returns a tuple (idx, precomputed_dict).
#     """
#     idx, smiles, candidates_dict, mol_transform, mol_label_transform, dtype = args
#     smi = smiles[idx]
#     query_label = mol_label_transform(smi)
#     # Compute the transformed query fingerprint.
#     if mol_transform:
#         query_fp = mol_transform(smi)
#         if isinstance(query_fp, np.ndarray):
#             query_fp = torch.as_tensor(query_fp, dtype=dtype)
#     else:
#         query_fp = smi  # fallback
#     # Retrieve candidate SMILES.
#     candidates_smi = candidates_dict[smi]
#     # Transform each candidate.
#     candidate_transformed = []
#     if mol_transform:
#         for c_smi in candidates_smi:
#             out = mol_transform(c_smi)
#             if isinstance(out, np.ndarray):
#                 out = torch.as_tensor(out, dtype=dtype)
#             candidate_transformed.append(out)
#     else:
#         candidate_transformed = candidates_smi
#     # Compute candidate labels.
#     candidate_labels = [mol_label_transform(c_smi) == query_label for c_smi in candidates_smi]
#     if not any(candidate_labels):
#         raise ValueError(f"Query molecule {smi} not found among its candidates during precomputation.")
#     return idx, {
#         "mol": query_fp,
#         "candidates": candidate_transformed,
#         "labels": candidate_labels,
#         "candidates_smiles": candidates_smi
#     }
#
# class MSnRetrievalDataset(MSnDataset):
#     """
#     Extension of MSnDataset that also loads a dictionary of candidate molecules for each item['smiles']
#     so that we can perform retrieval tasks.
#
#     For each valid index (where candidate SMILES exist), we precompute:
#       - "mol": transformed query molecule (e.g. its fingerprint)
#       - "candidates": list of transformed candidate representations
#       - "labels": list of booleans indicating whether each candidate matches the query
#       - "candidates_smiles": the original candidate SMILES list
#
#     The collated batch will include:
#       - 'spec': PyG DataBatch of graphs
#       - 'mol': tensor of shape [batch_size, fp_size]
#       - 'candidates': tensor of shape [sum(candidates per item), fp_size]
#       - 'labels': 1D tensor of booleans for all candidates
#       - 'batch_ptr': tensor of candidate counts per item
#       - 'candidates_smiles': list of candidate SMILES
#     """
#     def __init__(
#         self,
#         mol_label_transform: T.Callable = MolToInChIKey(),
#         candidates_pth: T.Optional[T.Union[Path, str]] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.mol_label_transform = mol_label_transform
#
#         # Load candidate SMILES from JSON.
#         if candidates_pth is None:
#             self.candidates_pth = utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_mass.json")
#         elif candidates_pth == 'bonus':
#             self.candidates_pth = utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json")
#         else:
#             self.candidates_pth = Path(candidates_pth)
#         with open(self.candidates_pth, "r") as f:
#             self.candidates_dict = json.load(f)
#
#         # Filter valid indices: keep only those indices for which candidate SMILES exist.
#         valid_indices = []
#         for idx, smi in enumerate(self.smiles):
#             if smi in self.candidates_dict:
#                 valid_indices.append(idx)
#             else:
#                 print(f"Warning: No candidates for SMILES {smi} (index {idx}); skipping.")
#         self.valid_indices = valid_indices
#
#         # Re-map root_identifier_to_index for valid indices.
#         new_map = {}
#         for new_i, old_i in enumerate(valid_indices):
#             rid = self.trees[old_i].root.spectrum.get('identifier')
#             new_map[rid] = new_i
#         self.root_identifier_to_index = new_map
#
#         print(f"Total valid indices: {len(self.valid_indices)}")
#         print(f"MSnRetrievalDataset length: {len(self)}")
#
#         # Precompute candidate-side transformations in parallel.
#         self.precomputed = {}
#         # Get the number of allocated CPUs from SLURM if available; otherwise use os.cpu_count()
#         allocated_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
#         # Use half of the allocated CPUs (at least 1) to avoid overcommitting.
#         num_workers = max(min(allocated_cpus - 2, len(self.valid_indices)), 1)
#         print(f"Using {num_workers} processes for precomputation.")
#
#         # Prepare a list of arguments for each valid index.
#         args_list = []
#         for idx in self.valid_indices:
#             args_list.append((idx, self.smiles, self.candidates_dict, self.mol_transform, self.mol_label_transform, self.dtype))
#         # Use a multiprocessing pool with a limited number of processes.
#         with mp.Pool(processes=num_workers, maxtasksperchild=10) as pool:
#             results = pool.map(_precompute_for_index, args_list)
#         for idx, pre in results:
#             self.precomputed[idx] = pre
#
#     def __len__(self):
#         return len(self.valid_indices)
#
#     @profile_function
#     def __getitem__(self, idx: int) -> dict:
#         # Map to the actual index in the underlying dataset.
#         real_idx = self.valid_indices[idx]
#         # Get the base item (e.g. 'spec', 'precursor_mz', etc.) from the parent.
#         item = super().__getitem__(real_idx)
#         # Set the query SMILES.
#         smi = self.smiles[real_idx]
#         item["smiles"] = smi
#         # Also include the original candidate SMILES.
#         item["candidates_smiles"] = self.candidates_dict[smi]
#         # Retrieve precomputed values.
#         pre = self.precomputed[real_idx]
#         item["mol"] = pre["mol"]
#         item["candidates"] = pre["candidates"]
#         item["labels"] = pre["labels"]
#         return item
#
#     @staticmethod
#     @profile_function
#     def collate_fn(batch: T.Iterable[dict]) -> dict:
#         collated_batch = {}
#         # Collate the PyG graphs.
#         spec_list = [item["spec"] for item in batch]
#         spec_batch = Batch.from_data_list(spec_list)
#         collated_batch["spec"] = spec_batch
#
#         # Collate the transformed query molecules.
#         mol_list = [item["mol"] for item in batch]
#         if isinstance(mol_list[0], torch.Tensor):
#             mol_list = torch.stack(mol_list, dim=0)
#         collated_batch["mol"] = mol_list
#
#         # Collate any additional scalar fields.
#         for k in batch[0].keys():
#             if k not in {"spec", "mol", "candidates", "labels", "candidates_smiles"}:
#                 collated_batch[k] = default_collate([item[k] for item in batch])
#
#         # Flatten candidates.
#         all_candidates = []
#         for item in batch:
#             for cand in item["candidates"]:
#                 all_candidates.append(cand)
#         if isinstance(all_candidates[0], torch.Tensor):
#             all_candidates = torch.stack(all_candidates, dim=0)
#         collated_batch["candidates"] = all_candidates
#
#         # Flatten candidate labels.
#         all_labels = sum([item["labels"] for item in batch], start=[])
#         collated_batch["labels"] = torch.as_tensor(all_labels, dtype=torch.bool)
#
#         # Build batch_ptr (number of candidates per item).
#         batch_ptr = [len(item["candidates"]) for item in batch]
#         collated_batch["batch_ptr"] = torch.as_tensor(batch_ptr, dtype=torch.int)
#
#         # Concatenate candidate SMILES.
#         all_cand_smiles = sum([item["candidates_smiles"] for item in batch], start=[])
#         collated_batch["candidates_smiles"] = all_cand_smiles
#
#         return collated_batch





# mp.set_sharing_strategy('file_system')
# def _precompute_for_index(args):
#     """
#     Precompute candidate transformations and labels for a single valid index.
#
#     Parameters:
#       - args: a tuple containing:
#           idx: the index (into self.smiles, etc.)
#           smiles: the list of SMILES (from self.smiles)
#           candidates_dict: dict mapping SMILES to candidate SMILES
#           mol_transform: function to transform a SMILES into a molecular representation
#           mol_label_transform: function to produce a label (e.g. InChIKey) from a SMILES
#           dtype: torch.dtype for tensor conversion
#     Returns:
#       A tuple (idx, precomputed_dict) where precomputed_dict contains:
#         "mol": the transformed query representation,
#         "candidates": a list of transformed candidate representations,
#         "labels": a list of booleans for each candidate,
#         "candidates_smiles": the original candidate SMILES list.
#     """
#     idx, smiles, candidates_dict, mol_transform, mol_label_transform, dtype = args
#     smi = smiles[idx]
#     query_label = mol_label_transform(smi)
#     # Compute the transformed query fingerprint.
#     if mol_transform:
#         query_fp = mol_transform(smi)
#         if isinstance(query_fp, np.ndarray):
#             query_fp = torch.as_tensor(query_fp, dtype=dtype)
#     else:
#         query_fp = smi  # fallback
#     # Retrieve candidate SMILES.
#     candidates_smi = candidates_dict[smi]
#     # Transform each candidate.
#     candidate_transformed = []
#     if mol_transform:
#         for c_smi in candidates_smi:
#             out = mol_transform(c_smi)
#             if isinstance(out, np.ndarray):
#                 out = torch.as_tensor(out, dtype=dtype)
#             candidate_transformed.append(out)
#     else:
#         candidate_transformed = candidates_smi
#     # Compute candidate labels.
#     candidate_labels = []
#     for c_smi in candidates_smi:
#         candidate_labels.append(mol_label_transform(c_smi) == query_label)
#     if not any(candidate_labels):
#         raise ValueError(f"Query molecule {smi} not found among its candidates during precomputation.")
#     return idx, {
#         "mol": query_fp,
#         "candidates": candidate_transformed,
#         "labels": candidate_labels,
#         "candidates_smiles": candidates_smi
#     }
#
#
#
# class MSnRetrievalDataset(MSnDataset):
#     """
#     Extension of MSnDataset that also loads a dictionary of candidate molecules for each item['smiles']
#     so we can perform retrieval tasks.
#
#     For each valid index (i.e. where candidate SMILES exist), the following are precomputed:
#       - "mol": the transformed query molecule (e.g. its fingerprint)
#       - "candidates": a list of transformed candidate representations
#       - "labels": a list of booleans indicating if each candidate matches the query
#       - "candidates_smiles": the original candidate SMILES list
#
#     The collated batch will include:
#       - 'spec' : DataBatch of PyG graphs
#       - 'mol'  : tensor of shape [batch_size, fp_size]
#       - 'candidates' : tensor of shape [sum(candidates per item), fp_size]
#       - 'labels' : 1D tensor of booleans for all candidates
#       - 'batch_ptr': tensor of candidate counts per item
#       - 'candidates_smiles': list of candidate SMILES
#     """
#     def __init__(
#         self,
#         mol_label_transform: T.Callable = MolToInChIKey(),
#         candidates_pth: T.Optional[T.Union[Path, str]] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.mol_label_transform = mol_label_transform
#
#         # Load candidate SMILES from JSON.
#         if candidates_pth is None:
#             self.candidates_pth = utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_mass.json")
#         elif candidates_pth == 'bonus':
#             self.candidates_pth = utils.hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json")
#         else:
#             self.candidates_pth = Path(candidates_pth)
#         with open(self.candidates_pth, "r") as f:
#             self.candidates_dict = json.load(f)
#
#         # Filter valid indices: only keep indices for which candidate SMILES exist.
#         valid_indices = []
#         for idx, smi in enumerate(self.smiles):
#             if smi in self.candidates_dict:
#                 valid_indices.append(idx)
#             else:
#                 print(f"Warning: No candidates for SMILES {smi} (index {idx}); skipping.")
#         self.valid_indices = valid_indices
#
#         # Re-map root_identifier_to_index for valid indices.
#         new_map = {}
#         for new_i, old_i in enumerate(valid_indices):
#             rid = self.trees[old_i].root.spectrum.get('identifier')
#             new_map[rid] = new_i
#         self.root_identifier_to_index = new_map
#
#         print(f"Total valid indices: {len(self.valid_indices)}")
#         print(f"MSnRetrievalDataset length: {len(self)}")
#
#         # Precompute candidate-side transformations in parallel.
#         self.precomputed = {}
#         allocated_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
#         num_workers = max(allocated_cpus - 2, 1)
#         print(f"Using {num_workers} processes for precomputation.")
#
#         # Prepare arguments for each valid index.
#         args_list = []
#         for idx in self.valid_indices:
#             args_list.append((idx, self.smiles, self.candidates_dict, self.mol_transform, self.mol_label_transform, self.dtype))
#         with multiprocessing.Pool(processes=num_workers) as pool:
#             results = pool.map(_precompute_for_index, args_list)
#         # Store precomputed results in a dictionary keyed by the valid index.
#         for idx, pre in results:
#             self.precomputed[idx] = pre
#
#     def __len__(self):
#         return len(self.valid_indices)
#
#     @profile_function
#     def __getitem__(self, idx: int) -> dict:
#         # Map to the actual index in the underlying dataset.
#         real_idx = self.valid_indices[idx]
#         # Get the base item (e.g., 'spec', 'precursor_mz', etc.) from the parent.
#         item = super().__getitem__(real_idx)
#         # Set the query SMILES.
#         smi = self.smiles[real_idx]
#         item["smiles"] = smi
#         # Also include the original candidate SMILES.
#         item["candidates_smiles"] = self.candidates_dict[smi]
#         # Retrieve precomputed values.
#         pre = self.precomputed[real_idx]
#         item["mol"] = pre["mol"]
#         item["candidates"] = pre["candidates"]
#         item["labels"] = pre["labels"]
#         return item
#
#     @staticmethod
#     @profile_function
#     def collate_fn(batch: T.Iterable[dict]) -> dict:
#         collated_batch = {}
#         # Collate the PyG graphs.
#         spec_list = [item["spec"] for item in batch]
#         spec_batch = Batch.from_data_list(spec_list)
#         collated_batch["spec"] = spec_batch
#
#         # Collate the transformed query molecules.
#         mol_list = [item["mol"] for item in batch]
#         if isinstance(mol_list[0], torch.Tensor):
#             mol_list = torch.stack(mol_list, dim=0)
#         collated_batch["mol"] = mol_list
#
#         # Collate additional scalar fields.
#         for k in batch[0].keys():
#             if k not in {"spec", "mol", "candidates", "labels", "candidates_smiles"}:
#                 collated_batch[k] = default_collate([item[k] for item in batch])
#
#         # For candidates, flatten the lists and stack if tensors.
#         all_candidates = []
#         for item in batch:
#             for cand in item["candidates"]:
#                 all_candidates.append(cand)
#         if isinstance(all_candidates[0], torch.Tensor):
#             all_candidates = torch.stack(all_candidates, dim=0)
#         collated_batch["candidates"] = all_candidates
#
#         # Flatten candidate labels.
#         all_labels = sum([item["labels"] for item in batch], start=[])
#         collated_batch["labels"] = torch.as_tensor(all_labels, dtype=torch.bool)
#
#         # Build batch_ptr (number of candidates per item).
#         batch_ptr = [len(item["candidates"]) for item in batch]
#         collated_batch["batch_ptr"] = torch.as_tensor(batch_ptr, dtype=torch.int)
#
#         # Concatenate candidate SMILES.
#         all_cand_smiles = sum([item["candidates_smiles"] for item in batch], start=[])
#         collated_batch["candidates_smiles"] = all_cand_smiles
#
#         return collated_batch








    # @profile_function
    # def __getitem__(self, idx: int) -> dict:
    #     t0 = time.perf_counter()
    #     # Map to the actual index.
    #     real_idx = self.valid_indices[idx]
    #     t1 = time.perf_counter()
    #     # Get base item from parent.
    #     item = super().__getitem__(real_idx)
    #     t2 = time.perf_counter()
    #     # Retrieve the true SMILES.
    #     smi = self.smiles[real_idx]
    #     t3 = time.perf_counter()
    #     item["smiles"] = smi
    #     t4 = time.perf_counter()
    #     # Include the original candidate SMILES.
    #     item["candidates_smiles"] = self.candidates_dict[smi]
    #     t5 = time.perf_counter()
    #     # Retrieve precomputed values.
    #     pre = self.precomputed[real_idx]
    #     t6 = time.perf_counter()
    #     item["mol"] = pre["mol"]
    #     t7 = time.perf_counter()
    #     item["candidates"] = pre["candidates"]
    #     t8 = time.perf_counter()
    #     item["labels"] = pre["labels"]
    #     t9 = time.perf_counter()
    #     print(f"__getitem__ timing breakdown for idx {idx}:")
    #     print(f"  Map valid index: {(t1-t0)*1000:.2f} ms")
    #     print(f"  Parent __getitem__: {(t2-t1)*1000:.2f} ms")
    #     print(f"  Retrieve SMILES: {(t3-t2)*1000:.2f} ms")
    #     print(f"  Set SMILES: {(t4-t3)*1000:.2f} ms")
    #     print(f"  Set candidates_smiles: {(t5-t4)*1000:.2f} ms")
    #     print(f"  Precomputed lookup: {(t6-t5)*1000:.2f} ms")
    #     print(f"  Set query mol: {(t7-t6)*1000:.2f} ms")
    #     print(f"  Set candidates: {(t8-t7)*1000:.2f} ms")
    #     print(f"  Set labels: {(t9-t8)*1000:.2f} ms")
    #     print(f"  Total __getitem__: {(t9-t0)*1000:.2f} ms")
    #     return item
    #
    # @staticmethod
    # @profile_function
    # def collate_fn(batch: T.Iterable[dict]) -> dict:
    #     t0 = time.perf_counter()
    #     collated_batch = {}
    #     t1 = time.perf_counter()
    #     # Collate the PyG specs.
    #     spec_list = [item["spec"] for item in batch]
    #     t2 = time.perf_counter()
    #     spec_batch = Batch.from_data_list(spec_list)
    #     t3 = time.perf_counter()
    #     collated_batch["spec"] = spec_batch
    #     t4 = time.perf_counter()
    #     # Collate query molecules.
    #     mol_list = [item["mol"] for item in batch]
    #     t5 = time.perf_counter()
    #     if isinstance(mol_list[0], torch.Tensor):
    #         mol_list = torch.stack(mol_list, dim=0)
    #     t6 = time.perf_counter()
    #     collated_batch["mol"] = mol_list
    #     t7 = time.perf_counter()
    #     # Collate any additional scalar fields.
    #     for k in batch[0].keys():
    #         if k not in {"spec", "mol", "candidates", "labels", "candidates_smiles"}:
    #             collated_batch[k] = default_collate([item[k] for item in batch])
    #     t8 = time.perf_counter()
    #     # Flatten candidates.
    #     all_candidates = []
    #     for item in batch:
    #         for cand in item["candidates"]:
    #             all_candidates.append(cand)
    #     t9 = time.perf_counter()
    #     if isinstance(all_candidates[0], torch.Tensor):
    #         all_candidates = torch.stack(all_candidates, dim=0)
    #     t10 = time.perf_counter()
    #     collated_batch["candidates"] = all_candidates
    #     t11 = time.perf_counter()
    #     # Flatten candidate labels.
    #     all_labels = sum([item["labels"] for item in batch], start=[])
    #     t12 = time.perf_counter()
    #     collated_batch["labels"] = torch.as_tensor(all_labels, dtype=torch.bool)
    #     t13 = time.perf_counter()
    #     # Build batch_ptr (number of candidates per item).
    #     batch_ptr = [len(item["candidates"]) for item in batch]
    #     t14 = time.perf_counter()
    #     collated_batch["batch_ptr"] = torch.as_tensor(batch_ptr, dtype=torch.int)
    #     t15 = time.perf_counter()
    #     # Concatenate candidate SMILES.
    #     all_cand_smiles = sum([item["candidates_smiles"] for item in batch], start=[])
    #     t16 = time.perf_counter()
    #     collated_batch["candidates_smiles"] = all_cand_smiles
    #     t17 = time.perf_counter()
    #     print("collate_fn timing breakdown:")
    #     print(f"  Init collated batch: {(t1-t0)*1000:.2f} ms")
    #     print(f"  Build spec list: {(t2-t1)*1000:.2f} ms")
    #     print(f"  PyG Batch creation: {(t3-t2)*1000:.2f} ms")
    #     print(f"  Set spec: {(t4-t3)*1000:.2f} ms")
    #     print(f"  Build mol list: {(t5-t4)*1000:.2f} ms")
    #     print(f"  Stack mol: {(t6-t5)*1000:.2f} ms")
    #     print(f"  Set mol: {(t7-t6)*1000:.2f} ms")
    #     print(f"  Collate additional fields: {(t8-t7)*1000:.2f} ms")
    #     print(f"  Build candidate list: {(t9-t8)*1000:.2f} ms")
    #     print(f"  Stack candidates: {(t10-t9)*1000:.2f} ms")
    #     print(f"  Set candidates: {(t11-t10)*1000:.2f} ms")
    #     print(f"  Build labels: {(t12-t11)*1000:.2f} ms")
    #     print(f"  Set labels: {(t13-t12)*1000:.2f} ms")
    #     print(f"  Build batch_ptr: {(t14-t13)*1000:.2f} ms")
    #     print(f"  Set batch_ptr: {(t15-t14)*1000:.2f} ms")
    #     print(f"  Build candidates_smiles: {(t16-t15)*1000:.2f} ms")
    #     print(f"  Set candidates_smiles: {(t17-t16)*1000:.2f} ms")
    #     print(f"  Total collate_fn: {(t17-t0)*1000:.2f} ms")
    #     return collated_batch