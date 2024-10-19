import pandas as pd
import ast
import json
import typing as T
import numpy as np
import torch
from collections import deque
import matchms
import massspecgym.utils as utils
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch
from matchms.importing import load_from_mgf
from massspecgym.featurize import SpectrumFeaturizer
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey


class MassSpecDataset(Dataset):
    """
    Dataset containing mass spectra and their corresponding molecular structures. This class is
    responsible for loading the data from disk and applying transformation steps to the spectra and
    molecules.
    """

    def __init__(
        self,
        spec_transform: Optional[SpecTransform] = None,
        mol_transform: Optional[MolTransform] = None,
        pth: Optional[Path] = None,
        return_mol_freq: bool = True,
        return_identifier: bool = True,
        dtype: T.Type = torch.float32
    ):
        """
        Args:
            mgf_pth (Optional[Path], optional): Path to the .tsv or .mgf file containing the mass spectra.
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

    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra[i]
        spec = (
            self.spec_transform(spec)
            if transform_spec and self.spec_transform
            else spec
        )
        spec = torch.as_tensor(spec, dtype=self.dtype)

        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]
        mol = self.mol_transform(mol) if transform_mol and self.mol_transform else mol
        if isinstance(mol, np.ndarray):
            mol = torch.as_tensor(mol, dtype=self.dtype)

        item = {"spec": spec, "mol": mol}

        # TODO: Add other metadata to the item. Should it be just done in subclasses?
        item.update({
            k: metadata[k] for k in ["precursor_mz", "adduct"]
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

        return item

    @staticmethod
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
        super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform

        # Download candidates from HuggigFace Hub
        if self.candidates_pth is None:
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_mass.json"
            )
        elif isinstance(self.candidates_pth, str):
            self.candidates_pth = utils.hugging_face_download(candidates_pth)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            self.candidates = json.load(file)

    def __getitem__(self, i) -> dict:
        item = super().__getitem__(i, transform_mol=False)

        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = item["mol"]

        # Get candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = self.candidates[item["mol"]]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]

        # Create neg/pos label mask by matching the query molecule with the candidates
        item_label = self.mol_label_transform(item["mol"])
        item["labels"] = [
            self.mol_label_transform(c) == item_label for c in item["candidates"]
        ]

        if not any(item["labels"]):
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        # Transform the query and candidate molecules
        item["mol"] = self.mol_transform(item["mol"])
        item["candidates"] = [self.mol_transform(c) for c in item["candidates"]]
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)
            # item["candidates"] = [torch.as_tensor(c, dtype=self.dtype) for c in item["candidates"]]

        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        # Standard collate for everything except candidates and their labels (which may have different length per sample)
        collated_batch = {}
        for k in batch[0].keys():
            if k not in ["candidates", "labels", "candidates_smiles"]:
                collated_batch[k] = default_collate([item[k] for item in batch])

        # Collate candidates and labels by concatenating and storing sizes of each list
        collated_batch["candidates"] = torch.as_tensor(
            np.concatenate([item["candidates"] for item in batch])
        )
        collated_batch["labels"] = torch.as_tensor(
            sum([item["labels"] for item in batch], start=[])
        )
        collated_batch["batch_ptr"] = torch.as_tensor(
            [len(item["candidates"]) for item in batch]
        )
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

    def to_pyg_data(self, featurizer: Optional[SpectrumFeaturizer] = None, hierarchical: bool = False):

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

        if not hierarchical:
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
    def __init__(self, pth=None, dtype=torch.float32, mol_transform=None, featurizer=None,
                 max_allowed_deviation: float = 0.005, prune_missing_spectra=True):
        # load dataset using the parent class
        super().__init__(pth=pth)

        self.mol_transform = mol_transform
        self.max_allowed_deviation = max_allowed_deviation
        self.dtype = dtype
        self.metadata = self.metadata[self.metadata["spectype"] == "ALL_ENERGIES"]

        # TODO: add identifiers (and split?) to the mgf file

        # Map identifier to spectra
        # Create mappings from 'IDENTIFIER' to spectra and spectra indices
        self.identifier_to_spectrum = {spectrum.get('identifier'): spectrum for spectrum in self.spectra}

        # Create feaurizer that wil parse and annotate PYG data
        self.featurizer = featurizer

        # get paths from the metadata
        self.all_tree_paths = self._parse_paths_from_df(self.metadata)

        # Generate trees from paths and their corresponding SMILES
        self.trees, self.pyg_trees, self.smiles = self._generate_trees(self.all_tree_paths,
                                                                       prune_missing_spectra=prune_missing_spectra)
        # TODO PYG trees

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

        item = {"spec_tree": spec_tree, "mol": mol_feature}
        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict]) -> dict:
        """
        Custom collate function to handle batches of spec_tree and mol.
        """

        spec_trees = [item['spec_tree'] for item in batch]
        mols = [item['mol'] for item in batch]

        batch_spec_trees = Batch.from_data_list(spec_trees)
        mols = torch.stack(mols) if isinstance(mols[0], torch.Tensor) else mols

        return {'spec_tree': batch_spec_trees, 'mol': mols}


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

    def _generate_trees(self, dataset_all_tree_paths: List[Tuple[str, float, List[Tuple[List[float], matchms.Spectrum]], matchms.Spectrum]],
                        prune_missing_spectra: bool = False) \
            -> Tuple[List['Tree'], List['Data'], List[str]]:
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

            pyg_tree = tree.to_pyg_data(self.featurizer)
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