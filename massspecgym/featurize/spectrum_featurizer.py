import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matchms
import torch
import torch.nn as nn
import re
from typing import List, Dict, Optional, Callable
from massspecgym.featurize.constants import *

class SpectrumFeaturizer:
    def __init__(self, config: Dict, mode: str = 'numpy'):
        """
        Initialize the SpectrumFeaturizer with a configuration dictionary.

        Parameters:
        - config: Dictionary specifying which features to include and any parameters.
        - mode: 'numpy' or 'torch', specifies the output type of the featurizer methods.

        Example 1:
            config_mode1 = {
                'features': ['collision_energy', 'ionmode', 'adduct', 'spectrum_stats'],
            }
        Example 2:
            When method needs more attributes:
            config_model2 = {
                'features': ['atom_counts'],
                'feature_attributes': {
                    'atom_counts': {
                        'top_n_atoms': 12,
                        'include_other': True,
                    },
                },
            }
        Example 3:
            config = {
                'features': ['collision_energy', 'ionmode', 'adduct', 'spectrum_stats', 'atom_counts'],
                'feature_attributes': {
                    'collision_energy': {
                        'encoding': 'continuous',  # Options: 'binning', 'continuous'
                    },
                    'atom_counts': {
                        'selected_atoms': ['C', 'H', 'O', 'N', 'S'],
                        'include_other': True,
                    },
                },
            }
        """
        self.config = config
        self.mode = mode.lower()
        assert self.mode in ['numpy', 'torch'], "Mode must be 'numpy' or 'torch'"

        # Map feature names to methods
        self.feature_methods = FEATURE_METHODS

    def featurize(self, node):
        """
        Featurize a TreeNode into a feature vector based on the configuration.
        Returns either a NumPy array or a PyTorch tensor based on the mode.
        """
        feature_tensors = []

        for feature_name in self.config.get('features', []):
            method_name = self.feature_methods.get(feature_name)
            if method_name is not None:
                method = getattr(self, method_name)
                feature_tensor = method(node)
                feature_tensors.append(feature_tensor)
            else:
                raise ValueError(f"Feature '{feature_name}' is not supported.")

        # Concatenate feature tensors
        if self.mode == 'numpy':
            feature_vector = np.concatenate(feature_tensors)
            return feature_vector  # Returns a NumPy array
        else:
            feature_tensor = torch.cat(feature_tensors)
            return feature_tensor  # Returns a PyTorch tensor

    def _featurize_value(self, node):
        """
        Featurize the 'value' attribute of the node.

        Supports 'continuous' or 'binning' encoding, with optional normalization.
        """
        value = float(node.value) if node.value is not None else 0.0
        encoding_method = self.config.get('feature_attributes', {}).get('value', {}).get('encoding', 'continuous')
        normalize = self.config.get('feature_attributes', {}).get('value', {}).get('normalize', False)

        if encoding_method == 'binning':
            bins = self.config.get('feature_attributes', {}).get('value', {}).get('bins', VALUE_BINS)
            bin_indices = np.digitize([value], bins) - 1
            bin_index = bin_indices[0]
            bin_index = min(bin_index, len(bins) - 1)
            if self.mode == 'numpy':
                one_hot = np.zeros(len(bins), dtype=np.float32)
                one_hot[bin_index] = 1.0
                return one_hot
            else:
                one_hot = torch.zeros(len(bins), dtype=torch.float32)
                one_hot[bin_index] = 1.0
                return one_hot

        elif encoding_method == 'continuous':
            # TODO normalization?
            if normalize:
                max_value = self.config.get('feature_attributes', {}).get('value', {}).get('max_value', 2000)
                min_value = self.config.get('feature_attributes', {}).get('value', {}).get('min_value', 0)
                value = (value - min_value) / (max_value - min_value)
            if self.mode == 'numpy':
                return np.array([value], dtype=np.float32)
            else:
                return torch.tensor([value], dtype=torch.float32)
        else:
            raise ValueError("Invalid encoding method for value.")

    def _featurize_collision_energy(self, node):
        spectrum = node.spectrum
        collision_energy = float(spectrum.get('collision_energy', 0.0)) if spectrum else 0.0
        encoding_method = self.config.get('feature_attributes', {}).get('collision_energy', {}).get('encoding', 'continuous')

        if encoding_method == 'binning':
            bins = self.config.get('feature_attributes', {}).get('collision_energy', {}).get('bins', COLLISION_ENERGY_BINS)
            bin_indices = np.digitize([collision_energy], bins) - 1
            bin_index = bin_indices[0]
            bin_index = min(bin_index, len(bins) - 1)
            if self.mode == 'numpy':
                one_hot = np.zeros(len(bins), dtype=np.float32)
                one_hot[bin_index] = 1.0
                return one_hot
            else:
                one_hot = torch.zeros(len(bins), dtype=torch.float32)
                one_hot[bin_index] = 1.0
                return one_hot
        elif encoding_method == 'continuous':
            if self.mode == 'numpy':
                return np.array([collision_energy], dtype=np.float32)
            else:
                return torch.tensor([collision_energy], dtype=torch.float32)
        else:
            raise ValueError("Invalid encoding method for collision_energy.")

    def _featurize_retention_time(self, node):
        spectrum = node.spectrum
        retention_time = float(spectrum.get('retention_time', 0.0)) if spectrum else 0.0
        encoding_method = self.config.get('feature_attributes', {}).get('retention_time', {}).get('encoding', 'continuous')

        if encoding_method == 'binning':
            bins = self.config.get('feature_attributes', {}).get('retention_time', {}).get('bins', RETENTION_TIME_BINS)
            bin_indices = np.digitize([retention_time], bins) - 1
            bin_index = bin_indices[0]
            bin_index = min(bin_index, len(bins) - 1)

            if self.mode == 'numpy':
                one_hot = np.zeros(len(bins), dtype=np.float32)
                one_hot[bin_index] = 1.0
                return one_hot
            else:
                one_hot = torch.zeros(len(bins), dtype=torch.float32)
                one_hot[bin_index] = 1.0
                return one_hot
        elif encoding_method == 'continuous':
            if self.mode == 'numpy':
                return np.array([retention_time], dtype=np.float32)
            else:
                return torch.tensor([retention_time], dtype=torch.float32)
        else:
            raise ValueError("Invalid encoding method for retention_time.")

    def _featurize_spectrum_stats(self, node):
        spectrum = node.spectrum
        peaks = spectrum.peaks if spectrum else None
        if peaks is None or len(peaks) == 0:
            mean_mz = mean_intensity = max_mz = max_intensity = num_peaks = 0.0
        else:
            mz_values = peaks.mz
            intensity_values = peaks.intensities
            mean_mz = np.mean(mz_values)
            mean_intensity = np.mean(intensity_values)
            max_mz = np.max(mz_values)
            max_intensity = np.max(intensity_values)
            num_peaks = float(len(mz_values))

        if self.mode == 'numpy':
            return np.array([mean_mz, mean_intensity, max_mz, max_intensity, num_peaks], dtype=np.float32)
        else:
            return torch.tensor([mean_mz, mean_intensity, max_mz, max_intensity, num_peaks], dtype=torch.float32)

    # Categorical Feature Methods
    def _featurize_ionmode(self, node):
        spectrum = node.spectrum
        ionmode = spectrum.get('ionmode', 'unknown') if spectrum else 'unknown'
        if self.mode == 'numpy':
            ionmode_onehot = np.array([int(ionmode == mode) for mode in IONMODE_VALUES], dtype=np.float32)
        else:
            ionmode_onehot = torch.tensor([int(ionmode == mode) for mode in IONMODE_VALUES], dtype=torch.float32)
        return ionmode_onehot

    def _featurize_adduct(self, node):
        spectrum = node.spectrum
        adduct = spectrum.get('adduct', 'unknown') if spectrum else 'unknown'
        if self.mode == 'numpy':
            adduct_onehot = np.array([int(adduct == a) for a in ADDUCT_VALUES], dtype=np.float32)
        else:
            adduct_onehot = torch.tensor([int(adduct == a) for a in ADDUCT_VALUES], dtype=torch.float32)
        return adduct_onehot

    def _featurize_ion_source(self, node):
        spectrum = node.spectrum
        ion_source = spectrum.get('ion_source', 'unknown') if spectrum else 'unknown'
        if self.mode == 'numpy':
            ion_source_onehot = np.array([int(ion_source == source) for source in ION_SOURCE_VALUES], dtype=np.float32)
        else:
            ion_source_onehot = torch.tensor([int(ion_source == source) for source in ION_SOURCE_VALUES], dtype=torch.float32)
        return ion_source_onehot

    # Other Feature Methods
    def _featurize_atom_counts(self, node):
        spectrum = node.spectrum
        formula = spectrum.get('formula') if spectrum else None
        atom_counts_config = self.config.get('feature_attributes', {}).get('atom_counts', {})
        selected_atoms = atom_counts_config.get('selected_atoms', COMMON_ATOMS)
        include_other = atom_counts_config.get('include_other', True)

        # Ensure selected_atoms is a list
        if not isinstance(selected_atoms, list):
            selected_atoms = [selected_atoms]

        if formula is None:
            vector_length = len(selected_atoms) + int(include_other)
            if self.mode == 'numpy':
                return np.zeros(vector_length, dtype=np.float32)
            else:
                return torch.zeros(vector_length, dtype=torch.float32)
        else:
            return self._calculate_atom_counts(formula, selected_atoms, include_other)

    def _calculate_atom_counts(self, formula, selected_atoms, include_other):
        # Initialize counts
        counts = {atom: 0 for atom in selected_atoms}
        other_atoms_count = 0

        # Parse the formula
        matches = re.findall('([A-Z][a-z]?)(\d*)', formula)
        for (atom, count) in matches:
            count = int(count) if count else 1
            if atom in counts:
                counts[atom] += count
            else:
                other_atoms_count += count

        # Build the feature vector
        feature_vector = [counts[atom] for atom in selected_atoms]
        if include_other:
            feature_vector.append(other_atoms_count)

        if self.mode == 'numpy':
            return np.array(feature_vector, dtype=np.float32)
        else:
            return torch.tensor(feature_vector, dtype=torch.float32)


    def _featurize_binned_peaks(self, node):
        spectrum = node.spectrum
        peaks = spectrum.peaks if spectrum else None

        # Get parameters for binning
        binned_peaks_config = self.config.get('feature_attributes', {}).get('binned_peaks', {})
        max_mz = binned_peaks_config.get('max_mz', 1005)
        bin_width = binned_peaks_config.get('bin_width', 1)
        to_rel_intensities = binned_peaks_config.get('to_rel_intensities', True)

        num_bins = int(np.ceil(max_mz / bin_width))

        if peaks is None or len(peaks) == 0:
            if self.mode == 'numpy':
                binned_intensities = np.zeros(num_bins, dtype=np.float32)
            else:
                binned_intensities = torch.zeros(num_bins, dtype=torch.float32)
        else:
            mzs = peaks.mz
            intensities = peaks.intensities

            # Calculate the bin indices for each mass
            bin_indices = np.floor(mzs / bin_width).astype(int)

            # Filter out mzs that exceed max_mz
            valid_mask = mzs <= max_mz
            valid_indices = bin_indices[valid_mask]
            valid_intensities = intensities[valid_mask]

            # Clip bin indices to ensure they are within the valid range
            valid_indices = np.clip(valid_indices, 0, num_bins - 1)

            if self.mode == 'numpy':
                # Initialize an array to store the binned intensities
                binned_intensities = np.zeros(num_bins, dtype=np.float32)
                # Use np.add.at to sum intensities in the appropriate bins
                np.add.at(binned_intensities, valid_indices, valid_intensities)
                # Normalize the intensities to relative intensities
                if to_rel_intensities and np.max(binned_intensities) > 0:
                    binned_intensities /= np.max(binned_intensities)
            else:
                # Initialize a tensor to store the binned intensities
                binned_intensities = torch.zeros(num_bins, dtype=torch.float32)
                # Convert valid_indices and valid_intensities to tensors

                # Ensure valid_indices_tensor is of type torch.long
                valid_indices_tensor = torch.from_numpy(valid_indices).long()
                # Convert valid_intensities_tensor to float32 to match binned_intensities
                valid_intensities_tensor = torch.from_numpy(valid_intensities).float()

                # Ensure both tensors are on the same device as binned_intensities
                if binned_intensities.is_cuda:
                    valid_indices_tensor = valid_indices_tensor.to(binned_intensities.device)
                    valid_intensities_tensor = valid_intensities_tensor.to(binned_intensities.device)

                # Use scatter_add to sum intensities in the appropriate bins
                binned_intensities = binned_intensities.scatter_add(0, valid_indices_tensor, valid_intensities_tensor)

                # Normalize the intensities to relative intensities
                if to_rel_intensities and torch.max(binned_intensities) > 0:
                    binned_intensities /= torch.max(binned_intensities)

        return binned_intensities

    def _featurize_spectrum_embedding(self, node):
        # TODO
        spectrum = node.spectrum
        if spectrum is None:
            # Return zero vector if spectrum is missing
            embedding_dim = self.config.get('feature_attributes', {}).get('spectrum_embedding', {}).get('embedding_dim', 128)
            return [0.0] * embedding_dim
        else:
            # Use the embedding model to get the spectrum representation
            embedding = self._get_spectrum_embedding(spectrum)
            return embedding.tolist()

    def _get_spectrum_embedding(self, spectrum):
        # TODO
        embedding_model = self.config.get('embedding_model')
        if embedding_model is None:
            raise ValueError("Embedding model is not specified in the configuration.")

        # Get the embedding from the model
        embedding = embedding_model.get_embedding(spectrum)
        return embedding