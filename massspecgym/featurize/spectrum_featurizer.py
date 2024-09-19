import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matchms
import torch
import torch.nn as nn
import re
from typing import List, Dict, Optional, Callable
from constants import *

class SpectrumFeaturizer:
    def __init__(self, config: Dict):
        """
        Initialize the SpectrumFeaturizer with a configuration dictionary.

        Parameters:
        - config: Dictionary specifying which features to include and any parameters.

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
        """
        self.config = config

        # Map feature names to methods
        self.feature_methods = FEATURE_METHODS

    def featurize(self, node) -> np.ndarray:
        """
        Featurize a TreeNode into a feature vector based on the configuration.
        """
        features = []

        for feature_name in self.config.get('features', []):
            method_name = self.feature_methods.get(feature_name)
            if method_name is not None:
                method = getattr(self, method_name)
                feature_values = method(node)
                features.extend(feature_values)
            else:
                raise ValueError(f"Feature '{feature_name}' is not supported.")

        feature_vector = np.array(features, dtype=np.float32)
        return feature_vector

    # Continuous Feature Methods
    def _featurize_collision_energy(self, node):
        spectrum = node.spectrum
        collision_energy = float(spectrum.get('collision_energy', 0.0)) if spectrum else 0.0
        return [collision_energy]

    def _featurize_retention_time(self, node):
        spectrum = node.spectrum
        retention_time = float(spectrum.get('retention_time', 0.0)) if spectrum else 0.0
        return [retention_time]

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
        return [mean_mz, mean_intensity, max_mz, max_intensity, num_peaks]

    # Categorical Feature Methods
    def _featurize_ionmode(self, node):
        spectrum = node.spectrum
        ionmode = spectrum.get('ionmode', 'unknown') if spectrum else 'unknown'
        ionmode_onehot = [int(ionmode == mode) for mode in IONMODE_VALUES]
        return ionmode_onehot

    def _featurize_adduct(self, node):
        spectrum = node.spectrum
        adduct = spectrum.get('adduct', 'unknown') if spectrum else 'unknown'
        adduct_onehot = [int(adduct == a) for a in ADDUCT_VALUES]
        return adduct_onehot

    def _featurize_ion_source(self, node):
        spectrum = node.spectrum
        ion_source = spectrum.get('ion_source', 'unknown') if spectrum else 'unknown'
        ion_source_onehot = [int(ion_source == source) for source in ION_SOURCE_VALUES]
        return ion_source_onehot

    # Other Feature Methods
    def _featurize_atom_counts(self, node):
        spectrum = node.spectrum
        formula = spectrum.get('formula') if spectrum else None
        atom_counts_config = self.config.get('feature_attributes', {}).get('atom_counts', {})
        top_n_atoms = atom_counts_config.get('top_n_atoms', len(COMMON_ATOMS))
        include_other = atom_counts_config.get('include_other', True)
        selected_atoms = COMMON_ATOMS[:top_n_atoms]

        if formula is None:
            vector_length = len(selected_atoms) + int(include_other)
            return [0] * vector_length
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
        return feature_vector

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
            binned_intensities = np.zeros(num_bins)
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

            # Initialize an array to store the binned intensities
            binned_intensities = np.zeros(num_bins)

            # Use np.add.at to sum intensities in the appropriate bins
            np.add.at(binned_intensities, valid_indices, valid_intensities)

            # Normalize the intensities to relative intensities
            if to_rel_intensities and np.max(binned_intensities) > 0:
                binned_intensities /= np.max(binned_intensities)

        return binned_intensities.tolist()

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