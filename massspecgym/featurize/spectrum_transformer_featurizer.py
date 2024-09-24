import torch
import torch.nn as nn
from typing import Dict
from massspecgym.featurize.spectrum_featurizer import SpectrumFeaturizer
from massspecgym.featurize.constants import *
import numpy as np

class SpectrumTransformerFeaturizer(SpectrumFeaturizer):
    def __init__(self, config: Dict):
        """
               Initialize the SpectrumTransformerFeaturizer with a configuration dictionary.

               Parameters:
               - config: Dictionary specifying which features to include and embedding parameters.

               Example:
                   config = {
                                   'features': ['collision_energy', 'ionmode', 'adduct', 'spectral_data'],
                                   'feature_attributes': {
                                       'collision_energy': {
                                           'encoding': 'continuous',  # Options: 'binning', 'continuous'
                                       },
                                       'spectral_data': {
                                           'max_peaks': 100,  # Limit to top 100 peaks
                                       },
                                   },
                                   'max_sequence_length': 512,
                                   'embedding_dim': 128,
                                   'cls_token_id': 0,
                                   'sep_token_id': 1,
                                   'pad_token_id': 2,
                                   'spec_start_token_id': 3,  # Optional
                               }
               """
        super().__init__(config)
        self.max_sequence_length = config.get('max_sequence_length', 512)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.special_tokens = {
            'CLS': config.get('cls_token_id', 0),
            'SEP': config.get('sep_token_id', 1),
            'PAD': config.get('pad_token_id', 2),
            'SPEC_START': config.get('spec_start_token_id', 1),  # Optional
        }

        self.output_peak_embedding_dim = self.config.get('feature_attributes', {}).get('spectral_data', {}).get(
            'output_peak_embedding_dim', '1d')  # Options: '1d', '2d'

        # Initialize embedding layers
        self.init_embeddings()

    def init_embeddings(self):
        """
        Initialize embedding layers for tokens.
        """
        # Initialize embeddings for categorical features and special tokens
        total_tokens = self._calculate_total_tokens()
        self.token_embeddings = nn.Embedding(num_embeddings=total_tokens, embedding_dim=self.embedding_dim)

        # Embedding layers for continuous numerical features
        self.numeric_feature_embeddings = nn.ModuleDict()
        for feature_name in CONTINUOUS_FEATURES.keys():
            if feature_name in self.config.get('features', []):
                self.numeric_feature_embeddings[feature_name] = nn.Linear(1, self.embedding_dim)

        # Embedding for atom counts
        if 'atom_counts' in self.config.get('features', []):
            # Determine the number of atoms
            atom_counts_config = self.config.get('feature_attributes', {}).get('atom_counts', {})
            top_n_atoms = atom_counts_config.get('top_n_atoms', len(COMMON_ATOMS))
            include_other = atom_counts_config.get('include_other', True)
            num_atom_features = top_n_atoms + int(include_other)

            self.atom_counts_embedding = nn.Linear(num_atom_features, self.embedding_dim)

        # Embedding layers for spectral data (m/z and intensity)
        self.mz_embedding = nn.Linear(1, self.embedding_dim)
        self.intensity_embedding = nn.Linear(1, self.embedding_dim)

    def _calculate_total_tokens(self):
        """
        Calculate the total number of tokens needed for categorical features and special tokens.
        """
        # Start with special tokens
        token_count = max(self.special_tokens.values()) + 1

        # Initialize token ID mappings
        self.categorical_token_ids = {}

        # Add tokens for categorical features using global constants
        for feature_name in self.config.get('features', []):
            if feature_name in CATEGORICAL_FEATURES:
                if feature_name == 'ionmode':
                    values = IONMODE_VALUES
                elif feature_name == 'adduct':
                    values = ADDUCT_VALUES
                elif feature_name == 'ion_source':
                    values = ION_SOURCE_VALUES
                else:
                    raise ValueError(f"Feature '{feature_name}' is not supported.")

                self.categorical_token_ids[feature_name] = {}
                for value in values:
                    self.categorical_token_ids[feature_name][value] = token_count
                    token_count += 1

            elif feature_name == 'collision_energy':
                encoding_method = self.config.get('feature_attributes', {}).get('collision_energy', {}).get('encoding',
                                                                                                            'continuous')
                if encoding_method == 'binning':
                    # Use predefined bins and assign token IDs
                    self.categorical_token_ids['collision_energy'] = {}
                    for i, bin_label in enumerate(COLLISION_ENERGY_BIN_TOKENS):
                        self.categorical_token_ids['collision_energy'][i] = token_count
                        token_count += 1
                # No need to assign token IDs if using continuous encoding

            elif feature_name == 'retention_time':
                encoding_method = self.config.get('feature_attributes', {}).get('retention_time', {}).get('encoding',
                                                                                                          'continuous')
                if encoding_method == 'binning':
                    bins = self.config.get('feature_attributes', {}).get('retention_time', {}).get('bins', RETENTION_TIME_BINS)
                    self.categorical_token_ids['retention_time'] = {}
                    for i in range(len(bins)):
                        self.categorical_token_ids['retention_time'][i] = token_count
                        token_count += 1

        return token_count

    def featurize(self, node) -> torch.Tensor:
        """
        Featurize a TreeNode into a sequence of embeddings suitable for transformer input.
        """
        sequence_embeddings = []
        position_ids = []
        position = 0

        # Start with [CLS] token
        cls_embedding = self.token_embeddings(torch.tensor(self.special_tokens['CLS']))
        sequence_embeddings.append(cls_embedding)
        position_ids.append(position)
        position += 1

        # Encode features
        for feature_name in self.config.get('features', []):
            method_name = self.feature_methods.get(feature_name)
            if method_name is not None:
                method = getattr(self, method_name)
                feature_embeddings = method(node)
                if feature_embeddings is not None:
                    for emb in feature_embeddings:
                        sequence_embeddings.append(emb)
                        position_ids.append(position)
                        position += 1
            else:
                raise ValueError(f"Feature '{feature_name}' is not supported.")

        # Add [SEP] token
        sep_embedding = self.token_embeddings(torch.tensor(self.special_tokens['SEP']))
        sequence_embeddings.append(sep_embedding)
        position_ids.append(position)
        position += 1

        # Optionally add a special token for spectral data
        if 'SPEC_START' in self.special_tokens:
            spec_start_embedding = self.token_embeddings(torch.tensor(self.special_tokens['SPEC_START']))
            sequence_embeddings.append(spec_start_embedding)
            position_ids.append(position)
            position += 1

        # Encode spectral data (if included)
        spectral_embeddings = None
        if 'spectral_data' in self.config.get('features', []):
            spectral_embeddings = self._featurize_spectral_data(node)
            for emb in spectral_embeddings:
                sequence_embeddings.append(emb)
                position_ids.append(position)
                position += 1
                if position >= self.max_sequence_length:
                    break  # Truncate if max length is reached

        # Add final [SEP] token
        sep_embedding = self.token_embeddings(torch.tensor(self.special_tokens['SEP']))
        sequence_embeddings.append(sep_embedding)
        position_ids.append(position)
        position += 1

        # Pad sequence if necessary
        if len(sequence_embeddings) < self.max_sequence_length:
            pad_length = self.max_sequence_length - len(sequence_embeddings)
            pad_embedding = self.token_embeddings(torch.tensor(self.special_tokens['PAD']))
            for _ in range(pad_length):
                sequence_embeddings.append(pad_embedding)
                position_ids.append(position)
                position += 1

        # Stack embeddings
        if spectral_embeddings and isinstance(spectral_embeddings[0], torch.Tensor):
            if spectral_embeddings[0].dim() == 1:
                # 1D embeddings, stack normally
                embeddings = torch.stack(sequence_embeddings)
            elif spectral_embeddings[0].dim() == 2:
                # 2D embeddings, need to handle accordingly
                # For example, flatten or stack differently
                embeddings = torch.cat(
                    [emb.unsqueeze(0) if emb.dim() == 1 else emb.view(-1, self.embedding_dim) for emb in
                     sequence_embeddings], dim=0)
            else:
                raise ValueError("Unexpected embedding dimension.")
        else:
            embeddings = torch.stack(sequence_embeddings)

        return embeddings

    def _featurize_collision_energy(self, node):
        # Similar implementation but returns embeddings
        spectrum = node.spectrum
        collision_energy = float(spectrum.get('collision_energy', 0.0)) if spectrum else 0.0
        encoding_method = self.config.get('feature_attributes', {}).get('collision_energy', {}).get('encoding', 'continuous')

        if encoding_method == 'binning':
            # Use predefined bins and token IDs
            bins = self.config.get('feature_attributes', {}).get('collision_energy', {}).get('bins', COLLISION_ENERGY_BINS)
            bin_indices = np.digitize([collision_energy], bins) - 1
            bin_index = bin_indices[0]
            bin_index = min(bin_index, len(bins) - 1)
            token_id = self.categorical_token_ids['collision_energy'][bin_index]
            token_embedding = self.token_embeddings(torch.tensor(token_id))
            return [token_embedding]
        elif encoding_method == 'continuous':
            ce_tensor = torch.tensor([[collision_energy]], dtype=torch.float32)
            ce_embedding = self.numeric_feature_embeddings['collision_energy'](ce_tensor).squeeze(0)
            return [ce_embedding]
        else:
            raise ValueError("Invalid encoding method for collision_energy.")

    def _featurize_retention_time(self, node):
        spectrum = node.spectrum
        retention_time = float(spectrum.get('retention_time', 0.0)) if spectrum else 0.0
        encoding_method = self.config.get('feature_attributes', {}).get('retention_time', {}).get('encoding',
                                                                                                  'continuous')

        if encoding_method == 'binning':
            # Define bins and assign token IDs (similar to collision_energy)
            bins = self.config.get('feature_attributes', {}).get('retention_time', {}).get('bins',
                                                                                           COLLISION_ENERGY_BINS)
            bin_indices = np.digitize([retention_time], bins) - 1
            bin_index = bin_indices[0]
            bin_index = min(bin_index, len(bins) - 1)
            token_id = self.categorical_token_ids['retention_time'][bin_index]
            token_embedding = self.token_embeddings(torch.tensor(token_id))
            return [token_embedding]
        elif encoding_method == 'continuous':
            rt_tensor = torch.tensor([[retention_time]], dtype=torch.float32)
            rt_embedding = self.numeric_feature_embeddings['retention_time'](rt_tensor).squeeze(0)
            return [rt_embedding]
        else:
            raise ValueError("Invalid encoding method for retention_time.")

    # Categorical Feature Methods
    def _featurize_ionmode(self, node):
        spectrum = node.spectrum
        ionmode = spectrum.get('ionmode', 'unknown') if spectrum else 'unknown'
        token_id = self.categorical_token_ids['ionmode'].get(ionmode, self.categorical_token_ids['ionmode']['unknown'])
        token_embedding = self.token_embeddings(torch.tensor(token_id))
        return [token_embedding]

    def _featurize_adduct(self, node):
        spectrum = node.spectrum
        adduct = spectrum.get('adduct', 'unknown') if spectrum else 'unknown'
        token_id = self.categorical_token_ids['adduct'].get(adduct, self.categorical_token_ids['adduct']['unknown'])
        token_embedding = self.token_embeddings(torch.tensor(token_id))
        return [token_embedding]

    def _featurize_ion_source(self, node):
        spectrum = node.spectrum
        ion_source = spectrum.get('ion_source', 'unknown') if spectrum else 'unknown'
        token_id = self.categorical_token_ids['ion_source'].get(ion_source, self.categorical_token_ids['ion_source']['unknown'])
        token_embedding = self.token_embeddings(torch.tensor(token_id))
        return [token_embedding]

    # Other Feature Methods
    def _featurize_atom_counts(self, node):
        counts = super()._featurize_atom_counts(node)
        counts_tensor = torch.tensor([counts], dtype=torch.float32)  # Shape: (1, num_atoms)
        atom_counts_embedding = self.atom_counts_embedding(counts_tensor).squeeze(0)
        return [atom_counts_embedding]

    def _featurize_spectral_data(self, node):
        # Modified to represent peaks as 2D embeddings
        spectrum = node.spectrum
        peaks = spectrum.peaks if spectrum else None

        spectral_embeddings = []

        if peaks is None or len(peaks) == 0:
            return spectral_embeddings  # Return empty list

        mz_values = peaks.mz
        intensity_values = peaks.intensities

        # Optionally, limit the number of peaks (e.g., top N peaks by intensity)
        max_peaks = self.config.get('feature_attributes', {}).get('spectral_data', {}).get('max_peaks', None)
        if max_peaks is not None and len(mz_values) > max_peaks:
            indices = np.argsort(intensity_values)[-max_peaks:]
            mz_values = mz_values[indices]
            intensity_values = intensity_values[indices]

        # For each peak, create a 2D embedding
        for mz, intensity in zip(mz_values, intensity_values):
            mz_tensor = torch.tensor([[mz]], dtype=torch.float32)
            intensity_tensor = torch.tensor([[intensity]], dtype=torch.float32)

            mz_emb = self.mz_embedding(mz_tensor).squeeze(0)  # Shape: (embedding_dim)
            intensity_emb = self.intensity_embedding(intensity_tensor).squeeze(0)  # Shape: (embedding_dim)

            if self.output_peak_embedding_dim == '1d':
                # Combine embeddings into a single 1D vector
                peak_embedding = torch.cat([mz_emb, intensity_emb], dim=0)  # Shape: (2 * embedding_dim)
            elif self.output_peak_embedding_dim == '2d':
                # Combine embeddings into a 2D tensor
                peak_embedding = torch.stack([mz_emb, intensity_emb], dim=0)  # Shape: (2, embedding_dim)
            else:
                raise ValueError("Invalid output_peak_embedding_dim. Choose '1d' or '2d'.")

            spectral_embeddings.append(peak_embedding)

        return spectral_embeddings
