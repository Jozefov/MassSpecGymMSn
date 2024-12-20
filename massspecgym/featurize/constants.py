# Global constants for categorical features
IONMODE_VALUES = ['positive', 'negative', 'unknown']
ADDUCT_VALUES = ['[M+H]+', '[M+NH4]+', '[M+H-H2O]+', '[M]+', '[M+Na]+', '[M+H-2H2O]+', '[M-H2O]+', 'unknown']
ION_SOURCE_VALUES = ['ESI', 'unknown']

# Global constants for collision energy bins
COLLISION_ENERGY_BINS = [0, 15, 20, 30, 45, 60, 75]
COLLISION_ENERGY_BIN_TOKENS = [f'collision_energy_bin_{i}' for i in range(len(COLLISION_ENERGY_BINS))]


# Global constants for retention time bins
RETENTION_TIME_BINS = [0, 20, 40, 60, 80, 100]
RETENTION_TIME_BIN_TOKENS = [f'collision_energy_bin_{i}' for i in range(len(RETENTION_TIME_BINS))]

# Global constant for common atoms
# 'Unknown' is solved in _featurize_atom_counts, as we can choose from atoms we want to include +1 unknown
COMMON_ATOMS = ['C', 'H', 'O', 'N', 'S', 'P', 'Cl', 'F', 'Br', 'I', 'Si', 'B']

VALUE_BINS = [0, 100, 200, 300, 400, 500, 750, 1000, 1500]

CONTINUOUS_FEATURES = {
    'collision_energy': '_featurize_collision_energy',
    'retention_time': '_featurize_retention_time',
    'spectrum_stats': '_featurize_spectrum_stats',
    'value': '_featurize_value',
}

CATEGORICAL_FEATURES = {
    'ionmode': '_featurize_ionmode',
    'adduct': '_featurize_adduct',
    'ion_source': '_featurize_ion_source',
}

OTHER_FEATURES = {
    'atom_counts': '_featurize_atom_counts',
    'binned_peaks': '_featurize_binned_peaks',
    'spectrum_embedding': '_featurize_spectrum_embedding',
    'spectral_data': '_featurize_spectral_data',
}

FEATURE_METHODS = {**CONTINUOUS_FEATURES, **CATEGORICAL_FEATURES, **OTHER_FEATURES}

FEATURES_REQUIRING_PREPROCESSING = {
    'spectrum_embedding': '_init_embeddings',
}