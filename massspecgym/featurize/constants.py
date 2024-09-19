# Global constants for categorical features
IONMODE_VALUES = ['positive', 'negative', 'unknown']
ADDUCT_VALUES = ['[M+H]+', '[M+NH4]+', '[M+H-H2O]+', '[M]+', '[M+Na]+', '[M+H-2H2O]+', '[M-H2O]+', 'unknown']
ION_SOURCE_VALUES = ['ESI', 'unknown']

# Global constant for common atoms
COMMON_ATOMS = ['C', 'H', 'O', 'N', 'S', 'P', 'Cl', 'F', 'Br', 'I', 'Si', 'B']

CONTINUOUS_FEATURES = {
    'collision_energy': '_featurize_collision_energy',
    'retention_time': '_featurize_retention_time',
    'spectrum_stats': '_featurize_spectrum_stats',
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
}

FEATURE_METHODS = {**CONTINUOUS_FEATURES, **CATEGORICAL_FEATURES, **OTHER_FEATURES}