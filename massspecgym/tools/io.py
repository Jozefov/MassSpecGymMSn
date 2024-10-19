import pandas as pd
import numpy as np
import os
import h5py
from typing import Dict


def load_embeddings(file_path: str, identifier_col: str = 'identifiers',
                    embedding_col: str = 'embeddings') -> Dict[str, np.ndarray]:
    """
    Load embeddings from a file and return a dictionary mapping identifiers to embeddings.

    Supported formats: 'hdf5', 'csv', 'tsv'

    Parameters:
    - file_path: Path to the embeddings file.
    - identifier_col: Column name for identifiers.
    - embedding_col: Column name for embeddings.

    Returns:
    - A dictionary mapping identifier (str) to embedding (np.ndarray).
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in ['.hdf5', '.h5']:
        return load_embeddings_hdf5(file_path, identifier_col, embedding_col)
    elif ext == '.csv':
        return load_embeddings_text(file_path, 'csv', identifier_col, embedding_col)
    elif ext == '.tsv':
        return load_embeddings_text(file_path, 'tsv', identifier_col, embedding_col)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Supported formats are .hdf5, .h5, .csv, .tsv.")


def load_embeddings_hdf5(file_path: str, identifier_key: str = 'identifiers', embedding_key: str = 'embeddings') -> \
Dict[str, np.ndarray]:
    """
    Load embeddings from an HDF5 file.

    The HDF5 file should contain two datasets:
    - 'identifiers': list/array of identifier strings
    - 'embeddings': 2D array of embeddings, aligned with 'identifiers'

    Parameters:
    - file_path: Path to the HDF5 file.
    - identifier_key: Dataset key for identifiers.
    - embedding_key: Dataset key for embeddings.

    Returns:
    - A dictionary mapping identifier to embedding.
    """
    embeddings_dict = {}
    with h5py.File(file_path, 'r') as h5f:
        identifiers = h5f[identifier_key][:]
        embeddings = h5f[embedding_key][:]

        identifiers = [id_.decode('utf-8') if isinstance(id_, bytes) else id_ for id_ in identifiers]
        for id_, emb in zip(identifiers, embeddings):
            embeddings_dict[id_] = emb
    return embeddings_dict


def load_embeddings_text(file_path: str, file_format: str = 'csv', identifier_col: str = 'identifier',
                         embedding_col: str = 'embedding') -> Dict[str, np.ndarray]:
    """
    Load embeddings from a textual file (CSV or TSV).

    The file should have columns for identifiers and embeddings.
    The embeddings can be stored as lists or in separate columns.

    Parameters:
    - file_path: Path to the CSV/TSV file.
    - file_format: 'csv' or 'tsv'.
    - identifier_col: Column name for identifiers.
    - embedding_col: Column name for embeddings (assumed to be a list-like string).

    Returns:
    - A dictionary mapping identifier to embedding.
    """
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported text file format: {file_format}")

    if identifier_col not in df.columns or embedding_col not in df.columns:
        raise KeyError(f"Missing required columns '{identifier_col}' or '{embedding_col}' in the {file_format.upper()} file.")

    embeddings_dict = {}
    for _, row in df.iterrows():
        identifier = row[identifier_col]
        embedding_str = row[embedding_col]
        try:
            embedding = np.array(eval(embedding_str))
            embeddings_dict[identifier] = embedding
        except Exception as e:
            print(f"Error parsing embedding for identifier {identifier}: {e}")
            continue
    return embeddings_dict