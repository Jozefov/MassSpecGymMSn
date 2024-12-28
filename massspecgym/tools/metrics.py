from matchms.similarity import CosineGreedy, CosineHungarian
from matchms import Spectrum
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch


def compute_cosine_greedy_score(specA: Spectrum, specB: Spectrum,
                                tolerance: float=0.1, mz_power: float=0.0,
                                intensity_power: float=1.0) -> Tuple[float,int]:
    """
    matchms CosineGreedy. Returns (score, matched_peaks).
    """
    if specA is None or specB is None:
        return (float('nan'), 0)
    cos = CosineGreedy(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    result = cos.pair(specA, specB)
    return (result["score"], result["matches"])

def compute_cosine_hungarian_score(specA: Spectrum, specB: Spectrum,
                                   tolerance: float=0.1, mz_power: float=0.0,
                                   intensity_power: float=1.0) -> Tuple[float,int]:
    """
    matchms CosineHungarian. Returns (score, matched_peaks).
    """
    if specA is None or specB is None:
        return (float('nan'), 0)
    cos = CosineHungarian(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    result = cos.pair(specA, specB)
    return (result["score"], result["matches"])

def dreams_embedding_similarity(embA: np.ndarray, embB: np.ndarray) -> float:
    """
    Simple cosine similarity for DreaMS or other embeddings
    stored as np.ndarray or torch.Tensor.
    """
    if embA is None or embB is None:
        return float('nan')
    if isinstance(embA, torch.Tensor):
        embA = embA.cpu().numpy()
    if isinstance(embB, torch.Tensor):
        embB = embB.cpu().numpy()

    dot = np.dot(embA, embB)
    normA = np.linalg.norm(embA)
    normB = np.linalg.norm(embB)
    if normA == 0.0 or normB == 0.0:
        return 0.0
    return dot / (normA * normB)