from massspecgym.data.datasets import MSnDataset, MassSpecDataset
from massspecgym.data.transforms import MolFingerprinter, SpecTokenizer
from massspecgym.data import MassSpecDataModule
from massspecgym.featurize import SpectrumFeaturizer

from scipy.stats import kruskal, spearmanr, linregress
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
from collections import defaultdict
from itertools import combinations
from typing import List
from scipy import stats
from math import comb
import numpy as np
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


from matplotlib.patches import Patch, Polygon
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess

import warnings


def summarize_similarity_distribution(scores: List[float], normality_alpha: float = 0.05) -> Dict[str, Any]:
    """
    Summarize similarity scores with extended statistics and normality tests.

    Parameters:
    - scores: List of similarity scores (floats)
    - normality_alpha: Significance level for normality tests

    Returns:
    - Dictionary with comprehensive statistics
    """
    arr = np.array(scores)
    arr = arr[~np.isnan(arr)]
    summary = {}

    if len(arr) == 0:
        summary = {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "count": 0,
            # "mode": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "IQR": np.nan,
            "range": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "ks_p": np.nan,
            "is_normal": False
        }
        return summary

    summary["mean"] = float(np.mean(arr))
    summary["std"] = float(np.std(arr))
    summary["median"] = float(np.median(arr))
    summary["count"] = len(arr)
    # summary["mode"] = float(stats.mode(arr).mode[0])
    summary["q1"] = float(np.percentile(arr, 25))
    summary["q3"] = float(np.percentile(arr, 75))
    summary["IQR"] = summary["q3"] - summary["q1"]
    summary["range"] = float(np.ptp(arr))
    summary["skewness"] = float(stats.skew(arr))
    summary["kurtosis"] = float(stats.kurtosis(arr))

    # Kolmogorov-Smirnov Test against normal distribution
    ks_stat, ks_p = stats.kstest(arr, 'norm', args=(summary["mean"], summary["std"]))

    summary["ks_p"] = ks_p

    # Determine normality based on KS test
    if ks_p >= normality_alpha:
        is_normal = True
    else:
        is_normal = False

    summary["is_normal"] = is_normal

    return summary


def perform_statistical_tests_with_effect_sizes(all_level_sims: Dict[Tuple[int, int], List[float]],
                                                alpha: float = 0.05) -> List[Dict[str, Any]]:
    """
    Perform statistical tests between all level pairs and calculate effect sizes.

    Parameters:
    - all_level_sims: Dict mapping level pairs to list of similarity scores
    - alpha: Significance level

    Returns:
    - List of dictionaries containing comparison results including effect sizes
    """
    # Summarize distributions and assess normality
    summaries = {level_pair: summarize_similarity_distribution(vals)
                 for level_pair, vals in all_level_sims.items()}

    # Prepare for pairwise comparisons
    level_pairs = list(all_level_sims.keys())
    comparison_results = []

    # Total number of comparisons for Bonferroni
    m = comb(len(level_pairs), 2) if len(level_pairs) > 1 else 0
    if m > 0:
        adjusted_alpha = alpha / m
    else:
        adjusted_alpha = alpha

    # Perform pairwise comparisons
    for (lvlA1, lvlB1), (lvlA2, lvlB2) in combinations(level_pairs, 2):
        # Define comparison labels
        group1_label = f"{lvlA1}-{lvlB1}"
        group2_label = f"{lvlA2}-{lvlB2}"

        scores1 = all_level_sims[(lvlA1, lvlB1)]
        scores2 = all_level_sims[(lvlA2, lvlB2)]

        summary1 = summaries[(lvlA1, lvlB1)]
        summary2 = summaries[(lvlA2, lvlB2)]

        # Decide which test to use based on normality
        if summary1["is_normal"] and summary2["is_normal"]:
            # Perform Welch's t-test
            t_stat, p_val = stats.ttest_ind(scores1, scores2, equal_var=False, nan_policy='omit')
            test_used = "Welch's t-test"
            # Calculate Cohen's d
            mean_diff = summary1["mean"] - summary2["mean"]
            pooled_var = ((summary1["std"] ** 2) / summary1["count"] + (summary2["std"] ** 2) / summary2["count"])
            pooled_std = np.sqrt(pooled_var)
            cohen_d = mean_diff / pooled_std
            effect_size = cohen_d
            effect_size_label = "Cohen's d"
        else:
            # Perform Mann-Whitney U test
            u_stat, p_val = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            test_used = "Mann-Whitney U test"
            # Calculate Rank-Biserial Correlation as effect size
            n1 = len(scores1)
            n2 = len(scores2)
            rbc = 1 - (2 * u_stat) / (n1 * n2)
            effect_size = rbc
            effect_size_label = "Rank-Biserial Correlation"

        # Determine significance with Bonferroni correction
        significant = p_val < adjusted_alpha

        comparison_results.append({
            "Group 1": group1_label,
            "Group 2": group2_label,
            "Test Used": test_used,
            "Statistic": t_stat if test_used == "Welch's t-test" else u_stat,
            "p-value": p_val,
            "Adjusted Alpha": adjusted_alpha,
            "Significant": significant,
            "Effect Size": effect_size,
            "Effect Size Type": effect_size_label
        })

    return comparison_results


def fit_distributions(data: List[float], distributions: List[str] = None) -> pd.DataFrame:
    """
    Fit multiple distributions to the data and evaluate goodness-of-fit using KS statistic.

    Parameters:
    - data: List of similarity scores (floats)
    - distributions: List of distribution names to fit. If None, a default list is used.

    Returns:
    - DataFrame with distribution parameters and KS statistics
    """
    if distributions is None:
        distributions = [
            'norm', 'expon', 'beta', 'gamma', 'lognorm', 'uniform', 'weibull_min',
            'weibull_max', 'pareto', 't', 'cauchy'
        ]

    results = []
    data = np.array(data)

    # Remove NaN values
    data = data[~np.isnan(data)]

    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            # Fit distribution to data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(data)

            # Perform KS test
            ks_stat, ks_p = stats.kstest(data, dist_name, args=params)

            # Append results
            results.append({
                'Distribution': dist_name,
                'Parameters': params,
                'KS Statistic': ks_stat,
                'KS p-value': ks_p
            })
        except Exception as e:
            print(f"Could not fit distribution {dist_name}: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='KS Statistic')
    return results_df


def fit_distributions_with_aic(data: List[float], distributions: List[str] = None) -> pd.DataFrame:
    """
    Fit multiple distributions to the data and evaluate goodness-of-fit using AIC.

    Parameters:
    - data: List of similarity scores (floats)
    - distributions: List of distribution names to fit. If None, a default list is used.

    Returns:
    - DataFrame with distribution parameters and AIC scores
    """
    if distributions is None:
        distributions = [
            'norm', 'expon', 'beta', 'gamma', 'lognorm', 'uniform', 'weibull_min',
            'weibull_max', 'pareto', 't', 'cauchy'
        ]

    results = []
    data = np.array(data)

    # Remove NaN values
    data = data[~np.isnan(data)]

    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            # Fit distribution to data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(data)

            # Calculate log-likelihood
            log_likelihood = np.sum(dist.logpdf(data, *params))

            # Number of parameters
            k = len(params)

            # Calculate AIC
            aic = 2 * k - 2 * log_likelihood

            # Append results
            results.append({
                'Distribution': dist_name,
                'Parameters': params,
                'AIC': aic
            })
        except Exception as e:
            print(f"Could not fit distribution {dist_name}: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='AIC')
    return results_df


def identify_top_fits(df: pd.DataFrame, criterion: str = 'KS Statistic', top_n: int = 5) -> pd.DataFrame:
    """
    Identify the top N best-fitting distributions based on a specified criterion.

    Parameters:
    - df: DataFrame containing distribution fit results
    - criterion: Column name to sort by ('KS Statistic' or 'AIC')
    - top_n: Number of top distributions to return

    Returns:
    - DataFrame of top N distributions
    """
    if criterion not in df.columns:
        raise ValueError(f"Criterion '{criterion}' not found in DataFrame columns.")

    top_fits = df.nsmallest(top_n, criterion)
    return top_fits

def summarize_all_level_pairs(
        all_level_sims: Dict[Tuple[int, int], List[float]],
        top_n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit distributions to all level pairs and summarize the top N best-fitting distributions based on KS and AIC.

    Parameters:
    - all_level_sims: Dict mapping level pairs to list of similarity scores (floats)
    - top_n: Number of top distributions to include in the summary

    Returns:
    - Tuple containing two DataFrames:
        1. Summary based on KS Statistic
        2. Summary based on AIC
    """
    ks_summary = []
    aic_summary = []

    for level_pair, sims in all_level_sims.items():
        if len(sims) == 0:
            print(f"Level pair {level_pair} has no similarity scores. Skipping.")
            continue
        print(f"\nFitting distributions for Level Pair {level_pair} with {len(sims)} scores...")

        # Fit distributions and compute KS statistics
        ks_df = fit_distributions(sims)
        if ks_df.empty:
            print(f"No distributions were successfully fitted for Level Pair {level_pair}.")
            continue
        top_ks = identify_top_fits(ks_df, criterion='KS Statistic', top_n=top_n)
        top_ks = top_ks.copy()
        top_ks['Level Pair'] = f"{level_pair[0]},{level_pair[1]}"  # Convert tuple to string
        ks_summary.append(top_ks)

        # Fit distributions and compute AIC
        aic_df = fit_distributions_with_aic(sims)
        if aic_df.empty:
            print(f"No distributions were successfully fitted for Level Pair {level_pair} based on AIC.")
            continue
        top_aic = identify_top_fits(aic_df, criterion='AIC', top_n=top_n)
        top_aic = top_aic.copy()
        top_aic['Level Pair'] = f"{level_pair[0]},{level_pair[1]}"  # Convert tuple to string
        aic_summary.append(top_aic)

    # Concatenate all summaries
    if ks_summary:
        ks_summary_df = pd.concat(ks_summary, ignore_index=True)
    else:
        ks_summary_df = pd.DataFrame()
    if aic_summary:
        aic_summary_df = pd.concat(aic_summary, ignore_index=True)
    else:
        aic_summary_df = pd.DataFrame()

    return ks_summary_df, aic_summary_df