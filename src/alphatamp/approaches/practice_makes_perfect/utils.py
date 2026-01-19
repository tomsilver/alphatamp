"""General utility methods."""

from typing import List, Tuple, Any, Set
from scipy.stats import beta as BetaRV


def _beta_bernoulli_posterior_alpha_beta(
    success_history: List[bool], alpha: float = 1.0, beta: float = 1.0
) -> Tuple[float, float]:
    """See https://gregorygundersen.com/blog/2020/08/19/bernoulli-beta/"""
    n = len(success_history)
    s = sum(success_history)
    alpha_n = alpha + s
    beta_n = n - s + beta
    return (alpha_n, beta_n)


def beta_bernoulli_posterior(
    success_history: List[bool], alpha: float = 1.0, beta: float = 1.0
) -> BetaRV:
    """Returns the RV."""
    alpha_n, beta_n = _beta_bernoulli_posterior_alpha_beta(success_history, alpha, beta)
    return BetaRV(alpha_n, beta_n)


def beta_bernoulli_posterior_mean(
    success_history: List[bool], alpha: float = 1.0, beta: float = 1.0
) -> float:
    """Faster computation to avoid instantiating BetaRV when not needed."""
    alpha_n, beta_n = _beta_bernoulli_posterior_alpha_beta(success_history, alpha, beta)
    return alpha_n / (alpha_n + beta_n)


def beta_from_mean_and_variance(
    mean: float,
    variance: float,
    variance_lower_pad: float = 1e-6,
    variance_upper_pad: float = 1e-3,
) -> BetaRV:
    """Recover a beta distribution given a mean and a variance.

    See https://stats.stackexchange.com/questions/12232/ for derivation.
    """
    # Clip variance.
    variance = max(
        min(variance, mean * (1 - mean) - variance_upper_pad), variance_lower_pad
    )
    alpha = ((1 - mean) / variance - 1 / mean) * (mean**2)
    beta = alpha * (1 / mean - 1)
    assert alpha > 0
    assert beta > 0
    rv = BetaRV(alpha, beta)
    assert abs(rv.mean() - mean) < 1e-6
    return rv


def get_all_subclasses(cls: Any) -> Set[Any]:
    """Get all subclasses of the given class."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )
