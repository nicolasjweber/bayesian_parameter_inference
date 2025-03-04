"""
Functions to compute the Kullback-Leibler and Jensen-Shannon statistical distances.

Author: Nicolas Joschua Weber
Date: 2025-03-04

This file is part of the thesis "Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency" at KIT.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy import integrate

def kl_div_integrand(x: float, p, q) -> float:
    """
    Defines the integrand used in the function "kl_div" for computation of the Kullback-Leibler divergence.

    Args:
        x (float): The point at which the integrand is evaluated
        p (function): The first kernel density estimate
        q (function): The second kernel density estimate
    
    Returns:
        float: The value of the integrand at the given point
    """
    p_x = p(x)

    # if p_x is very small, i.e. 0 or very close to 0, we return 0 because lim x->0+: x * log(x) = 0.
    if (p_x <= np.nextafter(0, 1, dtype=np.float64)):
        return 0
    q_x = q(x)

    return p_x * (np.log(p_x) - np.log(q_x))

def kl_div(p, q) -> float:
    """
    Compute the Kullback-Leibler divergence between the two given kernel density estimates p and q. 
    Utilizes scipy.integrate.quad for numerical integration.
    
    Args:
        p (function): The first kernel density estimate
        q (function): The second kernel density estimate
    
    Returns:
        float: The Kullback-Leibler divergence between the two distributions
    """
    return integrate.quad(kl_div_integrand, -np.inf, np.inf, args=(p, q), limit=100)[0]

def jensen_shannon_div(p, q) -> float:
    """
    Compute the Jensen-Shannon divergence between the two given kernel density estimates p and q.
    
    Args:
        p (gaussian_kde): The first kernel density estimate
        q (gaussian_kde): The second kernel density estimate
        
    Returns:
        float: The Jensen-Shannon divergence between the two kernel density estimates
    """

    def mixture_pdf(x: float) -> float:
        """
        Defines the mixture pdf of the two given kernel density estimates p and q at the given point x.
        
        Args:
            x (float): The point at which the mixture pdf is evaluated
        
        Returns:
            float: The value of the mixture pdf at the given point x
        """
        return 0.5 * p(x) + 0.5 * q(x)
    
    return 0.5 * (kl_div(p, mixture_pdf) + kl_div(q, mixture_pdf))


def kl_div_mc(p: gaussian_kde, q: gaussian_kde, num_points: int) -> float:
    """
    Estimate the Kullback-Leibler divergence between the two given kernel density estimates p and q using Monte Carlo estimation.
    This function is not used in the thesis, but is only included to provide an alternative to the integration-based approach above.
    Based on the implementation in the sbi library given at [1] and the StackOverflow answer at [2].
    
    Args:
        p (gaussian_kde): The first kernel density estimate
        q (gaussian_kde): The second kernel density estimate
        num_points (int): The number of points used for Monte Carlo estimation
        
    Returns:
        float: The Kullback-Leibler divergence between the two kernel density estimates
    
    [1] https://github.com/sbi-dev/sbi/blob/main/tests/test_utils.py
    [2] https://stackoverflow.com/a/74677029
    """
    points = np.concatenate((p.resample(num_points // 2), q.resample(num_points // 2)))[0]
    p_points = p.pdf(points)
    q_points = q.pdf(points)

    return np.log(p_points / q_points).mean()
