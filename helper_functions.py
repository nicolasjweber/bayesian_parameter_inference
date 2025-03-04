"""
Helper functions used in the code of the thesis "Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency" at KIT.

Author: Nicolas Joschua Weber
Date: 2025-03-04
"""

from concurrent.futures import ProcessPoolExecutor
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
import pandas as pd
import divergences

def sample_from_posterior(posterior: NeuralPosterior, number_of_samples: int, x_obs: torch.Tensor, show_progress_bars: bool = False) -> torch.Tensor:
    """
    Draws the specified number of samples from the given posterior distribution.
    
    Args:
        posterior (NeuralPosterior): The posterior distribution
        number_of_samples (int): The number of samples to be drawn
        x_obs (Tensor): The observation
        show_progress_bars (bool): Specifies whether progress bars during sampling should be shown
        
    Returns:
        samples (Tensor): The posterior samples
    """
    return posterior.sample((number_of_samples,), x=x_obs, show_progress_bars=show_progress_bars)

def build_theta_for_batch_simulator_from_samples_tensor(samples: torch.Tensor, inertia: bool = False, specific_observation: bool = False, specific_P0: float = 0.0, specific_P1: float = 0.0) -> torch.Tensor:
    """ 
    Builds the theta tensor for the batch simulator given the samples tensor. If inertia estimation or parameter estimation with given empirical power mismatch for a specific observation is conducted, additional parameters need to be given.
    If both parameters inertia and specific_observation are set to True, the method build the theta tensor for inertia estimation.
    The method is necessary as the batch simulator takes in more arguments than we draw from the prior, so we need to build the theta tensor manually.
    
    Args:
        samples (Tensor): The samples tensor
        inertia (bool): Specifies whether inertia estimation is conducted or not
        specific_observation (bool): Specifies whether parameter estimation for a specific observation is conducted or not
        specific_p0 (float): The specific p0 value of the given observation
        specific_p1 (float): The specific p1 value of the given observation

    Returns:
        theta (Tensor): The theta tensor to be used for the batch simulator
    """
    if (inertia):
        # parameter estimation with constrained frequency control
        phi_0 = 0.0
        c_1 = 0.0
        c_2 = 0.0
        proportionality_constant = 1.0
        
        theta = [[sample[0], phi_0, c_1, c_2, sample[1], sample[2], sample[3], sample[4], proportionality_constant] for sample in samples]

    elif (specific_observation):
        # parameter estimation with empirical power mismatch for a specific observation
        H = 1.0

        theta = [[sample[0], sample[1], sample[2], sample[3], specific_P0, specific_P1, sample[4], H, sample[5]] for sample in samples]

    else: 
        # parameter estimation with fixed inertia
        H = 1.0
        proportionality_constant = 1.0

        theta = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], H, proportionality_constant] for sample in samples]

    return torch.tensor(theta)

def compute_divergence_for_obs(x_obs: torch.Tensor, simulations: torch.Tensor) -> float:
    """
    Computes the mean Jensen-Shannon divergence between the given observation and the given simulations. Utilizes multiprocessing using ProcessPoolExecutor.
    If the observation contains NaN values, NaN is returned instead.

    Args:
        x_obs (Tensor): The observation
        simulations (Tensor): The simulations
        
    Returns:
        js_div (float): The Jensen-Shannon divergence measure
    """
    if (np.any(np.isnan(x_obs))):
        # this observation should be ignored because it contains NaN values
        return np.nan
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(divergences.jensen_shannon_div, gaussian_kde(x_obs), gaussian_kde(sim)) for sim in simulations]
        results = [future.result() for future in futures]
        
    return np.nanmean(results)


def plot_angular_velocity(df: pd.DataFrame, start: datetime.datetime, end: datetime.datetime) -> None:
    """
    Plots the angular velocity of the given dataframe between the given start date and end date.
    Expects the dataframe to have a column named "angular_velocity" and a DateTimeIndex.
    
    Args:
        df (DataFrame): The dataframe
        start (datetime): The start timestamp
        end (datetime): The end timestamp
    
    Returns:
        None
    """
    df[str(start):str(end)]["angular_velocity"].plot()


def plot_daily_profile(df: pd.DataFrame) -> None:
    """
    Plots the daily profile of the angular velocity of the given dataframe.
    Expects the dataframe to have a column named "angular_velocity" and a DateTimeIndex.
    
    Args:
        df (DataFrame): The dataframe
        
    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    plt.plot(np.arange(0, 24, 1 / 3600), df.groupby([df.index.hour, df.index.minute, df.index.second])["angular_velocity"].mean())
    ax.set_xticks([0, 6, 12, 18, 24])
    plt.xlabel("Hour of the day")
    plt.ylabel("Angular velocity [rad/s]")
    plt.grid()
    plt.show()


def plot_autocorrelation(df: pd.DataFrame) -> None:
    """
    Plots the autocorrelation function of the angular velocity for the given dataframe.
    Expects the dataframe to have a column named "angular_velocity".
    
    Args:
        df (DataFrame): The dataframe
    
    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.set_xticks(np.arange(0, 60 * 75 + 60 * 15, 60 * 15))
    ax.set_xticklabels([0, 15, 30, 45, 60, 75])
    ax.set_xlabel("lag [min]")
    ax.set_ylabel("autocorrelation")
    ax.grid()
    y_acf = acf(df["angular_velocity"], nlags=60 * 75, missing="conservative")
    # alternatively, use x = plot_acf(df["angular_velocity"], lags=60 * 75, title=title, missing="conservative", ax=ax)
    plt.plot(y_acf)
    plt.tight_layout()
    plt.show()