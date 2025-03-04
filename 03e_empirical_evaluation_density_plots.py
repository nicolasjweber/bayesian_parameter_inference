"""
Generates plots to compare the probability densities of empirical observations and their corresponding simulations.

Usage Example: python 03e_empirical_evaluation_density_plots.py --continental_europe --mallorca --days 90

Author: Nicolas Joschua Weber
Date: 2025-03-04

This file is part of the thesis "Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency" at KIT.
"""

import argparse
import sys
import datetime
import pickle
from typing import List, Tuple
from os import cpu_count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from helper_functions import sample_from_posterior, build_theta_for_batch_simulator_from_samples_tensor
from src.utils.simulator import batch_simulator
import divergences


def parse_arguments() -> None:
    """ 
    Parses the command line arguments.

    Args:
        None

    Returns:
        args (Namespace): The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generates plots to compare the probability densities of empirical observations and their corresponding simulations.")
    parser.add_argument("--continental_europe", action="store_true", help="Compute plot for the Continental Europe dataset")
    parser.add_argument("--mallorca", action="store_true", help="Compute plot for the Mallorca dataset")
    parser.add_argument("--only_plot", action="store_true", help="Only generates a new plot using the precomputed data")
    parser.add_argument("--days", type=int, default=30, help="Number of days to use for the plot")

    if len(sys.argv) < 2:
        # no command line arguments have been provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def compute_obs_and_sims(df: pd.DataFrame, posterior_filename: str, start: datetime.datetime, day: int) -> None:
    """
    Computes the observations and their simulations in the given dataframe for the specific day.
    
    Args:
        df (dataframe): The dataset
        posterior_filename (str): The filename of the posterior
        start (datetime): The start date
        day (int): the number of days after the start date for which the observations and simulations should be computed

    Returns:
        observations (np.array): The observations of this day in the dataset
        simulations_flat (np.array): The flattened simulations of this day in the dataset
    """
    # simulation parameters
    time_span = 3600
    dt = 1.0
    n_workers = cpu_count()
    number_of_samples = 100

    # read in the posterior
    in_s = open("out/" + posterior_filename, "rb")
    try:
        posterior = pickle.load(in_s)
    finally:
        in_s.close()

    start = start + datetime.timedelta(days=day)

    observations = np.zeros((24, time_span))
    simulations_flat = np.zeros((24, time_span * number_of_samples))
    
    for i in range(0, 24):
        end = start + datetime.timedelta(minutes=59, seconds=59)

        # we take a 60-minute long observation
        x_obs = df[str(start):str(end)]["angular_velocity"].to_numpy()

        # print current day and hour for debugging
        print("Day: " + str(day) + ", Hour: " + str(start.hour))

        # check if the observation is empty or contains NaN values or is outside the prior range for omega_0 of [-0.6, 0.6]
        # if yes, we skip this observation
        if (x_obs.size == 0 or np.any(np.isnan(x_obs)) or x_obs[0] < -0.6 or x_obs[0] > 0.6):
            observations[i] = np.nan
            simulations_flat[i] = np.nan

            start = start + datetime.timedelta(minutes=60)
            continue
        
        observations[i] = x_obs
        samples = sample_from_posterior(posterior, number_of_samples, x_obs)
        theta = build_theta_for_batch_simulator_from_samples_tensor(samples)
        simulation_results_posterior = batch_simulator(theta, time_span, dt, n_workers)
        simulations_flat[i] = simulation_results_posterior[1].flatten()

        start = start + datetime.timedelta(minutes=60)

    return observations, simulations_flat



def compute_plot_data_CE(days: int) -> None:
    """
    Computes the data needed for the distribution plot for Continental Europe.
    Reads in the dataset, computes the plot data and saves the results to a file.

    Args:
        days (int): The number of days used to generate the plot

    Returns:
        None
    """
    start = datetime.datetime(2021, 1, 1, 0, 0, 0)
    f_ref = 50

    # read in empirical CE-dataset (2021)
    df = pd.read_csv("2021.csv", header=None)
    df.columns = ["date", "freq"]
    df["angular_velocity"] = 2 * np.pi * (df["freq"] - f_ref)
    df.index = pd.to_datetime(df["date"])

    # compute plot data for the posterior
    posterior_filename = "posterior_NPE_3600s_dt1.0.pickle"
    obs_and_sims = [compute_obs_and_sims(df, posterior_filename, start, day) for day in range(0, days)]
    with open("obs_and_sims_CE.pkl", "wb") as file:
        pickle.dump(obs_and_sims, file)


def compute_plot_data_Mallorca(days: int) -> None:
    """
    Computes the data needed for the distribution plot for the Balearic grid.
    Reads in the dataset, computes the plot data and saves the results to a file.

    Args:
        days (int): The number of days used to generate the plot

    Returns:
        None
    """
    start = datetime.datetime(2019, 10, 1, 0, 0, 0)

    # read in ES_PM01 dataset (2019)
    df = pd.read_csv("ES_PM01.csv", delimiter=";", usecols=["Time", "f50_ES_PM", "QI_ES_PM"])
    # we will look at the angular velocity. note that f50_ES_PM is given in mHz
    df["angular_velocity"] = 2 * np.pi * (df["f50_ES_PM"] / 1000)
    dateandtime = pd.to_datetime(df["Time"])
    df["date"] = dateandtime.dt.date
    df["hour"] = dateandtime.dt.hour

    # handling low quality data by masking all rows where there has been a time point with QI_ES_PM = 2 in the same hour
    # additionally, use the datetime as index (allows us to access the data by time more easily)
    df = df.merge(df[df["QI_ES_PM"] == 2][["date", "hour"]].drop_duplicates(), how="left", on=["date", "hour"], indicator=True)
    df["DatetimeIndex"] = pd.DatetimeIndex(df["Time"])
    df.set_index("DatetimeIndex", inplace=True)
    df.mask((df["_merge"] == "both"), inplace=True)

    # drop unnecessary and redundant columns
    df.drop(columns=["Time", "_merge"], inplace=True)

    # compute plot data for the posterior
    posterior_filename = "posterior_NPE_3600s_dt1.0.pickle"
    obs_and_sims = [compute_obs_and_sims(df, posterior_filename, start, day) for day in range(0, days)]
    with open("obs_and_sims_ES_PM.pkl", "wb") as file:
        pickle.dump(obs_and_sims, file)


def generate_plot(obs_and_sims: List[Tuple[np.ndarray, np.ndarray]], filename: str, x_lim: float) -> None:
    """
    Generates the plot for comparing the distributions of the observations and their simulations.
    Expects the data produced by the compute_plot_data functions as input.
    
    Args:
        obs_and_sims (List[Tuple[np.ndarray, np.ndarray]]): The list of observations and simulations
        filename (str): The filename of the plot pdf-file
        x_lim (float): The x-axis limit for the plot
        
    Returns:
        None
    """
    observations = np.array([x[0] for x in obs_and_sims])
    simulations = np.array([x[1] for x in obs_and_sims])

    sns.kdeplot(observations.flatten(), bw_adjust=1)
    sns.kdeplot(simulations.flatten(), bw_adjust=1)
    plt.yscale("log")
    plt.xlim(-x_lim, x_lim)
    plt.ylim(1e-4, 1e1)
    plt.xlabel("Angular velocity [rad/s]")
    plt.grid()
    legend_handles = [Line2D([0], [0], color="tab:blue", lw=2, label="observations"),
                      Line2D([0], [0], color="tab:orange", lw=2, label="simulations")]
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close("all")

    
def print_divergence(obs_and_sims: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Prints the Jensen-Shannon divergence between the given observations and simulations.
    Expects the data produced by the compute_plot_data functions as input.
    
    Args:
        obs_and_sims (List[Tuple[np.ndarray, np.ndarray]]): The list of observations and simulations
        
    Returns:
        None
    """
    observations = np.array([x[0] for x in obs_and_sims])
    simulations = np.array([x[1] for x in obs_and_sims])

    print(f"Jensen-Shannon divergence between observations and simulations is {divergences.jensen_shannon_div(gaussian_kde(observations.flatten()[~np.isnan(observations.flatten())]), gaussian_kde(simulations.flatten()[~np.isnan(simulations.flatten())]))}")


def generate_plot_CE() -> None:
    """
    Generates the plot for the Continental Europe dataset.
    Reads in the data previously computed by the compute_plot_data_CE function and generates the plot.
    The resulting plot is saved to a pdf-file.

    Args:
        None
    
    Returns:
        None
    """
    # read in the plot data, which has been generated previously, and generate the plot for CE
    obs_and_sims = pickle.load(open("obs_and_sims_CE.pkl", "rb"))
    generate_plot(obs_and_sims, "figures/distribution_comparison_CE.pdf", 1.0)


def generate_plot_Mallorca() -> None:
    """
    Generates the plot for the Mallorcan dataset.
    Reads in the data previously computed by the compute_plot_data_Mallorca function and generates the plot.
    The resulting plot is saved to a pdf-file.

    Args:
        None
    
    Returns:
        None
    """
    # read in the plot data, which has been generated previously, and generate the plot for CE
    obs_and_sims = pickle.load(open("obs_and_sims_ES_PM.pkl", "rb"))
    generate_plot(obs_and_sims, "figures/distribution_comparison_ES_PM.pdf", 1.5)


if __name__ == "__main__":
    # Main function to generate the plots for the empirical evaluation of the parameter estimation with constrained frequency control.
    # Parses the command line arguments to determine for which datasets the plots should be generated.
    # If the only_plot command line argument is given, only the plot is generated using the precomputed data.
    args = parse_arguments()

    if args.continental_europe:
        if not args.only_plot:
            compute_plot_data_CE(args.days)
        generate_plot_CE()

    if args.mallorca:
        if not args.only_plot:
            compute_plot_data_Mallorca(args.days)
        generate_plot_Mallorca()
