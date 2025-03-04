"""
Generates the empirical evaluation plots for the parameter estimation with constrained frequency control (60-second intervals).

Usage Example: python 03d_empirical_evaluation_inertia_divergence_plot.py --continental_europe --mallorca --days 90

Author: Nicolas Joschua Weber
Date: 2025-03-04

This file is part of the thesis "Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency" at KIT.
"""
import argparse
import sys
import datetime
import pickle
from typing import List
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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
    parser = argparse.ArgumentParser(description="Computes the plots for the empirical evaluation of the parameter estimation with constrained frequency control.")
    parser.add_argument("--continental_europe", action="store_true", help="Compute plot for the Continental Europe dataset")
    parser.add_argument("--mallorca", action="store_true", help="Compute plot for the Mallorca dataset")
    parser.add_argument("--only_plot", action="store_true", help="Only generates a new plot using the precomputed divergence data")
    parser.add_argument("--days", type=int, default=90, help="Number of days to use for the plot")

    if len(sys.argv) < 2:
        # no command line arguments have been provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()



def compute_divergence_for_hour(df: pd.DataFrame, posterior_filename: str, start: datetime.datetime, days: int, hour: int) -> None:
    """
    Computes the Jensen-Shannon divergence of the posterior for a specific hour over all days of the given dataset.
    
    Args:
        df (dataframe): The dataset
        posterior_filename (str): The filename of the posterior
        start (datetime): The start date
        days (int): The number of days
        hour (int): The hour of the day
    
    Returns:
        means_for_hour (np.array): The mean Jensen-Shannon divergence at the given hour for each day
    
    """
    # simulation parameters
    time_span = 60
    dt = 0.01
    n_workers = cpu_count()
    number_of_samples = 100

    # read in the posterior
    in_s = open("out/" + posterior_filename, "rb")
    try:
        posterior = pickle.load(in_s)
    finally:
        in_s.close()

    start = start + datetime.timedelta(hours=hour)
    means_for_hour = np.zeros(days)
    for i in range(0, days):
        print(f"hour {hour}, day {i} starting")

        # we select the 1-minute long observation starting at the given hour mark
        x_obs = df[str(start):str(start + datetime.timedelta(seconds=59))]["angular_velocity"].to_numpy()

        # check if the observation is empty or contains NaN values or is outside the prior range for omega_0 of [-0.6, 0.6]
        # if yes, we skip this observation
        if (x_obs.size == 0 or np.any(np.isnan(x_obs)) or x_obs[0] < -0.6 or x_obs[0] > 0.6):
            means_for_hour[i] = np.nan

            start = start + datetime.timedelta(days=1)
            continue

        # sample from the posterior and simulate the trajectories
        simulations = batch_simulator(build_theta_for_batch_simulator_from_samples_tensor(sample_from_posterior(posterior, number_of_samples, x_obs), inertia=True), time_span, dt, n_workers)[1][:, ::int(1/dt)]
        print(f"hour {hour}, day {i} simulations completed")

        # compute the Jensen-Shannon divergence between the observation and the simulations for this hour
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(divergences.jensen_shannon_div, gaussian_kde(x_obs), gaussian_kde(sim)) for sim in simulations]
            means_for_hour[i] = np.nanmean([future.result() for future in futures])

        start = start + datetime.timedelta(days=1)
        print(f"hour {hour}, day {i} completed")


    return means_for_hour


def compute_prior_divergence_for_hour(df: pd.DataFrame, prior_filename: str, start: datetime.datetime, days: int, hour: int) -> None:
    """
    Computes the Jensen-Shannon divergence of the prior for a specific hour over all days of the given dataset.
    
    Args:
        df (dataframe): The dataset
        prior_filename (str): The filename of the prior
        start (datetime): The start date
        days (int): The number of days
        hour (int): The hour of the day
    
    Returns:
        means_for_hour (np.array): The mean Jensen-Shannon divergence at the given hour for each day
    
    """
    # simulation parameters
    time_span = 60
    dt = 0.01
    n_workers = cpu_count()
    number_of_samples = 100

    # read in the prior
    in_s = open("out/" + prior_filename, "rb")
    try:
        prior = pickle.load(in_s)
    finally:
        in_s.close()

    start = start + datetime.timedelta(hours=hour)
    means_for_hour = np.zeros(days)
    for i in range(0, days):
        print(f"prior: hour {hour}, day {i} starting")

        # we select the 1-minute long observation starting at the given hour mark
        x_obs = df[str(start):str(start + datetime.timedelta(seconds=59))]["angular_velocity"].to_numpy()

        # check if the observation is empty or contains NaN values. If yes, we skip this observation
        if (x_obs.size == 0 or np.any(np.isnan(x_obs))):
            means_for_hour[i] = np.nan
            start = start + datetime.timedelta(days=1)

            continue

        # sample from the prior and simulate the trajectories
        # note: prior does not take observation into consideration during sampling. Therefore the sampling is independent from the data, the start date and the specific hour of the day
        simulations = batch_simulator(build_theta_for_batch_simulator_from_samples_tensor(prior.sample((number_of_samples, )), inertia=True), time_span, dt, n_workers)[1][:, ::int(1/dt)]
        print(f"prior: hour {hour}, day {i} simulations completed")

        # compute the Jensen-Shannon divergence between the observation and the simulations for this hour
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(divergences.jensen_shannon_div, gaussian_kde(x_obs), gaussian_kde(sim)) for sim in simulations]
            means_for_hour[i] = np.nanmean([future.result() for future in futures])

        start = start + datetime.timedelta(days=1)
        print(f"prior: hour {hour}, day {i} completed")

    return means_for_hour


def compute_plot_data_CE(days: int) -> None:
    """
    Computes the data needed for the divergence plot for Continental Europe.
    Reads in the dataset, computes the Jensen-Shannon divergence for each hour of the day and saves the results to a file.

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
    posterior_filename = "posterior_NPE_inertia_60s_dt0.01.pickle"
    means_for_hours_1 = [compute_divergence_for_hour(df, posterior_filename, start, days, i) for i in range(0, 24)]
    with open("means_for_hours_CE_inertia.pkl", "wb") as file:
        pickle.dump(means_for_hours_1, file)

    # compute plot data for the prior
    prior_filename = "prior_NPE_inertia_60s_dt0.01.pickle"
    means_for_hours_prior = [compute_prior_divergence_for_hour(df, prior_filename, start, days, i) for i in range(0, 24)]
    with open("means_for_hours_CE_inertia_prior.pkl", "wb") as file:
        pickle.dump(means_for_hours_prior, file)


def compute_plot_data_Mallorca(days: int) -> None:
    """
    Computes the data needed for the divergence plot for the Balearic grid.
    Reads in the dataset, computes the Jensen-Shannon divergence for each hour of the day and saves the results to a file.

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
    posterior_filename = "posterior_NPE_inertia_60s_dt0.01.pickle"
    means_for_hours_1 = [compute_divergence_for_hour(df, posterior_filename, start, days, i) for i in range(0, 24)]
    with open("means_for_hours_ES_PM_inertia.pkl", "wb") as file:
        pickle.dump(means_for_hours_1, file)
    
    # compute plot data for the prior
    prior_filename = "prior_NPE_inertia_60s_dt0.01.pickle"
    means_for_hours_prior = [compute_prior_divergence_for_hour(df, prior_filename, start, days, i) for i in range(0, 24)]
    with open("means_for_hours_ES_PM_inertia_prior.pkl", "wb") as file:
        pickle.dump(means_for_hours_prior, file)



def generate_plot(means_for_hours_1: List[float], means_for_hours_prior: List[float], filename: str) -> None:
    """
    Generates the plot for the Jensen-Shannon divergence for each hour of the day.
    Expects the data produced by the compute_plot_data functions as input.
    
    Args:
        means_for_hours_1 (list): The list of mean Jensen-Shannon divergences for each hour of the day for the posterior
        means_for_hours_prior (list): The list of mean Jensen-Shannon divergences for each hour of the day for the prior
        filename (str): The filename of the plot pdf-file
        
    Returns:
        None
    """
    x = range(0, 24 + 1, 1)
    plt.figure(figsize=(8, 5))

    # plot the mean and the 25th and 75th percentile for the posterior
    mean = [np.nanmean(means_for_hours_1[i]) for i in range(0, len(means_for_hours_1))]
    q25 = [np.nanpercentile(means_for_hours_1[i], 25) for i in range(0, len(means_for_hours_1))]
    q75 = [np.nanpercentile(means_for_hours_1[i], 75) for i in range(0, len(means_for_hours_1))]
    plt.plot(x, np.append(mean, mean[-1]), drawstyle="steps-post", color="tab:blue")
    plt.fill_between(x, np.append(q25, q25[-1]), np.append(q75, q75[-1]), step="post", alpha=0.5, color="tab:blue")
    plt.axhline(y=np.nanmean(mean), color="tab:blue", label="mean", linestyle="--")

    # plot the mean and the 25th and 75th percentile for the prior
    mean_prior = [np.nanmean(means_for_hours_prior[i]) for i in range(0, len(means_for_hours_prior))]
    q25_prior = [np.nanpercentile(means_for_hours_prior[i], 25) for i in range(0, len(means_for_hours_prior))]
    q75_prior = [np.nanpercentile(means_for_hours_prior[i], 75) for i in range(0, len(means_for_hours_prior))]
    plt.plot(x, np.append(mean_prior, mean_prior[-1]), drawstyle="steps-post", color="tab:green")
    plt.fill_between(x, np.append(q25_prior, q25_prior[-1]), np.append(q75_prior, q75_prior[-1]), step="post", alpha=0.5, color="tab:green")
    plt.axhline(y=np.nanmean(means_for_hours_prior), color="tab:green", label="mean", linestyle="--")

    # plot settings, including legend
    plt.grid()
    plt.xticks([0, 6, 12, 18, 24])
    plt.xticks(np.arange(0, 25, 1), minor=True)
    plt.xlabel("Hour of the day")
    plt.ylabel("Jensen-Shannon divergence")
    plt.ylim(ymin=0)
    legend_handles = [Line2D([0], [0], color="tab:green", lw=2, label="prior"),
                      Line2D([0], [0], color="tab:blue", lw=2, label="general posterior")]
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close("all")

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
    means_for_hours_1 = pickle.load(open("means_for_hours_CE_inertia.pkl", "rb"))
    means_for_hours_prior = pickle.load(open("means_for_hours_CE_inertia_prior.pkl", "rb"))
    generate_plot(means_for_hours_1, means_for_hours_prior, "figures/empirical_evaluation_CE_inertia_divergence_v3.pdf")


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
    # read in the plot data, which has been generated previously, and generate the plot for Mallorca
    means_for_hours_1 = pickle.load(open("means_for_hours_ES_PM_inertia.pkl", "rb"))
    means_for_hours_prior = pickle.load(open("means_for_hours_ES_PM_inertia_prior.pkl", "rb"))
    generate_plot(means_for_hours_1, means_for_hours_prior, "figures/empirical_evaluation_ES_PM_inertia_divergence_v3.pdf")


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
