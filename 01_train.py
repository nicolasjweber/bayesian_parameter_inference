"""
Runs the posterior estimation for the simulation-based Bayesian parameter inference with neural networks for the Aggregated Swing Equation (ASE) model.
Parses the command-line arguments and runs the posterior estimation based on the specified arguments, 
either for parameter estimation with fixed inertia or with constrained frequency control (= inertia estimation).
Supports different prior distributions and inference algorithms.
The prior and posterior distributions are saved to files in the output directory.

Usage Example: python 01_train.py --parameter_estimation

Author: Nicolas Joschua Weber
Date: 2025-03-04

This file is part of the thesis "Data-Driven Bayesian Parameter Estimation with Neural Networks for Power Grid Frequency" at KIT.
"""

from typing import Any
import argparse
import pickle
import sys
import os
import torch
import pandas as pd
from numpy import pi
from sbi import utils
from sbi.inference import NPE, NRE_A, NLE
from sbi.utils.user_input_checks import process_prior
from sbi.neural_nets import posterior_nn
from src.utils.simulator import batch_simulator
from helper_functions import build_theta_for_batch_simulator_from_samples_tensor


def parse_arguments():
    """ 
    Parses the command line arguments.

    Args:
        None

    Returns:
        args (Namespace): The parsed arguments
    """

    parser = argparse.ArgumentParser(description="Estimates neural posteriors for parameter inference of the Aggregated Swing Equation (ASE) model.")
    parser.add_argument("--parameter_estimation", action="store_true", help="Train posterior for parameter estimation (with inertia constant H set to 1)")
    parser.add_argument("--parameter_estimation_daytime_specific", type=str, metavar="CSV_FILE", help="Train posteriors for parameter estimation (with inertia constant H set to 1) for each hour of the day, with prior parameter ranges provided in a csv file")
    parser.add_argument("--parameter_estimation_synthetical", action="store_true", help="Train posterior for parameter estimation (with inertia constant H set to 1) on synthetical data")
    parser.add_argument("--inertia_estimation", action="store_true", help="Train posterior for inertia estimation (with c_1 = c_2 = 0 and time_span = 60)")
    parser.add_argument("--specific_observation", type=float, nargs = 2, metavar=("P0", "P1"), help="Train posterior for parameter estimation of a specific observation with given P0 and P1")
    parser.add_argument("--number_of_samples", type=int, default=20000, help="Number of samples drawn from the prior")
    parser.add_argument("--seed", type=int, default=1, help="Manual seed (for reproducibility)")
    parser.add_argument("--density_estimator", type=str, default="maf", help="Density estimator to use (maf, made, nsf, mdn) for NPE and NLE")
    parser.add_argument("--classifier", type=str, default="resnet", help="Classifier to use (linear, mlp, resnet) for NRE")
    parser.add_argument("--hidden_features", type=int, default=50, help="Number of hidden features")
    parser.add_argument("--num_transforms", type=int, default=5, help="Number of transforms (only relevant if density_estimator is maf or nsf)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers for the simulator (default is all available cores)")
    parser.add_argument("--batch_size", type=float, default=-1, help="Batch size for the simulator")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep for the simulator")
    parser.add_argument("--time_span", type=int, default=900, help="Time span of the simulations in seconds (ignored for inertia estimation)")
    parser.add_argument("--output_dir", type=str, default="out", metavar="DIR", help="Output directory")
    parser.add_argument("--algorithm", type=str, default="NPE", help="Inference algorithm to use (NPE, NRE, NLE)")

    if len(sys.argv) < 2:
        # no command line arguments have been provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


# Parse the command line arguments
args = parse_arguments()

# Set torch seed for reproducibility
torch.manual_seed(args.seed)



def run(prior: torch.distributions.Distribution, time_span: int, inertia: bool = False, specific_observation: bool = False, specific_p0: float = 0.0, specific_p1: float = 0.0) -> None:
    """ 
    Estimates the posterior distribution for the given prior distribution and time span using the inference algorithm 
    specified in the command line arguments (NPE, NRE, NLE).

    Args:
        prior (Distribution): The prior distribution
        time_span (int): The time span of the simulations in seconds
        inertia (bool): Specifies whether inertia estimation is conducted or not
        specific_observation (bool): Specifies whether parameter estimation for a specific observation is conducted or not
        specific_p0 (float): The specific p0 value of the given observation
        specific_p1 (float): The specific p1 value of the given observation

    Returns:
        posterior (NeuralPosterior): The resulting posterior distribution
    """
    samples_prior = prior.sample((args.number_of_samples,))
    simulation_results_prior = batch_simulator(build_theta_for_batch_simulator_from_samples_tensor(samples_prior, inertia, specific_observation, specific_p0, specific_p1), time_span, args.dt, args.num_workers, args.batch_size)

    prior, _, _ = process_prior(prior)

    if args.algorithm == "NPE":
        density_estimator = posterior_nn(model=args.density_estimator, hidden_features=args.hidden_features, num_transforms=args.num_transforms)
        inference = NPE(prior=prior, density_estimator=density_estimator)
    elif args.algorithm == "NRE":
        inference = NRE_A(prior=prior, classifier=args.classifier)
    elif args.algorithm == "NLE":
        inference = NLE(prior=prior, density_estimator=args.density_estimator)
    else:
        raise ValueError("Specified algorithm is not supported yet. Please choose NPE, NRE or NLE instead.")

    inference = inference.append_simulations(samples_prior, simulation_results_prior[1][:, ::int(1/args.dt)])
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    return posterior


def save(filename: str, obj: Any) -> None:
    """ 
    Saves the given object to a file.

    Args:
        filename (str): The filename
        obj (object): The object to be saved

    Returns:
        None
    """
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, filename + ".pickle"), "wb") as out_s:
        pickle.dump(obj, out_s)



def inertia_estimation_posterior() -> None:
    """
    Estimates the posterior distribution for parameter estimation with constrained frequency control and saves the it to a file.

    For inertia estimation, the parameters for primary and secondary c_1 and c_2 are set to 0 and we only look at intervals of 60 seconds length.
    The format of the prior is as follows: (omega_0, P_0, P_1, epsilon, H)
    
    Args:
        None
        
    Returns:
        None
    """
    prior_inertia = utils.BoxUniform(
        low=torch.tensor([-0.6, -0.05, -0.01, 0, 1]),
        high=torch.tensor([0.6, 0.05, 0.01, 0.01, 11])
    )
 
    time_span_inertia = 60
    posterior_inertia = run(prior_inertia, time_span_inertia, inertia=True)

    filename_prior = f"prior_{args.algorithm}_inertia_{time_span_inertia}s_dt{args.dt}"
    save(filename_prior, prior_inertia)

    filename_posterior = f"posterior_{args.algorithm}_inertia_{time_span_inertia}s_dt{args.dt}"
    save(filename_posterior, posterior_inertia)


def parameter_estimation_posterior() -> None:
    """
    Estimates the posterior distribution for parameter estimation with fixed inertia and saves it to a file.
    
    For parameter estimation, the inertia constant is set to 1 and the format of the prior is as follows: 
    (omega_0, phi_0, c_1, c_2, P_0, P_1, epsilon)
    
    Args:
        None
        
    Returns:
        None
    """
    prior = utils.BoxUniform(
        low=torch.tensor([-0.6, -pi, 0.0001, 0.000001, -0.05, -0.01, 0]),
        high=torch.tensor([0.6, pi, 0.1, 0.001, 0.05, 0.01, 0.05])
    )
    
    posterior = run(prior, args.time_span)

    filename_prior = f"prior_{args.time_span}s"
    save(filename_prior, prior)

    filename_posterior = f"posterior_{args.algorithm}_{args.time_span}s_dt{args.dt}"
    save(filename_posterior, posterior)


def parameter_estimation_posterior_60s() -> None:
    """
    Estimates the posterior distribution for parameter estimation with fixed inertia and a time span of 60 seconds.
    
    For parameter estimation, the inertia constant is set to 1 and the format of the prior is as follows:
    (omega_0, phi_0, c_1, c_2, P_0, P_1, epsilon)

    This method is only used for testing purposes to see how well the performance is on short time intervals. 
    The noise is reduced compared to the above parameter estimation due to the shorter interval.
    
    Args:
        None
        
    Returns:
        None
    """
    prior = utils.BoxUniform(
        low=torch.tensor([-0.6, -pi, 0.0001, 0.000001, -0.05, -0.01, 0]),
        high=torch.tensor([0.6, pi, 0.1, 0.001, 0.05, 0.01, 0.01])
    )
    
    time_span = 60
    posterior = run(prior, time_span)

    filename = f"posterior_{args.algorithm}_{time_span}s_dt{args.dt}"
    save(filename, posterior)


def parameter_estimation_posterior_synthetic() -> None:
    """ 
    Estimates the posterior distribution for parameter estimation with a uniform prior whose parameter ranges are centered around theta_true.
    The resulting posterior is saved to a file and can be used for synthetic evaluation.

    For parameter estimation, the inertia constant is set to 1 and the format of the prior is as follows:
    (omega_0, phi_0, c_1, c_2, P_0, P_1, epsilon)
    
    Args:
        None
    
    Returns:
        None
    """
    theta_true = torch.tensor([0.1, 0.1, 0.2, 0.1, 0.1, 0.001, 0.001])

    prior_1 = utils.BoxUniform(low=0.1 * theta_true,
                               high=1.9 * theta_true)
    filename_prior_1 = f"prior_{args.algorithm}_synthetic1_{args.time_span}s_dt{args.dt}"
    save(filename_prior_1, prior_1)
    
    posterior_1 = run(prior_1, args.time_span)
    filename_posterior_1 = f"posterior_{args.algorithm}_synthetic1_{args.time_span}s_dt{args.dt}"
    save(filename_posterior_1, posterior_1)

    prior_2 = utils.BoxUniform(low=0.9 * theta_true,
                               high=1.1 * theta_true)
    filename_prior_2 = f"prior_{args.algorithm}_synthetic2_{args.time_span}s_dt{args.dt}"
    save(filename_prior_2, prior_2)

    posterior_2 = run(prior_2, args.time_span)
    filename_posterior_2 = f"posterior_{args.algorithm}_synthetic2_{args.time_span}s_dt{args.dt}"
    save(filename_posterior_2, posterior_2)


def specific_observation_posterior(specific_p0: float, specific_p1: float) -> None:
    """ 
    Estimates the posterior distribution for parameter estimation given fixed, empirical values for the parameters P_0 and P_1 and saves it to a file.

    For parameter estimation, the inertia constant is set to 1 and the format of the prior is as follows:
    (omega_0, phi_0, c_1, c_2, epsilon, proportionality_factor)

    The proportionality factor is introduced to allow the empirical values of P_0 and P_1 to be rescaled.
    
    Args:
        specific_p0 (float): The specific p0 value of the given observation
        specific_p1 (float): The specific p1 value of the given observation
        
    Returns:
        None
    """
    prior = utils.BoxUniform(
        low=torch.tensor([-0.6, -pi, 0.0001, 0.000001, 0, -0.1]),
        high=torch.tensor([0.6, pi, 0.1, 0.001, 0.05, 0.1])
    )
    
    posterior = run(prior, args.time_span, specific_observation=True, specific_p0=specific_p0, specific_p1=specific_p1)

    filename = f"posterior_{args.algorithm}_specific_observation_{args.time_span}s_dt{args.dt}_p0{specific_p0}_p1{specific_p1}"
    save(filename, posterior)
    

def parameter_estimation_posterior_daytime_specific() -> None:
    """ 
    Estimates daytime-specific posterior distributions for parameter estimation and saves both the prior and posterior distributions to files.
    
    For parameter estimation, the inertia constant is set to 1 and the format of the prior is as follows:
    (omega_0, phi_0, c_1, c_2, P_0, P_1, epsilon)
    
    The prior parameter ranges for primary and secondary control parameters c_1 and c_2 are provided in the csv file specified in the command line arguments.
    The file needs to contain the columns c1_min, c1_max, c2_min, c2_max and the rows need to correspond to the 24 hours of the day.

    Args:
        None
        
    Returns:
        None
    """
    df = pd.read_csv(args.parameter_estimation_daytime_specific)

    for index, row in df.iterrows():
        print(f"Calculating posterior for hour {index}")
        prior = utils.BoxUniform(
            low=torch.tensor([-0.6, -pi, 0.5 * row["c1_min"], 0.5 * row["c2_min"], -0.05, -0.01, 0]),
            high=torch.tensor([0.6, pi, 2 * row["c1_max"], 2 * row["c2_max"], 0.05, 0.01, 0.05])
        )
        
        posterior = run(prior, args.time_span)

        filename_prior = f"prior_{args.time_span}s_hour_{index}"
        filename_posterior = f"posterior_{args.algorithm}_{args.time_span}s_hour_{index}_dt{args.dt}"

        save(filename_prior, prior)
        save(filename_posterior, posterior)


def main() -> None:
    """
    Main method which runs the posterior estimation based on the specified command line arguments.

    Args:
        None

    Returns:
        None
    """
    if args.parameter_estimation:
        print("Parameter Estimation with Fixed Inertia")
        parameter_estimation_posterior()

    if args.parameter_estimation_daytime_specific:
        print("Parameter Estimation with Fixed Inertia (Daytime-Specific)")
        parameter_estimation_posterior_daytime_specific()
     
    if args.parameter_estimation_synthetical:
        print("Parameter Estimation on Synthetical Data")
        parameter_estimation_posterior_synthetic()

    if args.inertia_estimation:
        print("Parameter Estimation with Constrained Frequency Control / Inertia Estimation")
        inertia_estimation_posterior()

    if args.specific_observation is not None:
        print("Parameter Estimation for a Specific Observation")
        print(f"P_0 is {args.specific_observation[0]}")
        print(f"P_1 is {args.specific_observation[1]}")

        specific_observation_posterior(specific_p0=args.specific_observation[0],
                                       specific_p1=args.specific_observation[1])


if __name__ == "__main__":
    main()
