"""
Aggregated Swing Equation Simulator Module

This module contains functions to simulate paths of a stochastic differential equation (SDE)
using the Euler-Maruyama method. Specifically the Aggregated Swing Equation (ASE), 
as a linear response model [1]:

    dω/dt = -c1/Hω - c2/Hφ + k(p0 + p1t)/H + εdW
    dφ/dt = ω

The main function `batch_simulator` allows users to run simulations in parallel, making efficient use
of computational resources. The `_ase_batch_solver` function is a helper function that performs
the actual numerical solution of the SDE for a batch of initial conditions and parameters.

We suggest to use "batch_size" equal to 1/3 of the number of CPU resources.

Original Author: Bastian Matthijs Bacher; May 9, 2024

Modified by: Nicolas Joschua Weber; March 4, 2025
Added inertia constant H and proportionality factor k to the model. 
Also fixed a bug in the automatic calculation of the batch size when n_simulations is small.

Usage Examples:
----------------
# Import the simulator function
from src.utils.simulator import batch_simulator

# Define the parameters in batch form
theta = torch.tensor([[0.0, 0.0, 1.0, 0.5, 0.1, 0.2, 0.01, 1, 1]])

# Set the simulation parameters
time_span = 10
dt = 0.01 
n_workers = 1 # Number of parallel workers, use os.cpu_count() for all available cores
batch_size = 1 # Recommended to use 1/3 of the number of CPU resources 

# Run the simulation
simulation_results = batch_simulator(theta, time_span, dt, n_workers, batch_size)

# The `simulation_results` tensor contains the simulated paths for the first variable (omega)

[1]: https://ieeexplore.ieee.org/document/8963682
"""

from logging import Logger
from typing import Callable, Tuple
from joblib import Parallel, delayed
import torch
from torch import Tensor
from sbi.simulators.simutils import tqdm_joblib
from tqdm.auto import tqdm


def _ase_batch_solver(thetas: Tensor, time_span: int, dt: float) -> Tensor:
    """
    Solves a batch of stochastic differential equations using the Euler-Maruyama method.
    Specifically a linear response formulation of the Aggregated Swing Equation (ASE) [1].

        dω/dt = -c_1*ω/H - c_2*φ/H + kΔP/H + εξ
        dφ/dt = ω

        where:
            ΔP := P_0 + P_1*t
            ξ := dW (Wiener Process increments)

    Args:
        thetas (Tensor): Parameters for the SDE, unpacked as (omega_0, phi_0, c_1, c_2, p_0, p_1, epsilon, H, k).
        time_span (int): Total time span for the simulation.
        dt (float): Timestep for the Euler-Maruyama method.

    Returns:
        Tensor: Simulated trajectories for both variables, shape (batch_size, T/dt, 2).

    [1]: https://ieeexplore.ieee.org/document/8963682
    """
    # Define batch size based on the initial conditions
    batch_size = thetas.shape[0]

    # Initialize time vector from 0 to T with steps of dt
    t = torch.linspace(0, time_span, int(time_span / dt))

    # Pre-allocate tensor for results with shape (batch_size, len(t), 2)
    x = torch.zeros(batch_size, len(t), 2)

    # Store initial conditions in the first timestep, from theta
    x[:, 0, :] = thetas[:, :2]

    # Prepare the Wiener Process increments for stochastic integration
    dW = torch.sqrt(torch.tensor([dt])) * torch.randn(batch_size, len(t) - 1)

    # Iterate over time steps using the Euler-Maruyama method
    for i in range(1, len(t)):
        x[:, i, 0] = (
            x[:, i - 1, 0]
            + (
                - thetas[:, 2] * x[:, i - 1, 0]  # -c_1 * ω
                - thetas[:, 3] * x[:, i - 1, 1]  # -c_2 * φ
                + thetas[:, 4] * thetas[:, 8] # p_0 * k
                + thetas[:, 5] * t[i - 1] * thetas[:, 8] # p_1 * t * k
            )
            * dt
            * (1 / thetas[:, 7]) # * (1 / H)
            + thetas[:, 6] * dW[:, i - 1]  # epsilon * dW
        )
        x[:, i, 1] = x[:, i - 1, 1] + x[:, i - 1, 0] * dt  # φ = ω * dt

    # Return the full tensor of results
    return x


def batch_simulator(
    thetas: Tensor,
    time_span: int,
    dt: float,
    n_workers: int = 1,
    batch_size: int = -1,
    show_progress_bar: bool = False,
    logger: Logger = None,
    solver: Callable = _ase_batch_solver,
) -> Tuple[Tensor, Tensor]:
    """
    Simulates a batch of stochastic differential equations in parallel.

    Args:
        thetas (Tensor): Parameters for the SDE in batch shape.
        time_span (int): Total time for the simulation.
        dt (float): Calculation timestep for the solver method.
        n_workers (int, optional): Number of parallel workers. Defaults to 1.
        batch_size (int, optional): Size of each batch. Defaults to -1 (automatic calculation).

    Returns:
        (Tensor, Tensor): Cleaned parameters and simulations with corresponding shapes (n, ), (n, time_span/dt).
    """
    # Find the total number of simulations to run
    n_simulations = len(thetas)

    if batch_size == -1:
        if n_workers == 1:
            batch_size = n_simulations
        else:
            # Recommended to use 1/3 of the number of CPU resources. If n_simulations is small, ensure batch_size >= 1.
            batch_size = max(1, n_simulations // (3 * n_workers))

    batches = thetas.split(batch_size)
    n_batches = len(batches)

    # Run the simulations in parallel, if n_workers > 1
    with tqdm_joblib(
        tqdm(
            batches,
            desc=f"Generating {n_simulations} simulations in {n_batches} batches.",
            total=n_batches,
            disable=not show_progress_bar,
        )
    ) as _:
        xs = Parallel(n_jobs=n_workers)(
            delayed(solver)(batch, time_span, dt) for batch in batches
        )

    # Concatenate the results
    xs = torch.cat(xs, dim=0)[:, :, 0]

    # Compute masks for NaNs and Infs
    nan_mask = torch.isnan(xs)
    inf_mask = torch.isinf(xs)

    # Count NaNs and Infs
    n_nans = nan_mask.sum().item()
    n_infs = inf_mask.sum().item()

    if logger is not None:
        logger.warn(f"Found {n_nans} NaNs and {n_infs} Infs in the simulation results.")
        logger.warn(
            f"Deleting {n_nans + n_infs}/{n_simulations} simulations, returning the rest."
        )

    # Combine NaN and Inf masks
    invalid_mask = nan_mask | inf_mask

    # Delete NaN and Inf simulations
    x_cleaned = xs[~invalid_mask.any(dim=1)]
    thetas_cleaned = thetas[~invalid_mask.any(dim=1)]

    # Return the cleaned parameters and simulations
    return thetas_cleaned, x_cleaned
