import pickle
import copy
import numpy as np
import scipy.signal as sp
from scipy.integrate import trapezoid
from tm_sinusoid_trap_pack import trajectory_sin_trap
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import os



########## TIME FUNCTIONS ##########
def time_lag_array(length_trajectory, n_points, lower_bound):
    """Returns the time lags used for convergence tests."""
    time_array = np.unique(np.logspace(
        np.log(lower_bound) / np.log(10),
        np.log(length_trajectory) / np.log(10),
        n_points,
        base=10,
    ).astype(int))
    return time_array

def tau_star(l, D_LJ) :
    """Characteristic time of second part of MSD(q1)."""
    _tau_star = l**2 / D_LJ
    return _tau_star

def total_simu_time(l, D_LJ) :
    """Returns the total duration in seconds of the simulation."""
    total_simulation_time = 100 * tau_star(l, D_LJ)
    return total_simulation_time

def num_steps(total_simu_time, time_step) :
    """ Returns the integer number of time steps the simulation will be running."""
    return int(total_simu_time/time_step)

def simulation_time_step(mu_11, mu_22, l, kappa_2) :
    """Dynamically choose the time step to avoid inconsistent precision between simulations."""
    kappa_1 = (0.5 * (2 * np.pi / l)**2)
    t1 = kappa_1 * mu_11
    t2 = kappa_2 * mu_22
    time_step = 0.01 * min(t1, t2)
    return time_step


########## DATA COLLECTION ##########

def displacements(q, time_lag):
    """Calculate the displacements."""
    print(len(q), time_lag)
    if time_lag >= len(q):
        raise ValueError("Gap size must be less than the length of the displacement array.")
    dq_time_lag = q[int(time_lag):] - q[:-int(time_lag)]
    return dq_time_lag


def bins_edges_and_centers(sigma, l):
    """Returns the bin edges and bin centers for a set of parameters."""
    num_bins = num_bin_calculation(sigma, l) 
    data_range = (-6 * sigma, 6 * sigma) 
    synthetic_data = np.array([data_range])
    _, bin_edges = np.histogram(synthetic_data, bins=num_bins, range=data_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers, num_bins


def num_bin_calculation(sigma, l) :
    """In case you want to make it dynamic; I found it to be unnecessary."""
    return 31


def sigma_calculation(N, tau, gamma_11, gamma_12, gamma_22, kappa, alpha, l, time_lag) :
    Deff_LJ = D_eff_Lifson_Jackson(gamma_11, alpha, l)
    sigma = np.sqrt(2*(Deff_LJ) * tau * time_lag)
    return sigma


########## SIMULATION FUNCTIONS ##########

def phi_1(q, alpha, l):
    """Potential phi_1."""
    return alpha * np.sin(2 * np.pi * q / l)


def phi_prime_1(q, alpha, l):
    """Derivative of the potential phi_1."""
    return (2 * np.pi * alpha / l) * np.cos(2 * np.pi * q / l)


def D_eff_Lifson_Jackson(gamma_11, alpha, l) :
    """Typical value of the long-time diffusion coefficient for a particle 
    escaping a sinusoidal trap with thermal fluctiations."""
    q1_th = np.linspace(0, l, 100000)
    Deff = l**2  / (
        np.trapz(np.exp(phi_1(q1_th, alpha, l)), q1_th) * np.trapz(np.exp(- phi_1(q1_th, alpha, l)), q1_th) * gamma_11
        )
    return Deff



def friction_matrix(g_11, g_12, g_22):
    """Calculate the friction matrix from g coefficients."""
    gamma_11 = g_11 ** 2 + g_12 ** 2
    gamma_12 = g_12 * (g_11 + g_22)
    gamma_22 = g_22 ** 2 + g_12 ** 2
    return gamma_11, gamma_12, gamma_22

def mu_matrix_fromgamma (gamma_11, gamma_12, gamma_22) :
    """Calculate the mobility matrix from friciton coefficients."""
    mu_11 =   (gamma_11 * gamma_22 - gamma_12**2)**(-1) * gamma_22
    mu_12 = - (gamma_11 * gamma_22 - gamma_12**2)**(-1) * gamma_12
    mu_22 =   (gamma_11 * gamma_22 - gamma_12**2)**(-1) * gamma_11
    return mu_11, mu_12, mu_22


def fill_displacements_pdf(q, tau, alpha, l, kappa, gamma_11, gamma_12, gamma_22, time_lag):
    """Bins are fixed for a given set of parameters."""
    dq = displacements(q, time_lag)
    Deff_LJ = D_eff_Lifson_Jackson(gamma_11, alpha, l)
    sigma = np.sqrt(2 * Deff_LJ  * tau * time_lag)  
    bin_edges, bin_centers, num_bins = bins_edges_and_centers(sigma, l)
    hist_values, bin_edges = np.histogram(dq, bins=bin_edges, density=False)
        
    return bin_centers, hist_values, num_bins


def main_task(args):
    """Main execution. Function computes trajectories, then the displacements for 
    each selected time step, and fills the corresponding histograms which are returned."""
    n, data_dict, g_11, g_12, g_22, kappa, alpha, l, output_file = args
    data = copy.deepcopy(data_dict)


    gamma_11, gamma_12, gamma_22 = friction_matrix(g_11, g_12, g_22)
    mu_11, _, mu_22 = mu_matrix_fromgamma(gamma_11, gamma_12, gamma_22)
    tau = simulation_time_step(mu_11, mu_22, l, kappa)
    D_LJ = D_eff_Lifson_Jackson(gamma_11, alpha, l)
    total_simulation_time = total_simu_time(l, D_LJ)
    time_step = simulation_time_step(mu_11, mu_22, l, kappa)
    N = num_steps(total_simulation_time, time_step)

    
    time_lags = time_lag_array(N, 20,  int(tau_star(l, D_LJ)/tau))



    for _ in range(data_dict["N_trajectories"]):
        q1, q2 = trajectory_sin_trap(N, g_11, g_12, g_22, tau, kappa, alpha, l)
        for time_lag in time_lags:
            bin_centers, hist_values, num_bins = fill_displacements_pdf(
                q1, tau, alpha, l, kappa, gamma_11, gamma_12, gamma_22, time_lag
            )
            data[f'histo_dq1_{time_lag}'] += (data_dict["N_trajectories"]) * hist_values
            data[f'bins_dq1_{time_lag}'] = bin_centers


    filename = output_file + "_" + str(n) + ".pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return f"Task {n} completed"

def main_execution():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("-g11", "--g_11", type=float, default=2., help="g matrix 00 val. (default: 1.0)")
    parser.add_argument("-g12", "--g_12", type=float, default=-2., help="g matrix 01 val. (default: 0.1)")
    parser.add_argument("-g22", "--g_22", type=float, default=4, help="g matrix 11 val. (default: 5.0)")
    parser.add_argument("-k", "--kappa", type=float, default=100., help="Compliance. (default: 0.001)")
    parser.add_argument("-alpha", "--alpha", type=float, default=1.0, help="Trap depth. (default: 1.0)")
    parser.add_argument("-l", "--l", type=float, default=1.0, help="Trap width. (default: 1.0)")
    parser.add_argument("-o", "--output", type=str, default="save", help="Output directory")
    parser.add_argument("-Nt", "--trajectory_per_task", type=int, default=10, help="Number of trajectories per task")
    parser.add_argument("-nt", "--n_tasks", type=int, default=10, help="Number of tasks to run")
    parser.add_argument("-np", "--n_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("-j", "--job_ID", type=int, default=10, help="JOB ID number")
    args = parser.parse_args()

    g_11 = args.g_11
    g_12 = args.g_12
    g_22 = args.g_22
    kappa = args.kappa
    alpha = args.alpha
    l = args.l
    output_dir = args.output
    N_trajectories = args.trajectory_per_task
    n_tasks = args.n_tasks
    job_ID = args.job_ID


    gamma_11, gamma_12, gamma_22 = friction_matrix(g_11, g_12, g_22)
    mu_11, _, mu_22 = mu_matrix_fromgamma(gamma_11, gamma_12, gamma_22)
    tau = simulation_time_step(mu_11, mu_22, l, kappa)
    D_LJ = D_eff_Lifson_Jackson(gamma_11, alpha, l)
    total_simulation_time = total_simu_time(l, D_LJ)
    N = num_steps(total_simulation_time, tau)


    time_lags = time_lag_array(N, 20,  int(tau_star(l, D_LJ)/tau))


    output_dir += f"_{job_ID}"

    data_dict = {}


    # Initialize histograms and bins for each time_lag using the computed num_bins
    for time_lag in time_lags:
        data_dict[f"num_bins_{time_lag}"] = num_bin_calculation(
            sigma_calculation(N, tau, gamma_11, gamma_12, gamma_22, kappa, alpha, l, time_lag), l
        )
        num_bins = data_dict[f"num_bins_{time_lag}"]
        data_dict[f"histo_dq1_{time_lag}"] = np.zeros(num_bins)
        data_dict[f"bins_dq1_{time_lag}"] = np.zeros(num_bins)

    # Add the number of trajectories
    data_dict["N_trajectories"] = N_trajectories
    data_dict["tau"] = tau

    # Construct the output file path
    output_file = os.path.join(
        output_dir,
        '-'.join(
            f'{key}_{value}' for key, value in {
                "N": N, "tau": tau, "g11": g_11, "g12": g_12, "g22": g_22,
                "kappa": kappa, "alpha": alpha, "l": l, "Ntrajectories": N_trajectories, "Job_ID": job_ID,
            }.items()
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    args_list = [
        (n, data_dict, g_11, g_12, g_22,  kappa, alpha, l, output_file)
        for n in range(n_tasks)
    ]

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(main_task, args_list))
    t2 = time.time()

    for result in results:
        print(result)

    print(t2 - t1)

if __name__ == "__main__":
    main_execution()
