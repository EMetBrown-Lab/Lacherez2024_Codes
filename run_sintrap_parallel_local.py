import pickle
import copy
import numpy as np
import scipy.signal as sp
from tm_sinusoid_trap_pack import trajectory_sin_trap
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import os

# Define the time lags for MSD plots
t = np.hstack([np.arange(10**i, 10 ** (i + 1), 10**i) for i in range(8)])

def MSD(x, t):
    msd = np.zeros(len(t))
    for n, i in enumerate(t):
        msd[n] = np.nanmean((x[:-i] - x[i:]) ** 2)
    return msd

def pdf(q, bins=100, density=True):
    pdf_q, bins_edge_q = np.histogram(q, bins=bins, density=density)
    bins_center_q = (bins_edge_q[0:-1] + bins_edge_q[1:]) / 2
    pdf_q /= np.trapz(pdf_q, bins_center_q)
    return pdf_q, bins_center_q

def displacement_pdf(q, gap, bins=10):
    """
    Compute the displacement differences at a specified gap size and plot the PDF.

    Parameters:
    q (numpy array): The displacement data.
    gap (int): The gap size to compute the displacement differences.
    bins (int): Number of bins for the histogram (default is 10).
    """
    # Ensure gap is less than the length of q
    if gap >= len(q):
        raise ValueError("Gap size must be less than the length of the displacement array.")

    # Compute the displacement differences at the given gap
    dq_gap = q[gap:] - q[:-gap]

    # Compute the histogram (frequency or density) and bin edges
    hist_values, bin_edges = np.histogram(dq_gap, bins=bins, density=True)

    # Calculate the bin centers by averaging the bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, hist_values




def pdf_displacement(q, bins=100, density=True):
    # Calculate displacements
    dq = np.diff(q)
    
    # Calculate the PDF of the displacements
    pdf_dq, bins_edge_dq = np.histogram(dq, bins=bins, density=density)
    bins_center_dq = (bins_edge_dq[:-1] + bins_edge_dq[1:]) / 2
    pdf_dq /= np.trapz(pdf_dq, bins_center_dq)
    
    return pdf_dq, bins_center_dq




def main(n, data_dict, N, g_11, g_12, g_22, tau, kappa, alpha, l, output_file):
    data = copy.deepcopy(data_dict)
    seeds = np.random.randint(0, 10000, size=data_dict["N_trajectories"])

    for _ in range(data_dict["N_trajectories"]):
        q1, q2 = trajectory_sin_trap(N, g_11, g_12, g_22, tau, kappa, alpha, l)
        msd_q1, msd_q2 = MSD(q1, t), MSD(q2, t)

        dq1_histo, dq1_bins_center = pdf_displacement(q1, bins=50, density=True)
        dq2_histo, dq2_bins_center = pdf_displacement(q2, bins=50, density=True)

        data["mean_msd_q1"] += msd_q1
        data["mean_msd_q2"] += msd_q2
        data["mean_dq1_histo"] += dq1_histo
        data["dq1_bins_center"] = dq1_bins_center
        data["mean_dq2_histo"] += dq2_histo
        data["dq2_bins_center"] = dq2_bins_center

    filename = output_file + "_" + str(n) + ".pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return "Task completed"

def main_execution():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("-N", "--num_steps", type=int, default=20_000_000, help="Number of steps (default: 20000000)")
    parser.add_argument("-t", "--time_step", type=float, default=0.0001, help="Time step duration (default: 0.0001)")
    parser.add_argument("-g11", "--g_11", type=float, default=1.0, help="g matrix 00 val. (default: 1.0)")
    parser.add_argument("-g12", "--g_12", type=float, default=0.1, help="g matrix 01 val. (default: 0.1)")
    parser.add_argument("-g22", "--g_22", type=float, default=5.0, help="g matrix 11 val. (default: 5.0)")
    parser.add_argument("-k", "--kappa", type=float, default=0.001, help="Compliance. (default: 0.001)")
    parser.add_argument("-alpha", "--alpha", type=float, default=1.0, help="Trap depth. (default: 1.0)")
    parser.add_argument("-l", "--l", type=float, default=1.0, help="Trap width. (default: 1.0)")
    parser.add_argument("-o", "--output", type=str, default="save", help="Output directory")
    parser.add_argument("-Nt", "--trajectory_per_task", type=int, default=100, help="Number of trajectories per task")
    parser.add_argument("-nt", "--n_tasks", type=int, default=10, help="Number of tasks to run")
    parser.add_argument("-np", "--n_workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    # Assign parsed arguments to variables
    N = args.num_steps
    tau = args.time_step
    g_11 = args.g_11
    g_12 = args.g_12
    g_22 = args.g_22
    kappa = args.kappa
    alpha = args.alpha
    l = args.l
    output_dir = args.output
    N_trajectories = args.trajectory_per_task
    n_tasks = args.n_tasks
    n_workers = args.n_workers

    data_dict = {
        "mean_msd_q1": np.zeros(len(t)),
        "mean_msd_q2": np.zeros(len(t)),
        "mean_dq1_histo": np.zeros(50),
        "dq1_bins_center": np.zeros(50),
        "mean_dq2_histo": np.zeros(50),
        "dq2_bins_center": np.zeros(50),
        "N_trajectories": N_trajectories,
    }

    simulation_parameters = {
        "N": N, "tau": tau, "g11": g_11, "g12": g_12, "g22": g_22,
        "kappa": kappa, "alpha": alpha, "l": l, "Ntrajectories": N_trajectories
    }

    output_file = output_dir + "/" + '-'.join(f'{key}_{value}' for key, value in simulation_parameters.items())
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(main, i, data_dict, N, g_11, g_12, g_22, tau, 
                                   kappa, alpha, l, output_file) for i in range(n_tasks)]

        for future in futures:
            result = future.result()  
            print(result)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main_execution()
