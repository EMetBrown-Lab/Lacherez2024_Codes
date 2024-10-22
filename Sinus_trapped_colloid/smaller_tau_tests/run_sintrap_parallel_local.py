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
t = np.hstack([np.arange(10**i, 10 ** (i + 1), 10**i) for i in range(7)])

def MSD(x, t):
    msd = np.zeros(len(t))
    for n, i in enumerate(t):
        msd[n] = np.nanmean((x[:-i] - x[i:]) ** 2)
    return msd

def pdf(q, bins=100, density=True):
    pdf_q, bins_edge_q = np.histogram(q, bins=bins, density=density)
    bins_center_q = (bins_edge_q[0:-1] + bins_edge_q[1:]) / 2
    return pdf_q, bins_center_q

def main(n, data_dict, N, g_11, g_12, g_22, tau, kappa, alpha, l, output_file):
    data = copy.deepcopy(data_dict)
    seeds = np.random.randint(0, 10000, size=data_dict["N_trajectories"])

    for _ in range(data_dict["N_trajectories"]):
        q1, q2 = trajectory_sin_trap(N, g_11, g_12, g_22, tau, kappa, alpha, l)
        msd_q1, msd_q2 = MSD(q1, t), MSD(q2, t)

        q1_histo, q1_bins_center = pdf(q1 % (l), bins=20, density=True)
        q2_histo, q2_bins_center = pdf(q2, bins=50, density=True)

        data["mean_msd_q1"] += msd_q1
        data["mean_msd_q2"] += msd_q2
        data["mean_q1_histo"] += q1_histo
        data["q1_bins_center"] = q1_bins_center
        data["mean_q2_histo"] += q2_histo
        data["q2_bins_center"] = q2_bins_center

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
        "mean_q1_histo": np.zeros(20),
        "q1_bins_center": np.zeros(20),
        "mean_q2_histo": np.zeros(50),
        "q2_bins_center": np.zeros(50),
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
        futures = [executor.submit(main, i, data_dict, N, g_11, g_12, g_22, tau, kappa, alpha, l, output_file) for i in range(n_tasks)]

        for future in futures:
            result = future.result()  
            print(result)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main_execution()
