import itertools
import os
import subprocess

# Define parameters for the script
Ns = [10_000_000]  # Points per trajectory
taus = [1e-4]#, 1e-4]
g11s = [1.0]
g12s = [.8, 1.0, 1.3, 1.7, 2.0, 2.2, 2.4, 2.6]
g22s = [10.]
ks = [0.0005, 0.0008, 0.001,  0.0013]
alphas = [1.0, 2.0]
ls = [.5, 1.0, 1.2]
n_tasks_list = [10] # Number of groups of trajectories
trajectories_per_task_list = [10]  # Number of trajectories in each task
n_workers = 13  # Number of workers for parallel execution


# Define the base output directory
base_output_dir = "output_file"

# Ensure the base output directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Iterate through all combinations of parameters
for N, tau, g11, g12, g22, k, alpha, l, n_tasks, trajectories_per_task in itertools.product(
    Ns, taus, g11s, g12s, g22s, ks, alphas, ls, n_tasks_list, trajectories_per_task_list
):
    # Create a unique output folder for each combination
    output_subdir = f"{base_output_dir}/{N}_{tau}_{g11}_{g12}_{g22}_{k}_{alpha}_{l}_{n_tasks}_{trajectories_per_task}"
    os.makedirs(output_subdir, exist_ok=True)

    # Command to run the main simulation script with the current parameter set
    command = [
        "python3", "run_sintrap_parallel_local.py", 
        "--num_steps", str(N),
        "--time_step", str(tau),
        "--g_11", str(g11),
        "--g_12", str(g12),
        "--g_22", str(g22),
        "--kappa", str(k),
        "--alpha", str(alpha),
        "--l", str(l),
        "--output", output_subdir,
        "--trajectory_per_task", str(trajectories_per_task),
        "--n_tasks", str(n_tasks),
        "--n_workers", str(n_workers)
    ]

    # Execute the command
    subprocess.run(command, check=True)

print("All simulations have been executed successfully.")
