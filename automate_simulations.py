import os
import itertools
import subprocess
from math import sqrt 

# Generate parameter files
def generate_param_file(output_file, **params):
    with open(output_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")

# Define parameters for local execution
base_output_dir = "output_file" 
log_dir = "logs" # Debugging
param_dir = "params" # Debugging

os.makedirs(base_output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(param_dir, exist_ok=True)


# Parameters to iterate through
g11s = [2 * sqrt(2 / 5)]
g12s = [- sqrt(2 / 5)]
g22s = [3 * sqrt(2 / 5)]
ks = [1., 0.1, ] # kappa
alphas = [1.0] # Delta u
ls = [1.0] # lambda
n_taskss = [100]
trajectory_per_tasks = [100] # Careful, comp time is not linear with kappa

# Iterate through all combinations of parameters
for g11, g12, g22, k, alpha, l, n_tasks, trajectory_per_task in itertools.product(
    g11s, g12s, g22s, ks, alphas, ls, n_taskss, trajectory_per_tasks
):
    output_file = f"{base_output_dir}/{g11}_{g12}_{g22}_{k}_{l}_{alpha}_{n_tasks}_{trajectory_per_task}"
    param_file = f"{param_dir}/{g11}_{g12}_{g22}_{k}_{l}_{alpha}_{n_tasks}_{trajectory_per_task}.txt"
    log_file = f"{log_dir}/{g11}_{g12}_{g22}_{k}_{l}_{alpha}_{n_tasks}_{trajectory_per_task}.out"

    # Generate parameter file
    params = {
        "g11": g11,
        "g12": g12,
        "g22": g22,
        "k": k,
        "alpha": alpha,
        "l": l,
        "n_tasks": n_tasks,
        "trajectory_per_task": trajectory_per_task,
    }

    generate_param_file(param_file, **params)

    # Run script
    command = [
        "python",
        "run_simulations.py",
        "-o", output_file,
        "-g11", str(g11),
        "-g12", str(g12),
        "-g22", str(g22),
        "-k", str(k),
        "-alpha", str(alpha),
        "-l", str(l),
        "--n_tasks", str(n_tasks),
        "--trajectory_per_task", str(trajectory_per_task),
    ]

    # Execute and log
    with open(log_file, "w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            print(f"Task failed for parameters: {params}. See {log_file} for details.")
