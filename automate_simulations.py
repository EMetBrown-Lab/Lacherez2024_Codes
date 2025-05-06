import os
import itertools
import subprocess
import yaml
from simulator import SinusoidalTrapSimulator

class SimulationBatchRunner:
    def __init__(self, config_file):
        """Initialize the runner with the configuration."""
        self.config_file = config_file
        self.config = self.load_config(config_file)

        self.base_output_dir = self.config['base_output_dir']
        self.log_dir = self.config['log_dir']
        self.param_dir = self.config['param_dir']

        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)

    def load_config(self, config_file):
        """Load the configuration from a YAML file."""
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def generate_param_file(self, output_file, **params):
        """Generate the parameter file for the simulation."""
        with open(output_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key}={value}\n")

    def get_output_subdir(self, **params):
        """Generate the directory path for output based on the parameters."""
        param_str = "_".join([f"{key}_{value}" for key, value in params.items()])
        return os.path.join(self.base_output_dir, param_str)

    def run_simulation(self, mu11, mu12, mu22, kappa, alpha, l, n_tasks, trajectory_per_task):
        """Run the simulation for a specific parameter set."""
        
        params = {
            "mu11": mu11,
            "mu12": mu12,
            "mu22": mu22,
            "kappa": kappa,
            "alpha": alpha,
            "l": l,
            "n_tasks": n_tasks,
            "trajectory_per_task": trajectory_per_task,
        }
        
        output_subdir = self.get_output_subdir(**params)

        param_file = os.path.join(self.param_dir, f"{mu11}_{mu12}_{mu22}_{kappa}_{l}_{alpha}_{n_tasks}_{trajectory_per_task}.txt")
        self.generate_param_file(param_file, **params)

        
        # Run the simulation
        simulator = SinusoidalTrapSimulator(mu11, mu12, mu22, kappa, alpha, l, self.base_output_dir, trajectory_per_task, n_tasks, output_subdir)
        simulator.run_all_tasks()

    def run_all_simulations(self):
        """Run all simulations based on the config parameters."""
        
        for mu11, mu12, mu22, kappa, alpha, l, n_tasks, trajectory_per_task in itertools.product(
            self.config['mu11'], self.config['mu12'], self.config['mu22'], self.config['ks'], 
            self.config['alphas'], self.config['ls'], self.config['n_taskss'], self.config['trajectory_per_tasks']
        ):
            self.run_simulation(mu11, mu12, mu22, kappa, alpha, l, n_tasks, trajectory_per_task)
            print(f"Simulation completed for: {mu11}, {mu12}, {mu22}, {kappa}, {alpha}, {l}, {n_tasks}, {trajectory_per_task}")

if __name__ == "__main__":
    # Run simulations based on the YAML configuration file. Cool if you have multiple config files to treat.
    config_file = "simulation_config.yaml"
    runner = SimulationBatchRunner(config_file)
    runner.run_all_simulations()

