import numpy as np
import scipy
import scipy.signal as sp
from scipy.integrate import trapezoid
from tm_sinusoid_trap_pack import trajectory_sin_trap
import pickle
import os
import copy
import time
from concurrent.futures import ProcessPoolExecutor

class SinusoidalTrapSimulator:
    def __init__(self, mu_11, mu_12, mu_22, kappa, alpha, l, output_dir, N_trajectories, n_tasks, output_subfile):
        self.mu = np.array([[mu_11, mu_12], [mu_12, mu_22]])
        self.kappa = kappa
        self.alpha = alpha
        self.l = l
        self.output_dir = f"{output_dir}"
        self.N_trajectories = N_trajectories
        self.n_tasks = n_tasks
        self.output_subfile = output_subfile
       
        self.gamma = np.linalg.inv(self.mu)
        self.gamma_11, self.gamma_12, self.gamma_22 = self.gamma[0, 0], self.gamma[1, 0], self.gamma[1, 1]
        self.tau = self.simulation_time_step()
        self.D_LJ = self.D_eff_Lifson_Jackson()
        self.total_simulation_time = 1000 * self.tau_star()
        self.N = int(self.total_simulation_time / self.tau)
        self.time_lags = self.time_lag_array(n_points=11, lower_bound=int(100 * self.tau_star() / self.tau))
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dict = self.init_data_dict()

    def simulation_time_step(self): #play around with this based on the simulation you're running.
        return 1e-3 #0.001 * min((0.5 * (2 * np.pi / self.l)**2) * self.mu[0, 0], self.kappa * self.mu[1, 1])

    def tau_star(self):
        return self.l**2 / self.D_eff_Lifson_Jackson()

    def D_eff_Lifson_Jackson(self):
        q = np.linspace(0, self.l, 100000)
        phi = self.alpha * np.sin(2 * np.pi * q / self.l)
        num = trapezoid(np.exp(phi), q)
        den = trapezoid(np.exp(-phi), q)
        return self.l**2 / (num * den * self.gamma_11)

    def time_lag_array(self, n_points, lower_bound):
        return np.unique(np.logspace(np.log10(lower_bound), np.log10(self.N - 1), n_points).astype(int))

    def init_data_dict(self):
        data = {
            "N_trajectories": self.N_trajectories,
            "tau": self.tau,
            "N": self.N,
            "tau_star": self.tau_star(),
            "Time_lags": self.time_lags,
        }
        for tl in self.time_lags:
            data[f"num_bins_{tl}"] = 31
            data[f"histo_dq1_{tl}"] = np.zeros(31)
            data[f"bins_dq1_{tl}"] = np.zeros(31)
        return data

    def run_all_tasks(self, max_workers=5):
    
        output_subfile_path = os.path.join(self.output_dir, self.output_filename_base())
        os.makedirs(output_subfile_path, exist_ok=True)  
        args_list = [(n, copy.deepcopy(self.data_dict), *self.mu.flatten(), self.kappa, self.alpha, self.l, output_subfile_path) for n in range(self.n_tasks)]

        t1 = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.main_task, args_list))
        t2 = time.time()

        for result in results:
            print(result)

        print(f"Total simulation time: {t2 - t1:.2f} seconds")

    def output_filename_base(self):
        return '-'.join(
            f"{key}_{value}" for key, value in {
                "N": self.N,
                "tau": self.tau,
                "mu11": self.mu[0, 0],
                "mu12": self.mu[0, 1],
                "mu22": self.mu[1, 1],
                "kappa": self.kappa,
                "alpha": self.alpha,
                "l": self.l,
                "Ntrajectories": self.N_trajectories,
            }.items()
        )

    @staticmethod
    def displacements(q, time_lag):
        if time_lag >= len(q):
            raise ValueError("Gap size must be less than the length of the displacement array.")
        return q[int(time_lag):] - q[:-int(time_lag)]

    @staticmethod
    def bins_edges_and_centers(sigma):
        num_bins = 31
        data_range = (- 3 * sigma, 3 * sigma) # Maybe automate the span of data_range based on the expected diffusion augmentation for the final code.
        _, bin_edges = np.histogram([data_range], bins=num_bins, range=data_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_edges, bin_centers, num_bins

    @classmethod
    def fill_displacements_pdf(cls, q, tau, alpha, l, kappa, gamma_11, gamma_12, gamma_22, time_lag):
        dq = cls.displacements(q, time_lag)
        Deff_LJ = cls.compute_D_eff(gamma_11, alpha, l)
        sigma = np.sqrt(2 * Deff_LJ * tau * time_lag)
        bin_edges, bin_centers, num_bins = cls.bins_edges_and_centers(sigma)
        hist_values, _ = np.histogram(dq, bins=bin_edges, density=False)
        return bin_centers, hist_values, num_bins

    @staticmethod
    def compute_D_eff(gamma_11, alpha, l):
        q1_th = np.linspace(0, l, 100000)
        phi = alpha * np.sin(2 * np.pi * q1_th / l)
        return l**2 / (
            trapezoid(np.exp(phi), q1_th) * trapezoid(np.exp(-phi), q1_th) * gamma_11
        )

    @classmethod
    def main_task(cls, args):
        n, data_dict, mu_11, mu_12, mu_21, mu_22, kappa, alpha, l, output_subfile_path = args

        logs_dir = os.path.join("./", "logs")
        os.makedirs(logs_dir, exist_ok=True)
    
        # Redirect stdout/stderr to a log file
        log_file_path = os.path.join(logs_dir, f"log_task_{n}.txt")
        with open(log_file_path, "w") as log_file:
            import sys
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = log_file
            sys.stderr = log_file
    
            try:
                mu = np.array([[mu_11, mu_12], [mu_21, mu_22]])
                gamma = np.linalg.inv(mu)
                g = scipy.linalg.sqrtm(gamma)
                g_11, g_12, g_22 = g[0, 0], g[1, 0], g[1, 1]
                gamma_11, gamma_12, gamma_22 = gamma[0, 0], gamma[1, 0], gamma[1, 1]
    
                tau = data_dict["tau"]
                N = data_dict["N"]
                time_lags = data_dict["Time_lags"]
    
                for _ in range(data_dict["N_trajectories"]):
                    print(f"Simulating trajectory with N={N}, g=({g_11}, {g_12}, {g_22})")
                    q1, q2 = trajectory_sin_trap(N, g_11, g_12, g_22, tau, kappa, alpha, l)
                    print("1 trajectory done.")
                    for time_lag in time_lags:
                        bin_centers, hist_values, num_bins = cls.fill_displacements_pdf(
                            q1, tau, alpha, l, kappa, gamma_11, gamma_12, gamma_22, time_lag
                        )
                        data_dict[f"histo_dq1_{time_lag}"] += hist_values
                        data_dict[f"bins_dq1_{time_lag}"] = bin_centers
    
                filename = os.path.join(output_subfile_path, f"{n}.pickle")
                with open(filename, "wb") as handle:
                    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
                print(f"Task {n} completed.")
                return f"Task {n} completed"
    
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

