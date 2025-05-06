import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.special import i0, i1
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


class DeanAnalysis:
    def __init__(self, folder_path, verbose=False):
        self.folder_path = folder_path
        self.verbose = verbose
        self.params = self.extract_parameters()
        self.gamma = np.linalg.inv(np.array([
            [self.params['mu11'], self.params['mu12']],
            [self.params['mu12'], self.params['mu22']]
        ]))
        self.gamma_11, self.gamma_12, self.gamma_22 = self.gamma[0, 0], self.gamma[1, 0], self.gamma[1, 1]
        self.data = self.load_pickle_files()
        self.time_array = np.array(self.data['Time_lags'])[0, :]
        self.tau = float(self.data['tau'][0])
        self.N = int(self.data['N'][0])
        self.epsilon = self.params['epsilon'] if self.params['epsilon'] else None
        self.mu_11 = self.params['mu11']
        self.mu_12 = self.params['mu12']
        self.mu_22 = self.params['mu22']
        self.alpha = self.params['alpha']
        self.l = self.params['l']

    def extract_parameters(self): #correct
        parts = self.folder_path.split('/')[-1].split('-')
        extracted_params = {}

        for part in parts:
            try:
                key, value = part.split('_', 1)

                #Convert to appropriate type
                if key in ['mu11', 'mu12', 'mu22', 'tau', 'kappa', 'alpha', 'l']:
                    extracted_params[key] = float(value)
                elif key in ['N', 'Ntrajectories']:
                    extracted_params[key] = int(value)
            except ValueError:
                continue

        # Calculate epsilon
        if 'kappa' in extracted_params and 'l' in extracted_params:
            extracted_params['epsilon'] = extracted_params['kappa'] / extracted_params['l']**2 * extracted_params['alpha']


        return extracted_params



    def load_pickle_files(self):
        data = {}
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith('.pickle'):
                with open(os.path.join(self.folder_path, filename), 'rb') as f:
                    new_data = pickle.load(f)
                    for k, v in new_data.items():
                        data.setdefault(k, []).append(v)
        return data

    def phi_1(self, q):
        return self.params['alpha'] * np.sin(2 * np.pi * q / self.params['l'])

    def phi_prime_1(self, q):
        return (2 * np.pi * self.params['alpha'] / self.params['l']) * np.cos(2 * np.pi * q / self.params['l'])

    def D_eff_LJ(self):
        q = np.linspace(0, self.params['l'], 1000000)
        return self.params['l']**2 / (
            trapezoid(np.exp(self.phi_1(q)), q) *
            trapezoid(np.exp(-self.phi_1(q)), q) *
            self.gamma_11
        )

    def calc_alpha_coeff(self):
        try:
            epsilon = self.params['epsilon']
            return 4 * np.pi ** 2 * epsilon * (self.gamma_12/self.gamma_11)**2 * self.alpha * i1(self.alpha) / i0(self.alpha)
        except KeyError:
            print('No epsilon found in params. Check kappa and lambda exctraction.')

    def D_deff_LJ_large_epsilon(self) :
        q = np.linspace(0, self.params['l'], 1000000)
        Zplus = trapezoid(np.exp(self.phi_1(q)), q)
        Zminus = trapezoid(np.exp(- self.phi_1(q)), q)

        return self.mu_11 * (self.mu_11 * self.mu_22 - self.mu_12 ** 2) / (self.l ** (-2) * Zplus * Zminus * (self.mu_11 * self.mu_22 - self.mu_12 ** 2))

    def MSD(self, bin_centers, histo):
        bin_width = np.mean(np.diff(bin_centers))
        histo = histo / (np.sum(histo) * bin_width)
        return trapezoid(bin_centers**2 * histo, bin_centers)

    def process(self):
        msd_list = []
        hist_keys = sorted([k for k in self.data if k.startswith("histo")], key=lambda x: float(x.split("_")[-1]))
        bin_keys = sorted([k for k in self.data if k.startswith("bins")], key=lambda x: float(x.split("_")[-1]))

        all_bins = []
        all_hists = []

        for n in range(len(self.time_array)):
            histograms = self.data[hist_keys[n]]
            bins = self.data[bin_keys[n]][0]
            cumulative_hist = sum(histograms)
            msd_q1 = self.MSD(bins, cumulative_hist)
            msd_list.append(msd_q1)

            if self.verbose:
                all_bins.append(bins)
                all_hists.append(cumulative_hist)

        if self.verbose:
            n_plots = len(all_hists)
            n_cols = 4
            n_rows = int(np.ceil(n_plots / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)

            for i, ax in enumerate(axes.flat):
                if i < n_plots:
                    ax.bar(all_bins[i], all_hists[i], width=np.diff(all_bins[i]).mean(), align='center', alpha=0.7)
                    ax.set_title(f"Time lag: {self.time_array[i]:.2e}")
                    ax.set_xlabel("q1")
                    ax.set_ylabel("Frequency")
                    ax.grid(True)
                else:
                    ax.axis('off')  # Hide unused subplots

            plt.suptitle("Histograms at Different Time Lags", fontsize=16)
            plt.show()

        msd_array = np.array(msd_list)
        time_scaled = self.time_array * self.tau
        D_LJ = self.D_eff_LJ()
        y_data = (msd_array / (2 * time_scaled * D_LJ)) - 1

        popt, pcov = curve_fit(lambda t, d: d, time_scaled[5:], y_data[5:], p0=[0.1])
        delta_D = popt[0]
        delta_D_error = np.sqrt(np.diag(pcov))[0]

        D_LJ_Dean_small = D_LJ * (1 + self.calc_alpha_coeff())
        D_LJ_Dean_big = self.D_deff_LJ_large_epsilon()
        delta_D_small_theory = (D_LJ_Dean_small - D_LJ) / D_LJ #small epsilon
        delta_D_big_theory = (D_LJ_Dean_big - D_LJ) / D_LJ #large epsilon

        self.save_results(D_LJ, D_LJ_Dean_small, delta_D_small_theory, D_LJ_Dean_big, delta_D_big_theory, delta_D, delta_D_error)

        import matplotlib.patches as patches
        
        plt.errorbar(time_scaled, y_data, fmt='o', label='Simulation data')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.ylim(1e-3, 1e-1)
        
        # Add horizontal band for delta_D ± delta_D_error
        plt.axhline(delta_D, color='r', linestyle='dashed', label='Measured ΔD/D')
        plt.fill_between(time_scaled, delta_D - delta_D_error, delta_D + delta_D_error,
                         color='r', alpha=0.3, label='Error range')
        
        plt.title(f"kappa = {self.params['kappa']}")
        plt.xlabel("Time")
        plt.ylabel("ΔD / D")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()




    def save_results(self, D_LJ, D_LJ_small_Dean, delta_D_small_theory, D_LJ_big_Dean, delta_D_big_theory, delta_D, delta_D_error):
        import csv
        os.makedirs("./results", exist_ok=True)
    
        filename = (
            f"mu11_{self.mu_11}_mu12_{self.mu_12}_mu22_{self.mu_22}_"
            f"tau_{self.tau}_kappa_{self.params.get('kappa', 'NA')}_"
            f"alpha_{self.alpha}_l_{self.l}_N_{self.N}.csv"
        )
        output_path = os.path.join("./results", filename)
        fieldnames = ["D_LJ", "D_LJ_small_Dean", "delta_D_small_theory","D_LJ_big_Dean", "delta_D_big_theory", "delta_D_measured", "delta_D_error"]
    
        with open(output_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "D_LJ": D_LJ,
                "D_LJ_small_Dean": D_LJ_small_Dean,
                "delta_D_small_theory": delta_D_small_theory,
                "D_LJ_big_Dean": D_LJ_big_Dean,
                "delta_D_big_theory": delta_D_big_theory,
                "delta_D_measured": delta_D,
                "delta_D_error": delta_D_error
            })
    
        if self.verbose:
            print(f"[SAVED] CSV results written to: {output_path}")

def process_folder(folder_path):
    try:
        analysis = DeanAnalysis(folder_path, verbose=True )
        analysis.process()
        return f"[DONE] Processed: {folder_path}"
    except Exception as e:
        return f"[ERROR] {folder_path} failed with error: {e}"

if __name__ == "__main__":
    base_path = "./output_file/"
    folder_paths = [
        os.path.join(base_path, folder)
        for folder in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, folder))
    ]

    print(f"[INFO] Found {len(folder_paths)} folders to process.")

    # Limit workers to avoid overloading CPU/memory
    max_workers = min(8, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_folder, path): path for path in folder_paths}
        for future in as_completed(futures):
            print(future.result())

