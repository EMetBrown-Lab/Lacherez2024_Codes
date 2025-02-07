"""File to process the pickled histograms for each time step, to extract the Dean Lifson Jackson coefficient
for each parameter set.
Returns a CSV for each parameter set containing the parameters, the time steps into consideration, the Dean coefficient, the LJ coefficient, the experimentally calculated
diffusion coefficient (mean of the histos) and their variance, """

import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.special import i0, i1
import matplotlib.pyplot as plt
from matplotlib import cm

############################################ Pickling and loading files ############################################
def load_pickle_files(sub_directory):
    dictionary_data = {}
    for filename in sorted(os.listdir(sub_directory)):
        if filename.endswith('.pickle'):
            with open(os.path.join(sub_directory, filename), 'rb') as f:
                data = pickle.load(f)
                print(data.items())
                for key, value in data.items():
                    if key not in dictionary_data:
                        dictionary_data[key] = []
                    dictionary_data[key].append(value)
    return dictionary_data

def extract_parameters_from_folder(folder_name):
    parts = folder_name.split('/')[-1].split('_')
    try:
        g11 = float(parts[0])
        g12 = float(parts[1])
        g22 = float(parts[2])
        kappa = float(parts[3])
        alpha = float(parts[5])
        l = float(parts[4])
        n_tasks = int(parts[6])
        trajectories_per_task = int(parts[7])
        return g11, g12, g22, kappa, alpha, l, n_tasks, trajectories_per_task
    except (IndexError, ValueError) as e:
        print(f"Error extracting parameters from folder name: {folder_name}")
        raise e


############################################ Stats and fiting ############################################

def fit_DeltaD(time, DeltaD) :
    return DeltaD

# def log_fit_fullMSD_q1(timelags, a, b):
#     return np.log(a*timelags**b)

def MSD(bin_centers, histo):
    histo /= trapezoid(histo, bin_centers)
    return trapezoid(bin_centers**2 * histo, bin_centers)


def time_lag_array(length_trajectory, n_points, lower_bound):
    """Returns the time lags used for convergence tests."""
    time_array = np.unique(np.logspace(
        np.log(lower_bound) / np.log(10),
        np.log(length_trajectory) / np.log(10),
        n_points,
        base=10,
    ).astype(int))
    return time_array[:-1]



############################################ Simulation functions ############################################

def phi_prime_1(q, alpha, l):
    return (2 * np.pi * alpha / l) * np.cos(2 * np.pi * q / l)

def phi_1(q, alpha, l):
    return alpha * np.sin(2 * np.pi * q / l)

def D_eff_Lifson_Jackson(gamma_11, alpha, l):
    q1_th = np.linspace(0, l, 100000)
    Deff = l**2 / (trapezoid(np.exp(phi_1(q1_th, alpha, l)), q1_th) * trapezoid(np.exp(-phi_1(q1_th, alpha, l)), q1_th) * gamma_11)
    return Deff

def calc_alpha_coefficient(alpha, l, gamma_11, gamma_12):
    q1_th = np.linspace(0, l, 100000)
    I1 = trapezoid(np.exp(phi_1(q1_th, alpha, l)), q1_th)
    I2 = trapezoid(np.exp(phi_1(q1_th, alpha, l)) * phi_prime_1(q1_th, alpha, l)**2, q1_th)
    return (gamma_12 / (l * gamma_11))**2 * I2 / I1


def mu_matrix_fromgamma (gamma_11, gamma_12, gamma_22) :
    mu_11 =   (gamma_11 * gamma_22 - gamma_12**2)**(-1) * gamma_22
    mu_12 = - (gamma_11 * gamma_22 - gamma_12**2)**(-1) * gamma_12
    mu_22 =   (gamma_11 * gamma_22 - gamma_12**2)**(-1) * gamma_11
    return mu_11, mu_12, mu_22



def tau_star(l, D_LJ) :
    _tau_star = l**2 / D_LJ
    return _tau_star



def gamma_matrix(g_11, g_12, g_22):
    gamma_11 = g_11**2 + g_12**2
    gamma_12 = g_12 * (g_11 + g_22)
    gamma_22 = g_22**2 + g_12**2
    return gamma_11, gamma_12, gamma_22


############################################ Time functions ############################################
def total_simu_time(l, D_LJ) :
    total_simulation_time = 100 * tau_star(l, D_LJ)
    return total_simulation_time


def num_steps(total_simu_time, time_step) :
    return int(total_simu_time/time_step)


def simulation_time_step(mu_11, mu_22, l, kappa_2) :
    """Dynamically choose the time step to avoid nconsistent precision between simulations."""
    kappa_1 = (0.5 * (2 * np.pi / l)**2)
    t1 = kappa_1 * mu_11
    t2 = kappa_2 * mu_22
    time_step = 0.01 * min(t1, t2)
    return time_step


############################################ Processing the data ############################################

def process_subfolders(folder_name): 
    try:
        g11, g12, g22, kappa, alpha, l, n_tasks, trajectories_per_task = extract_parameters_from_folder(folder_name)
        gamma_11, gamma_12, gamma_22 = gamma_matrix(g11, g12, g22)
        mu_11, _, mu_22 = mu_matrix_fromgamma (gamma_11, gamma_12, gamma_22)
        tau = simulation_time_step(mu_11, mu_22, l, kappa)
    except Exception as e:
        print(f"Skipping folder {folder_name} due to parameter extraction error.")
        raise e


    Deff_LJ = D_eff_Lifson_Jackson(gamma_11, alpha, l)
    total_simulation_time = total_simu_time(l, Deff_LJ)
    N = num_steps(total_simulation_time, tau)
    
    time_array = time_lag_array(N, 20, int(tau_star(l, Deff_LJ)/tau))

    msd_list=[]
        

    data = load_pickle_files(folder_name)
    histo_keys = sorted([i for i in list(data.keys()) if i.split("_")[0] == "histo"], key=lambda i: float(i.split("_")[-1]))
    bins_keys = sorted([i for i in list(data.keys()) if i.split("_")[0] == "bins"], key=lambda i: float(i.split("_")[-1]))

    for n, time_lag in enumerate(time_array):
        histo_key = histo_keys[n]
        bins_key = bins_keys[n]
        histograms = data.get(histo_key, [])
        bins = data.get(bins_key)[0]

        cumulative_hist = np.zeros_like(histograms[0])


        for idx, histo in enumerate(histograms):
            cumulative_hist += histo
            msd_q1 = MSD(bins, cumulative_hist)

        msd_list.append(msd_q1)

    p0 = [0.2]
    coefficients, covariance = curve_fit(
        fit_DeltaD, 
        time_array*tau,
        (msd_list[:]/(2 * time_array[:] * tau * D_eff_Lifson_Jackson(gamma_11, alpha, l)) - 1),
        p0 = p0
    )


    # coefficients, covariance = curve_fit(
    #     log_fit_fullMSD_q1, 
    #     time_array*tau,
    #     np.log(msd_list),
    #     p0 = p0
    # )

    DeltaD_plotting = ((msd_list/ (2 * time_array * tau * D_eff_Lifson_Jackson(gamma_11, alpha, l))) - 1 )
    plt.xlabel("$\\tau$")  # Update as necessary
    plt.ylabel("$\Delta D / D$")  # Update as necessary
    plt.ylim(1e-3, 1)
    plt.title(kappa)
    # plt.xlim(30, 250)
    # print(msd_list)
    # plt.grid()
    # plt.loglog(time_array[:]*tau, msd_list, 'o') # Only on relevant range and removing long times with poor stats
    plt.loglog(time_array*tau, DeltaD_plotting, 'o')
    # plt.loglog(time_array*tau, np.exp(log_fit_fullMSD_q1(time_array * tau, coefficients[0], coefficients[1])))
    print(coefficients[0])
    plt.hlines(coefficients[0], xmin = 1, xmax = 250)
    plt.show()



    Deff_LJ_theo = D_eff_Lifson_Jackson(gamma_11, alpha, l)
    Deff_LJ_Dean_theo = Deff_LJ_theo * (1 + kappa*calc_alpha_coefficient(alpha, l, gamma_11, gamma_12))
    Delta_D_norm_theo = (Deff_LJ_Dean_theo-Deff_LJ_theo)/Deff_LJ_theo
    Dlist_theo = [Deff_LJ_theo, Deff_LJ_Dean_theo, Delta_D_norm_theo]
    error_mat = np.sqrt(np.diag(covariance))
    # Delta_D_norm_exp = ((np.exp(coefficients[0])*tau/2)/Deff_LJ_theo) - 1
    Delta_D_norm_exp = coefficients[0]
    Dlist_experiments = [0, error_mat[0], Delta_D_norm_exp]

    epsilon = kappa / l**2

    simulation_parameters = {
            "N": N,
            "tau": tau,
            "g11": g11,
            "g12": g12,
            "g22": g22,
            "kappa": kappa,
            "alpha":alpha,
            "l": l,
            "Ntrajectories":trajectories_per_task,
            "Ntasks":n_tasks,
            "epsilon": epsilon,
    }

    df_parameters = pd.DataFrame(list(simulation_parameters.items()), columns=["Parameter", "Value"])
    df_Dlist_theo = pd.DataFrame(Dlist_theo)
    df_Dlist_experiments = pd.DataFrame(Dlist_experiments)

    df = pd.concat([
        df_parameters,
        df_Dlist_theo,
        df_Dlist_experiments
    ], axis=1)


    # Save the DataFrame to a CSV file
    output_file = f"results" 
    os.makedirs(output_file, exist_ok=True)
    output_csv = output_file + "/" + '-'.join('{}_{}'.format(key, str(value)) for key, value in simulation_parameters.items()) + ".csv"
    
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


############################################ MAIN ############################################

if __name__ == "__main__":
    base_directory = './output_file_finalsimus/'
    for set_folder in sorted(os.listdir(base_directory)):
        set_folder_path = os.path.join(base_directory, set_folder)
        if os.path.isdir(set_folder_path):
            print(f"Analyzing : {set_folder_path}")
            process_subfolders(set_folder_path)




