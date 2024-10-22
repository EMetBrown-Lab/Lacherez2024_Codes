import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit




def process_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def fitting_msd_q1(time, a, b) : 
    return a * time**b 

def fitting_msd_q1_log(log_time, log_a, b):
    return log_a + b * log_time


def process_subfolders(root_folder):
    t = np.hstack([np.arange(10 ** i, 10 ** (i + 1), 10 ** i) for i in range(7)])

    # Initialize lists to store results for the CSV file
    results = []

    # Loop over all subfolders in the root folder
    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        # Check if the path is a folder
        if os.path.isdir(subfolder_path):
            print(f"Entering subfolder: {subfolder_path}")

            # Initialize the accumulators for each subfolder
            msd_q1 = np.zeros(63)
            msd_q2 = np.zeros(63)
            q1_histo = np.zeros(20)
            q1_bins = np.zeros(20)  # Assumed to stay constant
            q2_histo = np.zeros(50)
            q2_bins = np.zeros(50)  # Assumed to stay constant

            # Loop over all files in the subfolder
            for file_name in os.listdir(subfolder_path):
                # Check if the file is a pickle file
                if file_name.endswith('.pickle'):
                    file_path = os.path.join(subfolder_path, file_name)
                    data = process_pickle_file(file_path)

                    # Accumulate the values from the current pickle file
                    msd_q1 += 1/(100) * data["mean_msd_q1"][~np.isnan(data["mean_msd_q1"])]
                    msd_q2 += 1/(100) * data["mean_msd_q2"][~np.isnan(data["mean_msd_q2"])]
                    q1_histo += 1/(100) * data["mean_q1_histo"]
                    q1_bins = data["q1_bins_center"]  
                    q2_histo += 1/(100) * data["mean_q2_histo"]
                    q2_bins = data["q2_bins_center"] 

            parameters = subfolder_path.split('/')[-1].split('_') 
            N = int(parameters[0])
            tau = float(parameters[1])
            g_11 = float(parameters[2])
            g_12 = float(parameters[3])
            g_22 = float(parameters[4])
            kappa = float(parameters[5])
            l = float(parameters[6])
            alpha = float(parameters[7])
            N_trajectories = int(parameters[8])
            gamma_11 = g_11**2 + g_12**2
            gamma_12 = g_12 * (g_11 + g_22)
            gamma_22 = g_22**2 + g_12**2

            q1_th = np.linspace(0, l, 1000)

            # Calculate the relevant quantities for figurews 2 3 4
            def phi_prime_1(q) :
                return 2 * np.pi * alpha/l * np.cos(2 * np.pi * q/l)
            def phi_1(q) :    
                return alpha * np.sin(2 * np.pi * q/l) 
            

            def D_eff_Lifson_Jackson(gamma_11, l_1) :
                Defflj = l_1**2  / (np.trapz(np.exp(phi_1(q1_th)), q1_th) * np.trapz(np.exp(- phi_1(q1_th)), q1_th) * gamma_11)
                return Defflj
            

            def D_eff_Lifson_Jackson_Dean(l, kappa) : 
                I1 = np.trapz(np.exp(phi_1(q1_th)), q1_th)
                I2 = np.trapz(np.exp(phi_1(q1_th)) * phi_prime_1(q1_th)**2, q1_th)
                I3 = np.trapz(np.exp(- phi_1(q1_th)), q1_th)
                Deffljd = l**2/ gamma_11 * (I1 * I3)**(-1) * (1 + kappa * gamma_12**2/gamma_11**2 * I2 / I1)
                return Deffljd



            Deff_LJ = D_eff_Lifson_Jackson(gamma_11, l)
            Deff_LJ_Dean = D_eff_Lifson_Jackson_Dean(l, kappa)
            Delta_D_norm = (Deff_LJ_Dean-Deff_LJ)/Deff_LJ

            Dlist = [Deff_LJ, Deff_LJ_Dean, Delta_D_norm]
                    
            coefficients, errors = curve_fit(fitting_msd_q1_log, np.log(t[30:55]*tau), np.log(msd_q1[30:55]))
            errors = np.sqrt(np.diag(errors))
            coefficients[0] = np.exp(coefficients[0])

            simulation_parameters = {
                "N": N,
                "tau": tau,
                "g11": g_11,
                "g12": g_12,
                "g22": g_22,
                "kappa": kappa,
                "l":l,
                "alpha": alpha,
                "Ntrajectories":N_trajectories,
            }


        df_parameters = pd.DataFrame(parameters)
        df_coefficients = pd.DataFrame(coefficients) 
        df_errors = pd.DataFrame(errors)
        df_Dlist = pd.DataFrame(Dlist)
        df_msd_q1 = pd.DataFrame(msd_q1)
        df_msd_q2 = pd.DataFrame(msd_q2)
        df_q1_histo = pd.DataFrame(q1_histo)
        df_q1_bins = pd.DataFrame(q1_bins)
        df_q2_histo = pd.DataFrame(q2_histo)
        df_q2_bins = pd.DataFrame(q2_bins)
            

        df = pd.concat([df_parameters, df_coefficients,
                               df_errors, df_Dlist,
                               df_msd_q1, df_msd_q2,
                               df_q1_histo, df_q1_bins,
                               df_q2_histo, df_q2_bins],
                               axis = 1)

    # Save the DataFrame to a CSV file
        output_file =  "./results"
        output_csv = output_file + "/" + '-'.join('{}_{}'.format(key, str(value)) for key, value in simulation_parameters.items()) + ".csv"  

        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    root_folder = "./output_files/"  
    process_subfolders(root_folder)


