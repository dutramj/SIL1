import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Ixx_prop_path = './datasets/Dataset_NominalPropellantIxxS50_IAE.csv'
Iyy_prop_path = './datasets/Dataset_NominalPropellantIyyS50_IAE.csv'
Izz_prop_path = './datasets/Dataset_NominalPropellantIyyS50_IAE.csv'
mass_prop_path = './datasets/Dataset_NominalPropellantMassS50_IAE.csv'
x_cg_prop_path = './datasets/Dataset_NominalPropellantXcgS50_IAE.csv'

# Importing data from csv
Ixx_data = pd.read_csv(Ixx_prop_path, sep=";", header=None).to_numpy()
Iyy_data = pd.read_csv(Iyy_prop_path, sep=";", header=None).to_numpy()
Izz_data = pd.read_csv(Izz_prop_path, sep=";", header=None).to_numpy()
mass_data = pd.read_csv(mass_prop_path, sep=";", header=None).to_numpy()
xcg_data = pd.read_csv(x_cg_prop_path, sep=";", header=None).to_numpy()
xcg_data[:, 1] = xcg_data[:, 1] / 1000. 

# Defining time constraints
freq = 200 # Simulation frequency, in Hz
t0 = np.floor(max(np.min(Ixx_data[:, 0]), np.min(Iyy_data[:, 0]), np.min(Izz_data[:, 0]), np.min(mass_data[:, 0]), np.min(xcg_data[:, 0]))) # Initial time value for regularly-spaced time vector, in s 
tfinal = min(np.max(Ixx_data[:, 0]), np.max(Iyy_data[:, 0]), np.max(Izz_data[:, 0]), np.max(mass_data[:, 0]), np.max(xcg_data[:, 0])) # Final time value, in s
t1 = np.floor(tfinal) # Final time value, in s
n_points = int(t1 - t0) * freq + 1 # Number of points

# Creating time vector
time = np.linspace(t0, t1, n_points)
time = np.concatenate((time, np.array([tfinal])), axis=0)

# Interpolating data
columns = ['Mass_kg', 'Xcg_m', 'Ixx_kgm2', 'Iyy_kgm2', 'Izz_kgm2']
raw_data = [mass_data, xcg_data, Ixx_data, Iyy_data, Izz_data]

data = dict()
data['Time_s'] = time

for i, c in enumerate(columns):
    data[c] = np.interp(time, raw_data[i][:, 0], raw_data[i][:, 1], left=np.min(raw_data[i][:, 1]), right=np.max(raw_data[i][:, 1]))


# Exporting to csv
output_filename = './datasets/Dataset_NominalPropellant_S50_MJ.csv'
df = pd.DataFrame(data)
df.to_csv(output_filename, index=False)

# Validating the results
for i, c in enumerate(columns):
    res = np.interp(raw_data[i][:, 0], time, data[c])
    error = res - raw_data[i][:, 1]

    plt.figure()
    plt.hist(error, bins=40)
    plt.title('Error ' + c)

plt.show()



