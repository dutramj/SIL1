import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defining the file path
thrust_path = './datasets/Dataset_NominalThrustS50_IAE_old.csv'

# Importing data from csv
thrust_data = pd.read_csv(thrust_path, sep=";", header=None).to_numpy()

thrust_data[:, 1] = thrust_data[:, 1] * 1000. 

# Defining time constraints
t0 = np.floor(np.min(thrust_data[:, 0])) # Initial time value for regularly-spaced time vector, in s 
tfinal = np.max(thrust_data[:, 0]) # Final time value, in s
t1 = np.floor(tfinal) # Final time value, in s
n_points = int(t1 - t0) * 200 + 1 # Number of points

# Creating time vector
time = np.linspace(t0, t1, n_points)
time = np.concatenate((time, np.array([tfinal])), axis=0)

# Interpolating data
data = dict()
data['Time_s'] = time
data['Thrust_N'] = np.interp(time, thrust_data[:, 0], thrust_data[:, 1], left=np.min(thrust_data[:, 1]), right=np.max(thrust_data[:, 1]))


# Exporting to csv
output_filename = './datasets/Dataset_NominalThrust_S50_MJ.csv'
df = pd.DataFrame(data)
df.to_csv(output_filename, index=False)

# Validating the results
res = np.interp(thrust_data[:, 0], time, data['Thrust_N'])
error = res - thrust_data[:, 1]

plt.figure()
plt.hist(error, bins=40)
plt.title('Error - Thrust_N')

plt.figure()
plt.plot(thrust_data[:, 0], thrust_data[:, 1], linewidth=2., color='b')
plt.scatter(time, data['Thrust_N'], color='r')
plt.xlabel('Time [s]')
plt.ylabel('Thrust [N]')

plt.show()