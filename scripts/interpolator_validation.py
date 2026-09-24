import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vahsimulator import LinearInterpolator1D, LinearInterpolator2D, LinearInterpolator3D, LinearInterpolator4D

#--------------------------- 4D INTERPOLATOR VALIDATION ---------------------------
COEFF_KEYS = ['DCA','DCY','DCN','DCLL','DCM','DCLN','DPANL']

data_path = './datasets/aerodecks/control_delta_bank_phase_01.csv'
data = pd.read_csv(data_path)

mask = (
    (data["FIN"] == "fin_1") &
    data['MACH'].between(0.1, 0.85) &
    data['ALPHA'].between(-0.5, 1.5) &
    data['BETA'].between(-0.5, 11.)
    )

df = data[mask].reset_index(drop=True)

alpha_vals = np.sort(df['ALPHA'].unique())
mach_vals  = np.sort(df['MACH'].unique())
beta_vals  = np.sort(df['BETA'].unique())
dcmd_vals  = np.sort(df['D_CMD'].unique())

n_alpha = len(alpha_vals)
n_mach  = len(mach_vals)
n_beta  = len(beta_vals)
n_dcmd  = len(dcmd_vals)
n_coeff = len(COEFF_KEYS)

alpha_index = {v:i for i,v in enumerate(alpha_vals)}
mach_index  = {v:i for i,v in enumerate(mach_vals)}
beta_index  = {v:i for i,v in enumerate(beta_vals)}
d_cmd_index = {v:i for i,v in enumerate(dcmd_vals)}

values_grid = np.full((n_alpha, n_beta, n_mach, n_dcmd, n_coeff), np.nan)

for _, row in df.iterrows():

    i = alpha_index[row['ALPHA']]
    j = beta_index[row['BETA']]
    k = mach_index[row['MACH']]
    l = d_cmd_index[row['D_CMD']]

    for m, key in enumerate(COEFF_KEYS):

        val = row[key]

        if not np.isnan(val):
            values_grid[i, j, k, l, m] = val

interpolator = LinearInterpolator4D(alpha_vals, beta_vals, mach_vals, dcmd_vals, values_grid)

# Using Only Delta CM
DCM = values_grid[:, :, :, :, 4]

for i in range(n_alpha):
    alpha = alpha_vals[i]
    for j in range(n_alpha):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        beta = beta_vals[j]

        for k in range(n_mach-1):
            for l in range(n_dcmd-1):

                Y = np.array([[mach_vals[k],   mach_vals[k]], [mach_vals[k+1], mach_vals[k+1]]])
                X = np.array([[dcmd_vals[l], dcmd_vals[l+1]], [dcmd_vals[l],   dcmd_vals[l+1]]])
                Z = np.array([[DCM[i, j, k, l], DCM[i, j, k, l+1]], [DCM[i, j, k+1, l], DCM[i, j, k+1, l+1]]])

                # Create intermediary values to test interpolationMes
                intermediate_cmd = (dcmd_vals[l+1] +  dcmd_vals[l])/2
                intermediate_mach = (mach_vals[k+1] +  mach_vals[k])/2
                points = [[alpha, beta, intermediate_mach, intermediate_cmd]]
                points = np.ascontiguousarray(points, dtype=np.float64)
                
                result = interpolator.evaluate(points)

                ax.plot_wireframe(X, Y, Z, label = 'Dataset DCM')
                ax.plot(intermediate_cmd, result[0, 4] , c='r', marker='*', zs=intermediate_mach, zdir='y', label='Interpolated CA')

        # Remove repeated labels
        h, l = ax.get_legend_handles_labels()
        by_label = dict(zip(l, h))

        ax.legend(by_label.values(), by_label.keys())
        ax.set_ylabel('Mach [-]')
        ax.set_xlabel('D_CMD [°]')
        ax.set_zlabel('CM')
        ax.set_title(f'Fin 1 - Angle-of-Attack: {alpha} deg - Angle-of-Sideslip: {beta} deg')

        # Set initial visualization angle
        ax.view_init(elev=32., azim=-3, roll=-9)

#--------------------------- 3D INTERPOLATOR VALIDATION ---------------------------
COEFF_KEYS = ['CA','CY','CYB','CYP','CYR','CN','CNA','CNQ','CNAD',
              'CLLB','CLLP','CLLR','CM','CMA','CMQ','CMAD','CLNB','CLNR','CLNP'
             ]

data_path = './datasets/aerodecks/core_bank_phase_01.csv'

data = pd.read_csv(data_path)
data = data[data['BETA'] == 0.].reset_index(drop=True)

alpha_vals = np.sort(data['ALPHA'].unique())
mach_vals  = np.sort(data['MACH'].unique())
alt_vals   = np.sort(data['ALTITUDE'].unique())

n_alpha = len(alpha_vals)
n_mach  = len(mach_vals)
n_alt   = len(alt_vals)
n_coeff = len(COEFF_KEYS)

alpha_index = {v:i for i,v in enumerate(alpha_vals)}
mach_index  = {v:i for i,v in enumerate(mach_vals)}
alt_index   = {v:i for i,v in enumerate(alt_vals)}

values_grid = np.full((n_alpha, n_mach, n_alt, n_coeff), np.nan)

# Generate a regular grid based on alpha and mach values
for _, row in data.iterrows():

    i = alpha_index[row['ALPHA']]
    j = mach_index[row['MACH']]
    k = alt_index[row['ALTITUDE']]

    for l, key in enumerate(COEFF_KEYS):

        val = row[key]

        if not np.isnan(val):
            values_grid[i, j, k, l] = val

interpolator = LinearInterpolator3D(alpha_vals, mach_vals, alt_vals, values_grid)

# Use only CA coeficient to validate interpolation
CA = values_grid[:,:,:,0]

for k in range(n_alt):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    alt = alt_vals[k]

    for i in range(n_alpha-1):

        for j in range(n_mach-1):

            X = np.array([[alpha_vals[i], alpha_vals[i]], [alpha_vals[i+1], alpha_vals[i+1]]])
            Y = np.array([[mach_vals[j], mach_vals[j+1]], [mach_vals[j], mach_vals[j+1]]])
            Z = np.array([[CA[i, j, k], CA[i, j+1, k]], [CA[i+1, j, k], CA[i+1, j+1, k]]])

            # Create intermediary values to test interpolation
            intermediate_alpha = (alpha_vals[i+1] +  alpha_vals[i])/2
            intermediate_mach = (mach_vals[j+1] +  mach_vals[j])/2
            points = [[intermediate_alpha, intermediate_mach, alt]]
            points = np.ascontiguousarray(points, dtype=np.float64)
            
            result = interpolator.evaluate(points)

            ax.plot_wireframe(X, Y, Z, label = 'Dataset CA')
            ax.plot(intermediate_alpha, result[0, 0] , c='r', marker='*', zs=intermediate_mach, zdir='y', label='Interpolated CA')

    # Remove repeated labels
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))

    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Alpha [°]')
    ax.set_ylabel('Mach')
    ax.set_zlabel('CA')
    ax.set_title(f'Altitude: {alt} m')

    # Set initial visualization angle
    ax.view_init(elev=32., azim=-3, roll=-9)

#--------------------------- 2D INTERPOLATOR VALIDATION ---------------------------
values_grid = np.full((n_alpha, n_mach, n_coeff), np.nan)

data = data[data['ALTITUDE'] == 0.].reset_index(drop=True)

# Generate a regular grid based on alpha and mach values
for _, row in data.iterrows():

    i = alpha_index[row['ALPHA']]
    j = mach_index[row['MACH']]

    for k, key in enumerate(COEFF_KEYS):

        val = row[key]

        if not np.isnan(val):
            values_grid[i, j, k] = val


interpolator = LinearInterpolator2D(alpha_vals, mach_vals, values_grid)

# Use only CA coeficient to validate interpolation
CA = values_grid[:,:,0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(n_alpha-1):

    for j in range(n_mach-1):

        X = np.array([[alpha_vals[i], alpha_vals[i]], [alpha_vals[i+1], alpha_vals[i+1]]])
        Y = np.array([[mach_vals[j], mach_vals[j+1]], [mach_vals[j], mach_vals[j+1]]])
        Z = np.array([[CA[i, j], CA[i, j+1]], [CA[i+1, j], CA[i+1, j+1]]])

        # Create intermediary values to test interpolation
        intermediate_alpha = (alpha_vals[i+1] +  alpha_vals[i])/2
        intermediate_mach = (mach_vals[j+1] +  mach_vals[j])/2
        result = interpolator.evaluate([[intermediate_alpha, intermediate_mach]])

        ax.plot_wireframe(X, Y, Z, label = 'Dataset CA')
        ax.plot(intermediate_alpha, result[0, 0] , c='r', marker='*', zs=intermediate_mach, zdir='y', label='Interpolated CA')

# Remove repeated labels
h, l = ax.get_legend_handles_labels()
by_label = dict(zip(l, h))

ax.legend(by_label.values(), by_label.keys())
ax.set_xlabel('Alpha [°]')
ax.set_ylabel('Mach')
ax.set_zlabel('CA')

# Set initial visualization angle
ax.view_init(elev=32., azim=-3, roll=-9)

#--------------------------- 1D INTERPOLATOR VALIDATION ---------------------------

COEFF_KEYS = ['temperature_K','pressure_Pa','density_kg_m3', 'speed_of_sound_m_s' ]

data_path = './datasets/alcantara_fusion.csv'

data = pd.read_csv(data_path)

altitude_vals = np.sort(data['altitude_m'].unique())

n_altitude = len(altitude_vals)
n_coeff = len(COEFF_KEYS)

altitude_index = {v:i for i,v in enumerate(altitude_vals)}

values_grid = np.full((n_altitude, n_coeff), np.nan)

# Generate a regular grid based on altitude values
for _, row in data.iterrows():

    i = altitude_index[row['altitude_m']]

    for k, key in enumerate(COEFF_KEYS):

        val = row[key]

        if not np.isnan(val):
            values_grid[i, k] = val

# Use only temperature values to validate interpolation
temperature = values_grid[:,0]

# Create intermediary values to test interpolation
intermediate_altitude = altitude_vals[:-1] + np.diff(altitude_vals)/2

interpolator = LinearInterpolator1D(altitude_vals, values_grid)

result = interpolator.evaluate(intermediate_altitude)

interp_temp = result[:, 0]

fig, ax = plt.subplots()

ax.plot(altitude_vals/1000, temperature, linewidth=2, label='Dataset Temp.')
ax.plot(intermediate_altitude/1000, interp_temp, 'r--', label='Interpolated Temp.')
ax.set_title('Temperature')
ax.set_xlabel('Altitude [Km]')
ax.set_ylabel('Temperature [K]')
ax.legend()
ax.grid()

plt.show()
