import numpy as np
import pandas as pd

from vahsimulator.parameters import max_delta_fin


# Importing the coefficients dataframe
data = pd.read_csv("./datasets/aerodecks/raw_data/14XW_V09_aedb_mach_alt_alpha_mrc_0.0_0.0_0.0.csv")

# Computing the static margin
data['X-C.P.'] =  data['CMA'] / data['CNA']

# Adding Angle-of-Sideslip
data['BETA'] = 0.

# Concatenating for diferent values of BETA
df1 = data.copy()
df2 = data.copy()

df1['BETA'] = -10.
df2['BETA'] = 10.

data = pd.concat([df1, data], axis=0).reset_index(drop=True)
data = pd.concat([data, df2], axis=0).reset_index(drop=True)

del df1, df2

# Defining Reference point ('XCG')
data.rename(columns={"X_MRC": "XCG"}, inplace=True)

# Adding Canard columns
data['DF1'] = 0.
data['DF2'] = 0.
data['DF3'] = 0.
data['DF4'] = 0.

data['PANL 1'] = 0.
data['PANL 2'] = 0.
data['PANL 3'] = 0.
data['PANL 4'] = 0.

# Computing L/D
data['CL/CD'] = data['CL'] / data['CD']

# Loading dataframes as reference
df_rato_no_canard = pd.read_csv("./datasets/aerodecks/core_bank_phase_01.csv")
df_rato_canard    = pd.read_csv("./datasets/aerodecks/control_delta_bank_phase_01.csv") 

COEFF_KEYS = df_rato_no_canard.columns.to_list()
DER_KEYS   = df_rato_canard.columns.to_list()

# Generating the data in the  
data_coeff = {}
for key in COEFF_KEYS:
    data_coeff[key] = data[key].values if key in data.columns.to_list() else 0.0

df = pd.DataFrame(data_coeff)

df.sort_values(by=['MACH', 'ALTITUDE', 'BETA', 'ALPHA'], inplace=True)

# Saving the core databank dataframe as a csv file
df.to_csv("./datasets/aerodecks/core_bank_phase_04.csv", index=False)

## Creating Controls Databank
# Filtering data to consider only lowest altitude
data_controls = data[data['ALTITUDE'] == data['ALTITUDE'].min()]

data_controls_idx = data_controls.set_index(['MACH', 'BETA', 'ALPHA'])

dcmd_values = np.array([-max_delta_fin, 0., max_delta_fin])
dcmd_rad_values = np.deg2rad(dcmd_values)

fins_values = ['fin_1', 'fin_2', 'fin_3', 'fin_4']

fin_map = {
    'fin_1': ('CADR','CNDR','CLLDR','CMDR','CLNDR'),
    'fin_3': ('CADL','CNDL','CLLDL','CMDL','CLNDL'),
}

rows_out = []

for mach, beta, alpha in data_controls_idx.index:
    row = data_controls_idx.loc[(mach, beta, alpha)]

    for fin in fins_values:
        if fin in fin_map:
            cols = fin_map[fin]
            dca, dcn, dcll, dcm, dcln = row[list(cols)]
        else:
            dca = dcn = dcll = dcm = dcln = 0.0

        for d_cmd, d_cmd_rad in zip(dcmd_values, dcmd_rad_values):
            rows_out.append({
                'ALPHA': alpha,
                'BETA': beta,
                'MACH': mach,
                'FIN': fin,
                'D_CMD': d_cmd,
                'DCA': dca * d_cmd_rad,
                'DCY': 0.0,
                'DCN': dcn * d_cmd_rad,
                'DCLL': dcll * d_cmd_rad,
                'DCM': dcm * d_cmd_rad,
                'DCLN': dcln * d_cmd_rad,
                'DPANL': 0.0
            })

dict_controls = {k: [row[k] for row in rows_out] for k in rows_out[0]}

                    
df_controls = pd.DataFrame(dict_controls)

df_controls.sort_values(by=['MACH', 'BETA', 'FIN', 'ALPHA', 'D_CMD'], inplace=True)

# Saving the control derivatives coefficients as csv file
df_controls.to_csv("./datasets/aerodecks/control_delta_bank_phase_04.csv", index=False)
