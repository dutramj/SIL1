import numpy as np
import pandas as pd

input_file_name_prefix = "./datasets/aerodecks/raw_data/control_delta_bank_by_fin_phase_0"
output_file_name_prefix = "./datasets/aerodecks/control_delta_bank_phase_0"

phases = [1, 2, 3]

"""
The following convention is adopted to the canard settings:
 - A positive deflection is the one with a trailing edge moving downwards.

"""

deflec_correction_factor = { # Needed for matching deflection convetion adopted in simulator with the one used in DATCOM
    "fin_1": 1,
    "fin_2": 1,
    "fin_3": -1,
    "fin_4": -1,
}

hinge_moment_column = {        
    "fin_1": "DPANL_1",
    "fin_2": "DPANL_2",
    "fin_3": "DPANL_3",
    "fin_4": "DPANL_4",
}


for phase in phases:
    print(f"Phase is: {phase:1d}")

    # Loading Daframe
    file_name = input_file_name_prefix + f"{phase:1d}.csv"
    df = pd.read_csv(file_name)

    fins         = df['FIN'].unique().tolist()
    mach_values  = df['MACH'].unique().tolist()
    alpha_values = df['ALPHA'].unique().tolist()
    beta_values  = df['BETA'].unique().tolist()

    df_final = pd.DataFrame()

    for i, fin in enumerate(fins):
        df_fin = df[df['FIN'] == fin].reset_index(drop=True)

        # Correcting deflection
        df_fin['D_CMD'] = df_fin['D_CMD'] * deflec_correction_factor[fin]

        # Sorting results
        df_fin.sort_values(['ALPHA', 'BETA', 'MACH', 'FIN', 'D_CMD'], inplace=True)

        df_fin_red = df_fin[['ALPHA', 'BETA', 'MACH', 'FIN', 'D_CMD', 'DCA', 'DCY', 'DCN', 'DCLL', 'DCM', 'DCLN']]
        df_fin_red['DPANL'] = df_fin[hinge_moment_column[fin]]  

        df_final = pd.concat([df_final, df_fin_red], axis=0).reset_index(drop=True)
        del df_fin, df_fin_red

    # Adding Mach 0 Data
    print(f'Adding Mach 0 for Phase {phase}')
    mach_min = min(mach_values)
    df1 = df_final[df_final['MACH'] == mach_min].reset_index(drop=True)
    df1['MACH'] = 0.
    df_final = pd.concat([df1, df_final], axis=0).reset_index(drop=True)

    mach_values  = df_final['MACH'].unique().tolist()

    columns = df_final.columns.to_list()
    keys = ['MACH', 'BETA', 'FIN', 'ALPHA']

    data_fin0 = {}
    for column in columns:
        data_fin0[column] = []

    for mach in mach_values:
        for beta in beta_values:
            for fin in fins:
                for alpha in alpha_values:
                    data_fin0['MACH'].append(mach)
                    data_fin0['BETA'].append(beta)
                    data_fin0['FIN'].append(fin)
                    data_fin0['ALPHA'].append(alpha)

                    for key in columns:
                        if key not in keys:
                            data_fin0[key].append(0.)

    
    df_ = pd.DataFrame(data_fin0)
    df_final = pd.concat([df_final, df_], axis=0).reset_index(drop=True)

    keys.append('D_CMD')
    df_final.sort_values(by=keys, inplace=True)
    
    out_file_name = output_file_name_prefix + f"{phase}.csv"

    df_final.to_csv(out_file_name, index=False)

    del df, df_final
