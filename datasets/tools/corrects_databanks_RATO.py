import numpy as np
import pandas as pd

def corrects_cl(alpha, cl_base):
    k_correction = np.interp(alpha, [-10., -5., 0., 5., 10.], [2.3, 2.8, 3., 2.8, 2.3], left=2.3, right=2.3)
    
    return cl_base * k_correction

def corrects_cd(alpha, cd_base):
    k_correction = np.interp(alpha, [-10., 0., 10.], [1.5, 1.1, 1.5], left=1.5, right=1.5)

    return cd_base * k_correction

def computes_cn(alpha, cl, cd):
    cosa = np.cos(np.deg2rad(alpha))
    sina = np.sin(np.deg2rad(alpha))

    cn = cl * cosa + cd * sina

    return cn

def computes_ca(alpha, cl, cd):
    cosa = np.cos(np.deg2rad(alpha))
    sina = np.sin(np.deg2rad(alpha))

    cd = -cl * sina + cd * cosa

    return cd

if __name__ == "__main__":
    input_file_name_prefix  = "./datasets/aerodecks/raw_data/core_bank_phase_0"
    output_file_name_prefix = "./datasets/aerodecks/core_bank_phase_0"

    phases = [1, 2, 3]

    for phase in phases:
        print(f"Phase is: {phase}")
        
        file_name = input_file_name_prefix + f"{phase:1d}.csv"
        
        df = pd.read_csv(file_name)

        # Correcting Beta Coefficients
        keys = ['CYB', 'CLNB', 'CLLB', 'CYR', 'CLNR', 'CLLR', 'CYP', 'CLNP', 'CLLP']
        
        mach_values  = df['MACH'].unique().tolist()
        beta_values  = df['BETA'].unique().tolist()
        alt_values   = df['ALTITUDE'].unique().tolist()

        df_final = pd.DataFrame()
        
        for alt in alt_values:
            df_alt = df[df['ALTITUDE'] == alt].reset_index(drop=True)
            for mach in mach_values:
                df_mach = df_alt[df_alt['MACH'] == mach].reset_index(drop=True)

                df_beta0 = df_mach[df_mach['BETA'] == 0.].reset_index(drop=True)
                for beta in beta_values:
                    df_1 = df_mach[df_mach['BETA'] == beta].reset_index(drop=True)
                    
                    for key in keys:
                        df_1[key] = df_beta0[key]

                    df_final = pd.concat([df_final, df_1], axis=0).reset_index(drop=True)

        
        # Adding Mach 0 Data
        print(f'Adding Mach 0 for Phase {phase}')
        mach_min = min(mach_values)
        df1 = df_final[df_final['MACH'] == mach_min].reset_index(drop=True)
        df1['MACH'] = 0.
        
        df_ = pd.concat([df1, df_final], axis=0).reset_index(drop=True)

        df_.sort_values(by=['MACH', 'ALTITUDE', 'BETA', 'ALPHA'], inplace=True)

        # Adding Factor on Drag
        if phase == 3:
            df_['CL'] = df_.apply(lambda row: corrects_cl(row['ALPHA'], row['CL']), axis=1)
            df_['CD'] = df_.apply(lambda row: corrects_cd(row['ALPHA'], row['CD']), axis=1)
            df_['CN'] = df_.apply(lambda row: computes_cn(row['ALPHA'], row['CL'], row['CD']), axis=1)
            df_['CA'] = df_.apply(lambda row: computes_ca(row['ALPHA'], row['CL'], row['CD']), axis=1)
            df_['CL/CD'] = df_['CL'] / df_['CD']

            df_['PANL 2'] = 0.0
            df_['PANL 3'] = 0.0
            df_['PANL 4'] = 0.0

        out_file_name = output_file_name_prefix + f"{phase:1d}.csv"

        df_.to_csv(out_file_name, index=False)
