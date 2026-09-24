"""
Generate trajectory table with Time, Mach, Static Pressure, Static Temperature, and Angle of Attack.
"""

import pandas as pd
import os


def generate_trajectory_table(input_file, output_file='trajectory_table.csv'):
    """
    Generate trajectory table from simulation data.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Output CSV filename (default: 'trajectory_table.csv')
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    columns_to_extract = [
        'TIME__s',
        'MACH_NUMBER',
        'PRESSURE__Pa',
        'TEMPERATURE__K',
        'ANGLE_OF_ATTACK__deg'
    ]
    
    missing_columns = [col for col in columns_to_extract if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in file: {missing_columns}")
    
    trajectory_df = df[columns_to_extract].copy()
    
    output_path = os.path.join('records', output_file)
    trajectory_df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\nTrajectory table generated successfully!")
    print(f"File saved at: {output_path}")
    print(f"\nTable statistics:")
    print(f"  - Total points: {len(trajectory_df)}")
    print(f"  - Initial time: {trajectory_df['TIME__s'].iloc[0]:.2f} s")
    print(f"  - Final time: {trajectory_df['TIME__s'].iloc[-1]:.2f} s")
    print(f"  - Max Mach: {trajectory_df['MACH_NUMBER'].max():.3f}")
    print(f"  - Min pressure: {trajectory_df['PRESSURE__Pa'].min():.2f} Pa")
    print(f"  - Min temperature: {trajectory_df['TEMPERATURE__K'].min():.2f} K")
    print(f"  - Max angle of attack: {trajectory_df['ANGLE_OF_ATTACK__deg'].abs().max():.3f}°")
    
    print(f"\nFirst 5 rows:")
    print(trajectory_df.head().to_string(index=False))
    
    return trajectory_df


if __name__ == "__main__":
    input_file = 'records/simulation_data_14x_aut.csv'
    output_file = 'trajectory_table_cimatec.csv'
    
    try:
        trajectory_table = generate_trajectory_table(input_file, output_file)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Check if the simulation file exists in the 'records/' directory")
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
