import os
import glob
import pickle
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, classification_report, silhouette_score)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Professional style for presentations
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linewidth'] = 1.0

# Professional color palette
COLORS = {
    'success': '#2ecc71',  # Vibrant green
    'failed': '#e74c3c',   # Vibrant red
    'primary': '#3498db',  # Bright blue
    'secondary': '#9b59b6', # Purple
    'warning': '#f39c12',  # Orange
    'info': '#1abc9c',     # Teal
    'dark': '#34495e',     # Dark gray
}

# Mission success criteria for payload delivery
MACH_THRESHOLD = 7.0
ALTITUDE_MIN = 30000.0  # meters
ALTITUDE_MAX = 40000.0  # meters
FLIGHT_PATH_ANGLE_MAX = 5.0  # degrees


def check_delivery_success(df):
    """Check if simulation achieved successful payload delivery conditions"""
    mach_ok = df['MACH_NUMBER'] >= MACH_THRESHOLD
    altitude_ok = (df['ALTITUDE__m'] >= ALTITUDE_MIN) & (df['ALTITUDE__m'] <= ALTITUDE_MAX)
    path_angle_ok = np.abs(df['FLIGHT_PATH_ANGLE__deg']) <= FLIGHT_PATH_ANGLE_MAX
    
    delivery_success = mach_ok & altitude_ok & path_angle_ok
    
    return delivery_success.any()


def get_delivery_metrics(df):
    """Extract key metrics when delivery conditions are met"""
    mach_ok = df['MACH_NUMBER'] >= MACH_THRESHOLD
    altitude_ok = (df['ALTITUDE__m'] >= ALTITUDE_MIN) & (df['ALTITUDE__m'] <= ALTITUDE_MAX)
    path_angle_ok = np.abs(df['FLIGHT_PATH_ANGLE__deg']) <= FLIGHT_PATH_ANGLE_MAX
    
    delivery_window = mach_ok & altitude_ok & path_angle_ok
    
    if not delivery_window.any():
        return None
    
    delivery_idx = delivery_window.idxmax()
    
    return {
        'time': df.loc[delivery_idx, 'TIME__s'],
        'altitude': df.loc[delivery_idx, 'ALTITUDE__m'],
        'mach': df.loc[delivery_idx, 'MACH_NUMBER'],
        'flight_path_angle': df.loc[delivery_idx, 'FLIGHT_PATH_ANGLE__deg'],
        'angle_of_attack': df.loc[delivery_idx, 'ANGLE_OF_ATTACK__deg'],
        'sideslip': df.loc[delivery_idx, 'SIDESLIP_ANGLE__deg'],
        'dynamic_pressure': df.loc[delivery_idx, 'DYNAMIC_PRESSURE__Pa'],
        'mass': df.loc[delivery_idx, 'MASS__kg'],
        'latitude': df.loc[delivery_idx, 'LATITUDE__deg'],
        'longitude': df.loc[delivery_idx, 'LONGITUDE__deg'],
        'velocity_u': df.loc[delivery_idx, 'STATE_BODY_VEL_U__m_s'],
        'velocity_v': df.loc[delivery_idx, 'STATE_BODY_VEL_V__m_s'],
        'velocity_w': df.loc[delivery_idx, 'STATE_BODY_VEL_W__m_s'],
    }


def analyze_simulations(directory='records'):
    """Load and analyze all simulations"""
    csv_files = glob.glob(os.path.join(directory, 'simulation_data_*.csv'))
    
    if not csv_files:
        print(f"No simulation files found in {directory}")
        return None
    
    print(f"Found {len(csv_files)} simulation files")
    
    results = []
    simulations = []
    
    for file_path in tqdm(csv_files, desc="Analyzing simulations"):
        try:
            sim_id = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                continue
            
            simulations.append((sim_id, df))
            
            success = check_delivery_success(df)
            delivery_metrics = get_delivery_metrics(df) if success else None
            
            result = {
                'simulation_id': sim_id,
                'success': success,
                'max_altitude': df['ALTITUDE__m'].max(),
                'max_mach': df['MACH_NUMBER'].max(),
                'min_flight_path_angle': df['FLIGHT_PATH_ANGLE__deg'].min(),
                'max_flight_path_angle': df['FLIGHT_PATH_ANGLE__deg'].max(),
                'flight_duration': df['TIME__s'].max(),
                'propellant_consumed': df['MASS__kg'].iloc[0] - df['MASS__kg'].iloc[-1],
                'max_dynamic_pressure': df['DYNAMIC_PRESSURE__Pa'].max(),
                'max_acceleration': np.sqrt(
                    df['SPECIFIC_FORCE_X__m_s2']**2 + 
                    df['SPECIFIC_FORCE_Y__m_s2']**2 + 
                    df['SPECIFIC_FORCE_Z__m_s2']**2
                ).max(),
                'delivery_metrics': delivery_metrics
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    simulations.sort(key=lambda x: x[0])
    
    return simulations, results


def plot_trajectory_comparison(simulations, results, output_dir='plots/monte_carlo'):
    """Plot all trajectories with success/failure coloring"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful_ids = {r['simulation_id'] for r in results if r['success']}
    
    fig = plt.figure(figsize=(14, 10))
    
    # Altitude vs Time
    ax1 = plt.subplot(2, 3, 1)
    for sim_id, df in simulations:
        color = COLORS['success'] if sim_id in successful_ids else COLORS['failed']
        ax1.plot(df['TIME__s'], df['ALTITUDE__m']/1000.0, linewidth=2.0, alpha=0.7, color=color)
    ax1.axhline(y=ALTITUDE_MIN/1000.0, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.axhline(y=ALTITUDE_MAX/1000.0, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Time [s]', fontweight='bold')
    ax1.set_ylabel('Altitude [km]', fontweight='bold')
    ax1.set_title('Altitude Profiles', fontweight='bold', fontsize=16)
    ax1.grid(True, alpha=0.4, linewidth=1.2)
    
    # Mach vs Time
    ax2 = plt.subplot(2, 3, 2)
    for sim_id, df in simulations:
        color = COLORS['success'] if sim_id in successful_ids else COLORS['failed']
        ax2.plot(df['TIME__s'], df['MACH_NUMBER'], linewidth=2.0, alpha=0.7, color=color)
    ax2.axhline(y=MACH_THRESHOLD, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Time [s]', fontweight='bold')
    ax2.set_ylabel('Mach Number', fontweight='bold')
    ax2.set_title('Mach Number Profiles', fontweight='bold', fontsize=16)
    ax2.grid(True, alpha=0.4, linewidth=1.2)
    
    # Flight Path Angle vs Time
    ax3 = plt.subplot(2, 3, 3)
    for sim_id, df in simulations:
        color = COLORS['success'] if sim_id in successful_ids else COLORS['failed']
        ax3.plot(df['TIME__s'], df['FLIGHT_PATH_ANGLE__deg'], linewidth=2.0, alpha=0.7, color=color)
    ax3.axhline(y=FLIGHT_PATH_ANGLE_MAX, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax3.axhline(y=-FLIGHT_PATH_ANGLE_MAX, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Time [s]', fontweight='bold')
    ax3.set_ylabel('Flight Path Angle [°]', fontweight='bold')
    ax3.set_title('Flight Path Angle Profiles', fontweight='bold', fontsize=16)
    ax3.grid(True, alpha=0.4, linewidth=1.2)
    
    # Dynamic Pressure vs Time
    ax4 = plt.subplot(2, 3, 4)
    for sim_id, df in simulations:
        color = COLORS['success'] if sim_id in successful_ids else COLORS['failed']
        ax4.plot(df['TIME__s'], df['DYNAMIC_PRESSURE__Pa']/1000.0, linewidth=2.0, alpha=0.7, color=color)
    ax4.set_xlabel('Time [s]', fontweight='bold')
    ax4.set_ylabel('Dynamic Pressure [kPa]', fontweight='bold')
    ax4.set_title('Dynamic Pressure Profiles', fontweight='bold', fontsize=16)
    ax4.grid(True, alpha=0.4, linewidth=1.2)
    
    # Angle of Attack vs Time
    ax5 = plt.subplot(2, 3, 5)
    for sim_id, df in simulations:
        color = COLORS['success'] if sim_id in successful_ids else COLORS['failed']
        ax5.plot(df['TIME__s'], df['ANGLE_OF_ATTACK__deg'], linewidth=2.0, alpha=0.7, color=color)
    ax5.set_xlabel('Time [s]', fontweight='bold')
    ax5.set_ylabel('Angle of Attack [°]', fontweight='bold')
    ax5.set_title('Angle of Attack Profiles', fontweight='bold', fontsize=16)
    ax5.grid(True, alpha=0.4, linewidth=1.2)
    
    # Mass vs Time
    ax6 = plt.subplot(2, 3, 6)
    for sim_id, df in simulations:
        color = COLORS['success'] if sim_id in successful_ids else COLORS['failed']
        ax6.plot(df['TIME__s'], df['MASS__kg'], linewidth=2.0, alpha=0.7, color=color)
    ax6.set_xlabel('Time [s]', fontweight='bold')
    ax6.set_ylabel('Mass [kg]', fontweight='bold')
    ax6.set_title('Mass Profiles', fontweight='bold', fontsize=16)
    ax6.grid(True, alpha=0.4, linewidth=1.2)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=COLORS['success'], lw=3, alpha=0.9),
                   Line2D([0], [0], color=COLORS['failed'], lw=3, alpha=0.9)]
    fig.legend(custom_lines, ['Successful Delivery', 'Failed Delivery'], 
              loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=13, frameon=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_statistics(simulations, results, output_dir='plots/monte_carlo'):
    """Plot statistical analysis with mean and std bands"""
    os.makedirs(output_dir, exist_ok=True)
    
    max_time = max([df['TIME__s'].max() for _, df in simulations])
    time_common = np.linspace(0, max_time, 500)
    
    params = {
        'ALTITUDE__m': ('Altitude [km]', 1000.0),
        'MACH_NUMBER': ('Mach Number', 1.0),
        'FLIGHT_PATH_ANGLE__deg': ('Flight Path Angle [°]', 1.0),
        'DYNAMIC_PRESSURE__Pa': ('Dynamic Pressure [kPa]', 1000.0)
    }
    
    fig = plt.figure(figsize=(12, 10))
    
    for idx, (param, (label, scale)) in enumerate(params.items(), 1):
        ax = plt.subplot(2, 2, idx)
        
        interpolated_data = []
        for sim_id, df in simulations:
            if param in df.columns:
                f = interp1d(df['TIME__s'], df[param], bounds_error=False, fill_value='extrapolate')
                interpolated_data.append(f(time_common))
        
        if interpolated_data:
            data_array = np.array(interpolated_data) / scale
            mean_data = np.mean(data_array, axis=0)
            std_data = np.std(data_array, axis=0)
            
            ax.plot(time_common, mean_data, 'b-', linewidth=2, label='Mean')
            ax.fill_between(time_common, mean_data - std_data, mean_data + std_data, 
                           alpha=0.3, color='blue', label='±1σ')
            ax.fill_between(time_common, mean_data - 2*std_data, mean_data + 2*std_data, 
                           alpha=0.2, color='blue', label='±2σ')
            
            if param == 'ALTITUDE__m':
                ax.axhline(y=ALTITUDE_MIN/scale, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axhline(y=ALTITUDE_MAX/scale, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            elif param == 'MACH_NUMBER':
                ax.axhline(y=MACH_THRESHOLD/scale, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            elif param == 'FLIGHT_PATH_ANGLE__deg':
                ax.axhline(y=FLIGHT_PATH_ANGLE_MAX/scale, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axhline(y=-FLIGHT_PATH_ANGLE_MAX/scale, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Statistics')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_delivery_analysis(results, output_dir='plots/monte_carlo'):
    """Plot delivery success analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    fig = plt.figure(figsize=(14, 10))
    
    # Success rate pie chart
    ax1 = plt.subplot(2, 3, 1)
    sizes = [len(successful), len(failed)]
    colors = [COLORS['success'], COLORS['failed']]
    labels = [f'Successful\n({len(successful)})', f'Failed\n({len(failed)})']
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                         startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    ax1.set_title(f'Delivery Success Rate\n({len(successful)}/{len(results)} missions)', 
                  fontweight='bold', fontsize=16)
    
    # Max Altitude distribution
    ax2 = plt.subplot(2, 3, 2)
    if successful:
        ax2.hist([r['max_altitude']/1000 for r in successful], bins=15, 
                alpha=0.75, color=COLORS['success'], label='Successful', edgecolor='white', linewidth=1.5)
    if failed:
        ax2.hist([r['max_altitude']/1000 for r in failed], bins=15, 
                alpha=0.75, color=COLORS['failed'], label='Failed', edgecolor='white', linewidth=1.5)
    ax2.axvline(x=ALTITUDE_MIN/1000, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.axvline(x=ALTITUDE_MAX/1000, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Maximum Altitude [km]', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Maximum Altitude Distribution', fontweight='bold', fontsize=16)
    ax2.legend(fontsize=12, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.4, linewidth=1.2)
    
    # Max Mach distribution
    ax3 = plt.subplot(2, 3, 3)
    if successful:
        ax3.hist([r['max_mach'] for r in successful], bins=15, 
                alpha=0.75, color=COLORS['success'], label='Successful', edgecolor='white', linewidth=1.5)
    if failed:
        ax3.hist([r['max_mach'] for r in failed], bins=15, 
                alpha=0.75, color=COLORS['failed'], label='Failed', edgecolor='white', linewidth=1.5)
    ax3.axvline(x=MACH_THRESHOLD, color=COLORS['dark'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Maximum Mach Number', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Maximum Mach Distribution', fontweight='bold', fontsize=16)
    ax3.legend(fontsize=12, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.4, linewidth=1.2)
    
    # Propellant consumed
    ax4 = plt.subplot(2, 3, 4)
    if successful:
        ax4.hist([r['propellant_consumed'] for r in successful], bins=15, 
                alpha=0.75, color=COLORS['success'], label='Successful', edgecolor='white', linewidth=1.5)
    if failed:
        ax4.hist([r['propellant_consumed'] for r in failed], bins=15, 
                alpha=0.75, color=COLORS['failed'], label='Failed', edgecolor='white', linewidth=1.5)
    ax4.set_xlabel('Propellant Consumed [kg]', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Propellant Consumption', fontweight='bold', fontsize=16)
    ax4.legend(fontsize=12, frameon=True, shadow=True)
    ax4.grid(True, alpha=0.4, linewidth=1.2)
    
    # Max Dynamic Pressure
    ax5 = plt.subplot(2, 3, 5)
    if successful:
        ax5.hist([r['max_dynamic_pressure']/1000 for r in successful], bins=15, 
                alpha=0.75, color=COLORS['success'], label='Successful', edgecolor='white', linewidth=1.5)
    if failed:
        ax5.hist([r['max_dynamic_pressure']/1000 for r in failed], bins=15, 
                alpha=0.75, color=COLORS['failed'], label='Failed', edgecolor='white', linewidth=1.5)
    ax5.set_xlabel('Max Dynamic Pressure [kPa]', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('Maximum Dynamic Pressure', fontweight='bold', fontsize=16)
    ax5.legend(fontsize=12, frameon=True, shadow=True)
    ax5.grid(True, alpha=0.4, linewidth=1.2)
    
    # Max Acceleration
    ax6 = plt.subplot(2, 3, 6)
    if successful:
        ax6.hist([r['max_acceleration']/9.81 for r in successful], bins=15, 
                alpha=0.75, color=COLORS['success'], label='Successful', edgecolor='white', linewidth=1.5)
    if failed:
        ax6.hist([r['max_acceleration']/9.81 for r in failed], bins=15, 
                alpha=0.75, color=COLORS['failed'], label='Failed', edgecolor='white', linewidth=1.5)
    ax6.set_xlabel('Max Acceleration [g]', fontweight='bold')
    ax6.set_ylabel('Frequency', fontweight='bold')
    ax6.set_title('Maximum Acceleration', fontweight='bold', fontsize=16)
    ax6.legend(fontsize=12, frameon=True, shadow=True)
    ax6.grid(True, alpha=0.4, linewidth=1.2)
    ax6.set_xlabel('Max Acceleration [g]')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Maximum Acceleration')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delivery_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_delivery_conditions(results, output_dir='plots/monte_carlo'):
    """Plot delivery conditions for successful missions"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = [r for r in results if r['success'] and r['delivery_metrics']]
    
    if not successful:
        return
    
    fig = plt.figure(figsize=(14, 10))
    
    # Delivery altitude
    ax1 = plt.subplot(2, 3, 1)
    altitudes = [r['delivery_metrics']['altitude']/1000 for r in successful]
    ax1.hist(altitudes, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(x=np.mean(altitudes), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(altitudes):.2f} km')
    ax1.set_xlabel('Delivery Altitude [km]')
    ax1.set_ylabel('Frequency')
    ax1.set_title('payload Delivery Altitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Delivery Mach
    ax2 = plt.subplot(2, 3, 2)
    machs = [r['delivery_metrics']['mach'] for r in successful]
    ax2.hist(machs, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=np.mean(machs), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(machs):.2f}')
    ax2.set_xlabel('Delivery Mach Number')
    ax2.set_ylabel('Frequency')
    ax2.set_title('payload Delivery Mach')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Delivery time
    ax3 = plt.subplot(2, 3, 3)
    times = [r['delivery_metrics']['time'] for r in successful]
    ax3.hist(times, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(x=np.mean(times), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.2f} s')
    ax3.set_xlabel('Delivery Time [s]')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Time to payload Delivery')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Delivery flight path angle
    ax4 = plt.subplot(2, 3, 4)
    fpas = [r['delivery_metrics']['flight_path_angle'] for r in successful]
    ax4.hist(fpas, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=np.mean(fpas), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(fpas):.2f}°')
    ax4.set_xlabel('Delivery Flight Path Angle [°]')
    ax4.set_ylabel('Frequency')
    ax4.set_title('payload Delivery Flight Path Angle')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Delivery dynamic pressure
    ax5 = plt.subplot(2, 3, 5)
    q_dyns = [r['delivery_metrics']['dynamic_pressure']/1000 for r in successful]
    ax5.hist(q_dyns, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(x=np.mean(q_dyns), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(q_dyns):.2f} kPa')
    ax5.set_xlabel('Delivery Dynamic Pressure [kPa]')
    ax5.set_ylabel('Frequency')
    ax5.set_title('payload Delivery Dynamic Pressure')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Delivery mass
    ax6 = plt.subplot(2, 3, 6)
    masses = [r['delivery_metrics']['mass'] for r in successful]
    ax6.hist(masses, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax6.axvline(x=np.mean(masses), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(masses):.2f} kg')
    ax6.set_xlabel('Delivery Vehicle Mass [kg]')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Rocket Mass at payload Delivery')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delivery_conditions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_analysis(results, output_dir='plots/monte_carlo'):
    """Plot correlation analysis between key parameters"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    fig = plt.figure(figsize=(14, 10))
    
    # Max Altitude vs Max Mach
    ax1 = plt.subplot(2, 3, 1)
    if successful:
        ax1.scatter([r['max_altitude']/1000 for r in successful], 
                   [r['max_mach'] for r in successful], 
                   c='green', alpha=0.6, s=50, label='Successful')
    if failed:
        ax1.scatter([r['max_altitude']/1000 for r in failed], 
                   [r['max_mach'] for r in failed], 
                   c='red', alpha=0.6, s=50, label='Failed')
    ax1.axhline(y=MACH_THRESHOLD, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=ALTITUDE_MIN/1000, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=ALTITUDE_MAX/1000, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Maximum Altitude [km]')
    ax1.set_ylabel('Maximum Mach')
    ax1.set_title('Altitude vs Mach')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Propellant vs Max Altitude
    ax2 = plt.subplot(2, 3, 2)
    if successful:
        ax2.scatter([r['propellant_consumed'] for r in successful], 
                   [r['max_altitude']/1000 for r in successful], 
                   c='green', alpha=0.6, s=50, label='Successful')
    if failed:
        ax2.scatter([r['propellant_consumed'] for r in failed], 
                   [r['max_altitude']/1000 for r in failed], 
                   c='red', alpha=0.6, s=50, label='Failed')
    ax2.set_xlabel('Propellant Consumed [kg]')
    ax2.set_ylabel('Maximum Altitude [km]')
    ax2.set_title('Propellant vs Altitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Max Q vs Max Mach
    ax3 = plt.subplot(2, 3, 3)
    if successful:
        ax3.scatter([r['max_dynamic_pressure']/1000 for r in successful], 
                   [r['max_mach'] for r in successful], 
                   c='green', alpha=0.6, s=50, label='Successful')
    if failed:
        ax3.scatter([r['max_dynamic_pressure']/1000 for r in failed], 
                   [r['max_mach'] for r in failed], 
                   c='red', alpha=0.6, s=50, label='Failed')
    ax3.set_xlabel('Max Dynamic Pressure [kPa]')
    ax3.set_ylabel('Maximum Mach')
    ax3.set_title('Dynamic Pressure vs Mach')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Flight Duration vs Success
    ax4 = plt.subplot(2, 3, 4)
    if successful:
        ax4.hist([r['flight_duration'] for r in successful], bins=15, 
                alpha=0.7, color='green', label='Successful', edgecolor='black')
    if failed:
        ax4.hist([r['flight_duration'] for r in failed], bins=15, 
                alpha=0.7, color='red', label='Failed', edgecolor='black')
    ax4.set_xlabel('Flight Duration [s]')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Flight Duration Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Max Acceleration vs Max Mach
    ax5 = plt.subplot(2, 3, 5)
    if successful:
        ax5.scatter([r['max_acceleration']/9.81 for r in successful], 
                   [r['max_mach'] for r in successful], 
                   c='green', alpha=0.6, s=50, label='Successful')
    if failed:
        ax5.scatter([r['max_acceleration']/9.81 for r in failed], 
                   [r['max_mach'] for r in failed], 
                   c='red', alpha=0.6, s=50, label='Failed')
    ax5.set_xlabel('Max Acceleration [g]')
    ax5.set_ylabel('Maximum Mach')
    ax5.set_title('Acceleration vs Mach')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Flight Path Angle range
    ax6 = plt.subplot(2, 3, 6)
    if successful:
        fpa_ranges = [r['max_flight_path_angle'] - r['min_flight_path_angle'] for r in successful]
        ax6.hist(fpa_ranges, bins=15, alpha=0.7, color='green', label='Successful', edgecolor='black')
    if failed:
        fpa_ranges = [r['max_flight_path_angle'] - r['min_flight_path_angle'] for r in failed]
        ax6.hist(fpa_ranges, bins=15, alpha=0.7, color='red', label='Failed', edgecolor='black')
    ax6.set_xlabel('Flight Path Angle Range [°]')
    ax6.set_ylabel('Frequency')
    ax6.set_title('FPA Variation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results, output_dir='reports'):
    """Generate comprehensive Monte Carlo analysis report"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MONTE CARLO MISSION ANALYSIS - payload DELIVERY")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal Simulations: {len(results)}")
    report_lines.append(f"Successful Deliveries: {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    report_lines.append(f"Failed Deliveries: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("MISSION SUCCESS CRITERIA")
    report_lines.append("=" * 80)
    report_lines.append(f"Mach Number: ≥ {MACH_THRESHOLD}")
    report_lines.append(f"Altitude: {ALTITUDE_MIN/1000:.1f} - {ALTITUDE_MAX/1000:.1f} km")
    report_lines.append(f"Flight Path Angle: |γ| ≤ {FLIGHT_PATH_ANGLE_MAX}°")
    
    if successful:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("SUCCESSFUL DELIVERIES - KEY STATISTICS")
        report_lines.append("=" * 80)
        
        delivery_times = [r['delivery_metrics']['time'] for r in successful if r['delivery_metrics']]
        delivery_alts = [r['delivery_metrics']['altitude']/1000 for r in successful if r['delivery_metrics']]
        delivery_machs = [r['delivery_metrics']['mach'] for r in successful if r['delivery_metrics']]
        delivery_fpas = [r['delivery_metrics']['flight_path_angle'] for r in successful if r['delivery_metrics']]
        
        report_lines.append(f"\npayload Delivery Time [s]:")
        report_lines.append(f"  Mean: {np.mean(delivery_times):.2f}")
        report_lines.append(f"  Std:  {np.std(delivery_times):.2f}")
        report_lines.append(f"  Min:  {np.min(delivery_times):.2f}")
        report_lines.append(f"  Max:  {np.max(delivery_times):.2f}")
        
        report_lines.append(f"\npayload Delivery Altitude [km]:")
        report_lines.append(f"  Mean: {np.mean(delivery_alts):.2f}")
        report_lines.append(f"  Std:  {np.std(delivery_alts):.2f}")
        report_lines.append(f"  Min:  {np.min(delivery_alts):.2f}")
        report_lines.append(f"  Max:  {np.max(delivery_alts):.2f}")
        
        report_lines.append(f"\npayload Delivery Mach Number:")
        report_lines.append(f"  Mean: {np.mean(delivery_machs):.2f}")
        report_lines.append(f"  Std:  {np.std(delivery_machs):.2f}")
        report_lines.append(f"  Min:  {np.min(delivery_machs):.2f}")
        report_lines.append(f"  Max:  {np.max(delivery_machs):.2f}")
        
        report_lines.append(f"\npayload Delivery Flight Path Angle [°]:")
        report_lines.append(f"  Mean: {np.mean(delivery_fpas):.2f}")
        report_lines.append(f"  Std:  {np.std(delivery_fpas):.2f}")
        report_lines.append(f"  Min:  {np.min(delivery_fpas):.2f}")
        report_lines.append(f"  Max:  {np.max(delivery_fpas):.2f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ALL SIMULATIONS - PERFORMANCE METRICS")
    report_lines.append("=" * 80)
    
    max_alts = [r['max_altitude']/1000 for r in results]
    max_machs = [r['max_mach'] for r in results]
    propellants = [r['propellant_consumed'] for r in results]
    max_qs = [r['max_dynamic_pressure']/1000 for r in results]
    max_accs = [r['max_acceleration']/9.81 for r in results]
    
    report_lines.append(f"\nMaximum Altitude Achieved [km]:")
    report_lines.append(f"  Mean: {np.mean(max_alts):.2f}")
    report_lines.append(f"  Std:  {np.std(max_alts):.2f}")
    report_lines.append(f"  Min:  {np.min(max_alts):.2f}")
    report_lines.append(f"  Max:  {np.max(max_alts):.2f}")
    
    report_lines.append(f"\nMaximum Mach Number Achieved:")
    report_lines.append(f"  Mean: {np.mean(max_machs):.2f}")
    report_lines.append(f"  Std:  {np.std(max_machs):.2f}")
    report_lines.append(f"  Min:  {np.min(max_machs):.2f}")
    report_lines.append(f"  Max:  {np.max(max_machs):.2f}")
    
    report_lines.append(f"\nPropellant Consumed [kg]:")
    report_lines.append(f"  Mean: {np.mean(propellants):.2f}")
    report_lines.append(f"  Std:  {np.std(propellants):.2f}")
    report_lines.append(f"  Min:  {np.min(propellants):.2f}")
    report_lines.append(f"  Max:  {np.max(propellants):.2f}")
    
    report_lines.append(f"\nMaximum Dynamic Pressure [kPa]:")
    report_lines.append(f"  Mean: {np.mean(max_qs):.2f}")
    report_lines.append(f"  Std:  {np.std(max_qs):.2f}")
    report_lines.append(f"  Min:  {np.min(max_qs):.2f}")
    report_lines.append(f"  Max:  {np.max(max_qs):.2f}")
    
    report_lines.append(f"\nMaximum Acceleration [g]:")
    report_lines.append(f"  Mean: {np.mean(max_accs):.2f}")
    report_lines.append(f"  Std:  {np.std(max_accs):.2f}")
    report_lines.append(f"  Min:  {np.min(max_accs):.2f}")
    report_lines.append(f"  Max:  {np.max(max_accs):.2f}")
    
    if successful and failed:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("COMPARISON: SUCCESSFUL vs FAILED")
        report_lines.append("=" * 80)
        
        succ_max_alt = np.mean([r['max_altitude']/1000 for r in successful])
        fail_max_alt = np.mean([r['max_altitude']/1000 for r in failed])
        report_lines.append(f"\nMean Maximum Altitude [km]:")
        report_lines.append(f"  Successful: {succ_max_alt:.2f}")
        report_lines.append(f"  Failed:     {fail_max_alt:.2f}")
        report_lines.append(f"  Difference: {succ_max_alt - fail_max_alt:.2f}")
        
        succ_max_mach = np.mean([r['max_mach'] for r in successful])
        fail_max_mach = np.mean([r['max_mach'] for r in failed])
        report_lines.append(f"\nMean Maximum Mach:")
        report_lines.append(f"  Successful: {succ_max_mach:.2f}")
        report_lines.append(f"  Failed:     {fail_max_mach:.2f}")
        report_lines.append(f"  Difference: {succ_max_mach - fail_max_mach:.2f}")
        
        succ_prop = np.mean([r['propellant_consumed'] for r in successful])
        fail_prop = np.mean([r['propellant_consumed'] for r in failed])
        report_lines.append(f"\nMean Propellant Consumed [kg]:")
        report_lines.append(f"  Successful: {succ_prop:.2f}")
        report_lines.append(f"  Failed:     {fail_prop:.2f}")
        report_lines.append(f"  Difference: {succ_prop - fail_prop:.2f}")
    
    report_lines.append("\n" + "=" * 80)
    
    report_path = os.path.join(output_dir, 'monte_carlo_analysis.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    for line in report_lines:
        print(line)


def prepare_ml_dataset(results):
    """Prepare dataset for machine learning"""
    features = []
    targets_success = []
    targets_altitude = []
    targets_mach = []
    
    for r in results:
        features.append([
            r['max_altitude'],
            r['max_mach'],
            r['flight_duration'],
            r['propellant_consumed'],
            r['max_dynamic_pressure'],
            r['max_acceleration'],
            r['min_flight_path_angle'],
            r['max_flight_path_angle']
        ])
        targets_success.append(1 if r['success'] else 0)
        targets_altitude.append(r['max_altitude'])
        targets_mach.append(r['max_mach'])
    
    feature_names = ['Max_Altitude', 'Max_Mach', 'Flight_Duration', 'Propellant', 
                    'Max_Q', 'Max_Accel', 'Min_FPA', 'Max_FPA']
    
    return np.array(features), np.array(targets_success), np.array(targets_altitude), np.array(targets_mach), feature_names


def plot_pca_tsne(X, y_success, feature_names, output_dir='plots/monte_carlo'):
    """Plot PCA and t-SNE visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig = plt.figure(figsize=(16, 7))
    
    # PCA
    ax1 = plt.subplot(1, 2, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    successful = y_success == 1
    failed = y_success == 0
    
    ax1.scatter(X_pca[successful, 0], X_pca[successful, 1], c=COLORS['success'], s=150, 
               alpha=0.8, edgecolors='white', linewidths=2.0, label='Successful', marker='o')
    ax1.scatter(X_pca[failed, 0], X_pca[failed, 1], c=COLORS['failed'], s=150, 
               alpha=0.8, linewidths=2.0, label='Failed', marker='x')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'PCA - Total Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%', 
                 fontsize=16, fontweight='bold')
    ax1.legend(fontsize=13, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.4, linewidth=1.2)
    
    # t-SNE
    ax2 = plt.subplot(1, 2, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    ax2.scatter(X_tsne[successful, 0], X_tsne[successful, 1], c=COLORS['success'], s=150, 
               alpha=0.8, edgecolors='white', linewidths=2.0, label='Successful', marker='o')
    ax2.scatter(X_tsne[failed, 0], X_tsne[failed, 1], c=COLORS['failed'], s=150, 
               alpha=0.8, linewidths=2.0, label='Failed', marker='x')
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax2.set_title('t-SNE Visualization', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=13, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.4, linewidth=1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimensionality_reduction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] PCA/t-SNE visualization (Variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%)")


def perform_clustering(X, y_success, output_dir='plots/monte_carlo'):
    """Perform clustering analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig = plt.figure(figsize=(16, 7))
    
    # K-Means
    ax1 = plt.subplot(1, 2, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(X_scaled)
    silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, s=150, 
                         cmap='tab10', alpha=0.7, edgecolors='white', linewidths=2.0)
    
    successful = y_success == 1
    ax1.scatter(X_pca[successful, 0], X_pca[successful, 1], s=400, 
               facecolors='none', edgecolors=COLORS['success'], linewidths=3, marker='o', label='Successful')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'K-Means Clustering (k=3)\nSilhouette Score: {silhouette_kmeans:.3f}', 
                 fontsize=16, fontweight='bold')
    ax1.legend(fontsize=13, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.4, linewidth=1.2)
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # DBSCAN
    ax2 = plt.subplot(1, 2, 2)
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    clusters_dbscan = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
    n_noise = list(clusters_dbscan).count(-1)
    
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_dbscan, s=150, 
                         cmap='tab10', alpha=0.7, edgecolors='white', linewidths=2.0)
    
    ax2.scatter(X_pca[successful, 0], X_pca[successful, 1], s=400, 
               facecolors='none', edgecolors=COLORS['success'], linewidths=3, marker='o', label='Successful')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'DBSCAN Clustering\nClusters: {n_clusters}, Noise: {n_noise}', 
                 fontsize=16, fontweight='bold')
    ax2.legend(fontsize=13, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.4, linewidth=1.2)
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Clustering (K-Means silhouette: {silhouette_kmeans:.3f}, DBSCAN clusters: {n_clusters})")


def train_classification_models(X, y, feature_names, output_dir='plots/monte_carlo'):
    """Train and optimize classification models with Optuna"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if len(np.unique(y)) < 2:
        print("  [SKIP] Insufficient classes for classification (all same class)")
        return None
    
    # Adjust test size based on sample count
    test_size = 0.2 if len(X) >= 10 else max(0.1, 1.0/len(X))
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                             random_state=42, stratify=y if len(X) >= 10 else None)
    except ValueError:
        # If stratify fails, split without it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Check if test set has both classes
    if len(np.unique(y_test)) < 2:
        print("  [SKIP] Test set has only one class, cannot evaluate classification")
        return None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # SVM with RBF kernel
    print("\n  Training SVM Classifier (RBF kernel)...")
    def svm_objective(trial):
        C = trial.suggest_float('C', 0.01, 1000, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 10.0, log=True)
        
        svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
        # Use LOO if too few samples
        cv_folds = len(X_train) if len(X_train) < 5 else max(2, min(5, len(X_train)//2))
        cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=cv_folds, 
                                   scoring='accuracy', n_jobs=-1)
        return cv_scores.mean()
    
    study_svm = optuna.create_study(direction='maximize', sampler=optuna.samplers.CmaEsSampler(seed=42))
    study_svm.optimize(svm_objective, n_trials=1000, show_progress_bar=False)
    
    best_svm = SVC(**study_svm.best_params, kernel='rbf', random_state=42)
    best_svm.fit(X_train_scaled, y_train)
    y_pred_svm = best_svm.predict(X_test_scaled)
    
    results['SVM-RBF'] = {
        'model': best_svm,
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'best_params': study_svm.best_params
    }
    
    # Save models
    with open('models/classification_models.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('models/scaler_classification.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    y_pred = results['SVM-RBF']['model'].predict(X_test_scaled)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap='RdYlGn', aspect='auto', alpha=0.8)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Failed', 'Success'], fontsize=14)
    ax.set_yticklabels(['Failed', 'Success'], fontsize=14)
    ax.set_xlabel('Predicted', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=16, fontweight='bold')
    ax.set_title(f'SVM-RBF Classification\nAccuracy: {results["SVM-RBF"]["accuracy"]:.3f}\nC={results["SVM-RBF"]["best_params"]["C"]:.3f}, gamma={results["SVM-RBF"]["best_params"]["gamma"]:.4f}', 
                fontsize=18, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   fontsize=24, fontweight='bold', color=text_color)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] SVM-RBF Classification (Accuracy: {results['SVM-RBF']['accuracy']:.3f}, C: {results['SVM-RBF']['best_params']['C']:.3f}, gamma: {results['SVM-RBF']['best_params']['gamma']:.4f})")
    
    return results


def train_regression_models(X, y_alt, y_mach, feature_names, output_dir='plots/monte_carlo'):
    """Train and optimize regression models with Optuna"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Adjust test size based on sample count
    test_size = 0.2 if len(X) >= 10 else max(0.1, 1.0/len(X))
    
    X_train, X_test, y_alt_train, y_alt_test = train_test_split(X, y_alt, test_size=test_size, random_state=42)
    _, _, y_mach_train, y_mach_test = train_test_split(X, y_mach, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    print("\n  Training Altitude Regression (SVR-RBF)...")
    def svr_alt_objective(trial):
        C = trial.suggest_float('C', 0.01, 1000, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 10.0, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
        
        svr = SVR(C=C, gamma=gamma, epsilon=epsilon, kernel='rbf')
        # Use LOO if too few samples
        cv_folds = len(X_train) if len(X_train) < 5 else max(2, min(5, len(X_train)//2))
        cv_scores = cross_val_score(svr, X_train_scaled, y_alt_train, cv=cv_folds, 
                                   scoring='r2', n_jobs=-1)
        return cv_scores.mean()
    
    study_alt = optuna.create_study(direction='maximize', sampler=optuna.samplers.CmaEsSampler(seed=42))
    study_alt.optimize(svr_alt_objective, n_trials=1000, show_progress_bar=False)
    
    best_svr_alt = SVR(**study_alt.best_params, kernel='rbf')
    best_svr_alt.fit(X_train_scaled, y_alt_train)
    y_alt_pred = best_svr_alt.predict(X_test_scaled)
    
    results['Altitude'] = {
        'model': best_svr_alt,
        'r2': r2_score(y_alt_test, y_alt_pred),
        'rmse': np.sqrt(mean_squared_error(y_alt_test, y_alt_pred)),
        'predictions': y_alt_pred,
        'actual': y_alt_test,
        'best_params': study_alt.best_params
    }
    
    # Mach prediction
    print("  Training Mach Number Regression (SVR-RBF)...")
    def svr_mach_objective(trial):
        C = trial.suggest_float('C', 0.01, 1000, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 10.0, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
        
        svr = SVR(C=C, gamma=gamma, epsilon=epsilon, kernel='rbf')
        # Use LOO if too few samples
        cv_folds = len(X_train) if len(X_train) < 5 else max(2, min(5, len(X_train)//2))
        cv_scores = cross_val_score(svr, X_train_scaled, y_mach_train, cv=cv_folds, 
                                   scoring='r2', n_jobs=-1)
        return cv_scores.mean()
    
    study_mach = optuna.create_study(direction='maximize', sampler=optuna.samplers.CmaEsSampler(seed=42))
    study_mach.optimize(svr_mach_objective, n_trials=1000, show_progress_bar=False)
    
    best_svr_mach = SVR(**study_mach.best_params, kernel='rbf')
    best_svr_mach.fit(X_train_scaled, y_mach_train)
    y_mach_pred = best_svr_mach.predict(X_test_scaled)
    
    results['Mach'] = {
        'model': best_svr_mach,
        'r2': r2_score(y_mach_test, y_mach_pred),
        'rmse': np.sqrt(mean_squared_error(y_mach_test, y_mach_pred)),
        'predictions': y_mach_pred,
        'actual': y_mach_test,
        'best_params': study_mach.best_params
    }
    
    with open('models/regression_models.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('models/scaler_regression.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        ax.scatter(result['actual'], result['predictions'], alpha=0.75, s=150, 
                  edgecolors='white', linewidths=2.0, color=COLORS['primary'])
        
        min_val = min(result['actual'].min(), result['predictions'].min())
        max_val = max(result['actual'].max(), result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], color=COLORS['failed'], 
                linestyle='--', linewidth=3.0, label='Perfect Prediction')
        
        unit = 'km' if name == 'Altitude' else ''
        scale = 1000 if name == 'Altitude' else 1
        
        ax.set_xlabel(f'Actual {name} [{unit}]', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Predicted {name} [{unit}]', fontsize=14, fontweight='bold')
        ax.set_title(f'{name} Prediction\nR2 = {result["r2"]:.3f}, RMSE = {result["rmse"]/scale:.2f}', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=13, frameon=True, shadow=True)
        ax.grid(True, alpha=0.4, linewidth=1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] SVR-RBF Regression (Altitude R2: {results['Altitude']['r2']:.3f}, Mach R2: {results['Mach']['r2']:.3f})")
    
    return results


def _process_single_simulation(sim_id):
    """Process a single simulation - must be at module level for pickling"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'records', f'simulation_data_{sim_id}.csv')
    plots_dir = os.path.join(script_dir, 'plots', f'simulation_{sim_id}')
    reports_dir = os.path.join(script_dir, 'reports', f'simulation_{sim_id}')
    records_dir = os.path.join(script_dir, 'records', f'simulation_{sim_id}')
    plot_record_script = os.path.join(script_dir, 'plot_record.py')
    
    if not os.path.exists(input_file):
        return sim_id, False, f"File not found: {input_file}"
    
    try:
        result = subprocess.run(
            ['python', plot_record_script, 
             '--input', input_file,
             '--plots-dir', plots_dir,
             '--reports-dir', reports_dir,
             '--records-dir', records_dir],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=script_dir,
            check=False
        )
        
        if result.returncode == 0:
            if os.path.exists(plots_dir) and os.path.exists(reports_dir):
                plot_files = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
                report_files = len([f for f in os.listdir(reports_dir) if f.endswith('.txt')])
                return sim_id, True, f"Success (plots: {plot_files}, reports: {report_files})"
            else:
                return sim_id, False, "Output directories not created"
        else:
            error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            return sim_id, False, f"Error: {error_msg}"
    except subprocess.TimeoutExpired:
        return sim_id, False, "Timeout (10 min)"
    except Exception as e:
        return sim_id, False, f"Exception: {str(e)}"


def process_individual_plots(simulations, max_workers=4):
    """Process individual plots for each simulation sequentially (one at a time)"""
    print("\n" + "=" * 80)
    print("GENERATING INDIVIDUAL SIMULATION PLOTS")
    print("=" * 80 + "\n")
    
    sim_ids = [sim_id for sim_id, _ in simulations]
    
    successful = 0
    failed = 0
    
    for sim_id in tqdm(sim_ids, desc="Processing simulations"):
        sim_id_result, success, message = _process_single_simulation(sim_id)
        if success:
            successful += 1
            print(f"  [OK] Simulation {sim_id}: {message}")
        else:
            failed += 1
            print(f"  [X] Simulation {sim_id}: {message}")
    
    print(f"\n{'='*80}")
    print(f"Individual plots processing complete: {successful} successful, {failed} failed")
    print(f"{'='*80}\n")
    
    return successful, failed


def main():
    print("\n" + "=" * 80)
    print("MONTE CARLO ANALYSIS - payload DELIVERY MISSION")
    print("=" * 80 + "\n")
    
    simulations, results = analyze_simulations('records')
    
    if not simulations or not results:
        print("No simulations to analyze.")
        return
    
    print(f"\nProcessed {len(simulations)} simulations")
    print(f"Successful deliveries: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    process_individual_plots(simulations, max_workers=4)
    
    os.makedirs('plots/monte_carlo', exist_ok=True)
    
    print("\nGenerating statistical plots...")
    plot_trajectory_comparison(simulations, results)
    print("  [OK] Trajectory comparison")
    
    plot_statistics(simulations, results)
    print("  [OK] Statistical analysis")
    
    plot_delivery_analysis(results)
    print("  [OK] Delivery analysis")
    
    plot_delivery_conditions(results)
    print("  [OK] Delivery conditions")
    
    plot_correlation_analysis(results)
    print("  [OK] Correlation analysis")
    
    print("\nGenerating statistical report...")
    generate_report(results)
    
    print("\n" + "=" * 80)
    print("MACHINE LEARNING ANALYSIS")
    print("=" * 80)
    
    X, y_success, y_altitude, y_mach, feature_names = prepare_ml_dataset(results)
    print(f"\nDataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\nDimensionality Reduction...")
    plot_pca_tsne(X, y_success, feature_names)
    
    print("\nClustering Analysis...")
    perform_clustering(X, y_success)
    
    classification_results = None
    if len(np.unique(y_success)) >= 2:
        print("\nTraining Classification Models (Success Prediction)...")
        classification_results = train_classification_models(X, y_success, feature_names)
    
    print("\nTraining Regression Models...")
    regression_results = train_regression_models(X, y_altitude, y_mach, feature_names)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Statistical plots saved in: plots/monte_carlo/")
    print(f"ML models saved in: models/")
    print(f"Reports saved in: reports/")
    
    # List only files that actually exist
    print("\nGenerated files:")
    mc_dir = 'plots/monte_carlo'
    generated_files = [
        ('trajectory_comparison.png', 'Trajectory comparison'),
        ('statistics.png', 'Statistical analysis'),
        ('delivery_analysis.png', 'Delivery analysis'),
        ('delivery_conditions.png', 'Delivery conditions'),
        ('correlation_analysis.png', 'Correlation analysis'),
        ('dimensionality_reduction.png', 'PCA/t-SNE'),
        ('clustering_analysis.png', 'K-Means/DBSCAN'),
    ]
    if classification_results:
        generated_files.append(('classification_confusion_matrix.png', 'SVM-RBF Classification'))
    if regression_results:
        generated_files.append(('regression_predictions.png', 'SVR-RBF Regression'))
    
    for filename, desc in generated_files:
        filepath = os.path.join(mc_dir, filename)
        if os.path.exists(filepath):
            print(f"  - {filename} ({desc})")
    
    report_path = 'reports/monte_carlo_analysis.txt'
    if os.path.exists(report_path):
        print(f"  - monte_carlo_analysis.txt (report)")


if __name__ == "__main__":
    main()
