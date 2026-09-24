# Python standard libraries
import os
import sys
import argparse

# 3rd party libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import pandas as pd

# VAHSimulator library
from vahsimulator.parameters import length_rv, x_cg_rv, dr, lat_ref, lon_ref, alt_ref, we
from vahsimulator.utils import geodetic_to_ecef, ecef_to_ned, quaternion_to_euler, eci_to_body_quaternion, body_to_eci_quaternion, skew_matrix

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = (-99, 99)

column_groups = {
    'Atmosphere': ['SPEED_OF_SOUND__m_s', 'TEMPERATURE__K', 'PRESSURE__Pa', 'DENSITY__kg_m3'],
    
    'Wind - Linear Velocity': ['WIND_BODY_X__m_s', 'WIND_BODY_Y__m_s', 'WIND_BODY_Z__m_s'],
    'Wind - Angular Velocity': ['WIND_BODY_P__rad_s', 'WIND_BODY_Q__rad_s', 'WIND_BODY_R__rad_s'],
    
    'Propulsion Forces': ['PROP_FORCE_X__N', 'PROP_FORCE_Y__N', 'PROP_FORCE_Z__N'],
    'Propulsion Moments': ['PROP_MOMENT_X__Nm', 'PROP_MOMENT_Y__Nm', 'PROP_MOMENT_Z__Nm'],
    'Propulsion Properties': ['THRUST__N', 'MASS__kg', 'X_CG__m', 'EXHAUST_VELOCITY__m_s', 'MASS_FLOW_RATE__kg_s'],
    'Inertia Properties': ['IXX__kgm2', 'IYY__kgm2', 'IZZ__kgm2', 'IXY__kgm2', 'IXZ__kgm2', 'IYZ__kgm2'],
    'Inertia Rates': ['IXX_RATE__kgm2_s', 'IYY_RATE__kgm2_s', 'IZZ_RATE__kgm2_s'],
    
    'Aerodynamic Forces': ['AERO_FORCE_X__N', 'AERO_FORCE_Y__N', 'AERO_FORCE_Z__N'],
    'Aerodynamic Moments': ['AERO_MOMENT_X__Nm', 'AERO_MOMENT_Y__Nm', 'AERO_MOMENT_Z__Nm'],
    'Aerodynamic Properties': ['TRUE_AIRSPEED__m_s', 'DYNAMIC_PRESSURE__Pa', 'ANGLE_OF_ATTACK__deg', 
                              'SIDESLIP_ANGLE__deg', 'MACH_NUMBER', 'STATIC_MARGIN'],
    'Aerodynamic Derivatives': ['NA__m_s2', 'ND__m_s2', 'MA__1_s2', 'MQ__1_s2', 'MD__1_s2', 'MD_TVC__1_s2', 'MD_FIN__1_s2', 'LLDA__1_s2', 'LLP__1_s2'],
    'Aerodynamic Coefficients': ['CA', 'CY', 'CN', 'CLL', 'CM', 'CLN'],
    
    'Gravity': ['GRAVITY_FORCE_X__N', 'GRAVITY_FORCE_Y__N', 'GRAVITY_FORCE_Z__N'],
    
    'Position ECI': ['STATE_ECI_POS_X__m', 'STATE_ECI_POS_Y__m', 'STATE_ECI_POS_Z__m'],
    'Attitude Quaternion': ['STATE_QUAT_W', 'STATE_QUAT_X', 'STATE_QUAT_Y', 'STATE_QUAT_Z'],
    'Body Velocity': ['STATE_BODY_VEL_U__m_s', 'STATE_BODY_VEL_V__m_s', 'STATE_BODY_VEL_W__m_s'],
    'Specific Force': ['SPECIFIC_FORCE_X__m_s2', 'SPECIFIC_FORCE_Y__m_s2', 'SPECIFIC_FORCE_Z__m_s2'],
    'Body Rates': ['BODY_RATE_P__rad_s', 'BODY_RATE_Q__rad_s', 'BODY_RATE_R__rad_s'],
    'Geodetic Position': ['LATITUDE__deg', 'LONGITUDE__deg', 'ALTITUDE__m'],
    'Orientation Angles': ['HEADING__deg', 'FLIGHT_PATH_ANGLE__deg'],
    'Attitude Euler NED': ['ROLL_NED__deg', 'PITCH_NED__deg', 'YAW_NED__deg'],
    
    'Heating': ['STAGNATION_TEMPERATURE__K'],
    
    'IMU Specific Force': ['IMU_SPECIFIC_FORCE_X__m_s2', 'IMU_SPECIFIC_FORCE_Y__m_s2', 'IMU_SPECIFIC_FORCE_Z__m_s2'],
    'IMU Body Rates': ['IMU_BODY_RATE_P__rad_s', 'IMU_BODY_RATE_Q__rad_s', 'IMU_BODY_RATE_R__rad_s'],
    
    'GNSS Position ECI': ['GNSS_POS_ECI_X__m', 'GNSS_POS_ECI_Y__m', 'GNSS_POS_ECI_Z__m'],
    'GNSS Velocity ECI': ['GNSS_VEL_ECI_X__m_s', 'GNSS_VEL_ECI_Y__m_s', 'GNSS_VEL_ECI_Z__m_s'],
    'GNSS Yaw': ['GNSS_YAW__deg'],
    
    'Nav Quaternion': ['NAV_QUAT_W', 'NAV_QUAT_X', 'NAV_QUAT_Y', 'NAV_QUAT_Z'],
    'Nav Position ECI': ['NAV_POS_ECI_X__m', 'NAV_POS_ECI_Y__m', 'NAV_POS_ECI_Z__m'],
    'Nav Velocity ECI': ['NAV_VEL_ECI_X__m_s', 'NAV_VEL_ECI_Y__m_s', 'NAV_VEL_ECI_Z__m_s'],
    'Nav Acc Bias': ['NAV_ACC_BIAS_X__m_s2', 'NAV_ACC_BIAS_Y__m_s2', 'NAV_ACC_BIAS_Z__m_s2'],
    'Nav Gyro Bias': ['NAV_GYRO_BIAS_X__rad_s', 'NAV_GYRO_BIAS_Y__rad_s', 'NAV_GYRO_BIAS_Z__rad_s'],
    'Nav Quaternion Variance': ['NAV_VAR_QUAT_W', 'NAV_VAR_QUAT_X', 'NAV_VAR_QUAT_Y', 'NAV_VAR_QUAT_Z'],
    'Nav Position Variance': ['NAV_VAR_POS_ECI_X__m2', 'NAV_VAR_POS_ECI_Y__m2', 'NAV_VAR_POS_ECI_Z__m2'],
    'Nav Velocity Variance': ['NAV_VAR_VEL_ECI_X__m2_s2', 'NAV_VAR_VEL_ECI_Y__m2_s2', 'NAV_VAR_VEL_ECI_Z__m2_s2'],
    'Nav Acc Bias Variance': ['NAV_VAR_ACC_BIAS_X__m2_s4', 'NAV_VAR_ACC_BIAS_Y__m2_s4', 'NAV_VAR_ACC_BIAS_Z__m2_s4'],
    'Nav Gyro Bias Variance': ['NAV_VAR_GYRO_BIAS_X__rad2_s2', 'NAV_VAR_GYRO_BIAS_Y__rad2_s2', 'NAV_VAR_GYRO_BIAS_Z__rad2_s2'],
    
    'Guidance Commands': ['NORMAL_ACC_CMD__m_s2', 'LATERAL_ACC_CMD__m_s2', 'GAMMA_CMD__deg', 'PITCH_CMD__deg', 'YAW_CMD__deg'],
    'Target Position NED': ['TARGET_POS_NED_X__m', 'TARGET_POS_NED_Y__m', 'TARGET_POS_NED_Z__m'],
    
    'Control Commands TVC': ['DELTA_Q_TVC_CMD__deg', 'DELTA_R_TVC_CMD__deg'],
    'Control Commands Canard': ['DELTA_P_CANARD_CMD__deg', 'DELTA_Q_CANARD_CMD__deg', 'DELTA_R_CANARD_CMD__deg'],
    'Control Commands Fin': ['DELTA_A_FIN_CMD__deg', 'DELTA_E_FIN_CMD__deg'],
    
    'TVC Deflections': ['TVC_DELTA_Q__deg', 'TVC_DELTA_R__deg'],
    
    'RCS Moments': ['RCS_MOMENT_X__Nm', 'RCS_MOMENT_Y__Nm', 'RCS_MOMENT_Z__Nm'],
    
    'Canard Deflections': ['CANARD_DELTA_1__deg', 'CANARD_DELTA_2__deg', 'CANARD_DELTA_3__deg', 'CANARD_DELTA_4__deg'],
    
    'Fin Deflections': ['FIN_DELTA_LEFT__deg', 'FIN_DELTA_RIGHT__deg'],
    
    'Battery': ['SOC__percent', 'VOLTAGE__V', 'CURRENT__A', 'C_RATING__As']
}


def add_phase_regions(ax, time, phase_data, max_phase=4):
    phase_colors = {
        1: '#FFE5E5',
        2: '#E5F5FF',
        3: '#E5FFE5',
        4: '#FFF5E5'
    }
    
    phases_found = set()
    phase_regions = []
    
    if phase_data is None or len(phase_data) == 0:
        return phases_found, phase_regions
    
    current_phase = None
    start_time = time[0]
    
    for i in range(len(phase_data)):
        if phase_data[i] is None or np.isnan(phase_data[i]):
            continue
            
        phase_val = int(phase_data[i])
        
        if phase_val > max_phase:
            continue
            
        phases_found.add(phase_val)
        
        if phase_val != current_phase:
            if current_phase is not None:
                end_time = time[i]
                if current_phase in phase_colors:
                    ax.axvspan(start_time, end_time, 
                             facecolor=phase_colors[current_phase], 
                             alpha=0.7, zorder=0)
                    phase_regions.append((start_time, end_time, current_phase))
            
            current_phase = phase_val
            start_time = time[i]
    
    if current_phase is not None and current_phase in phase_colors:
        ax.axvspan(start_time, time[-1], 
                 facecolor=phase_colors[current_phase], 
                 alpha=0.7, zorder=0)
        phase_regions.append((start_time, time[-1], current_phase))
    
    return phases_found, phase_regions


def add_phase_labels(ax, phase_regions, data_x=None, data_y=None):
    phase_names = {
        1: 'Phase 1',
        2: 'Phase 2',
        3: 'Phase 3',
        4: 'Phase 4'
    }
    
    if not phase_regions:
        return
    
    y_lim = ax.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    
    for start_time, end_time, phase_val in phase_regions:
        center_time = (start_time + end_time) / 2
        
        if data_x is not None and data_y is not None:
            region_mask = (data_x >= start_time) & (data_x <= end_time)
            if np.any(region_mask):
                region_data = data_y[region_mask]
                region_data = region_data[~np.isnan(region_data)]
                
                if len(region_data) > 0:
                    test_positions = np.array([0.97, 0.93, 0.88, 0.82, 0.75, 
                                              0.25, 0.18, 0.12, 0.07, 0.03])
                    text_height = y_range * 0.07
                    best_score = float('-inf')
                    best_pos = 0.97
                    best_va = 'top'
                    
                    for rel_pos in test_positions:
                        y_test = y_lim[0] + y_range * rel_pos
                        
                        if rel_pos > 0.5:
                            y_box_min = y_test - text_height
                            y_box_max = y_test
                            va = 'top'
                        else:
                            y_box_min = y_test
                            y_box_max = y_test + text_height
                            va = 'bottom'
                        
                        conflicts = np.sum((region_data >= y_box_min) & (region_data <= y_box_max))
                        
                        margin_bonus = 0
                        if rel_pos > 0.85 or rel_pos < 0.15:
                            margin_bonus = 50
                        elif rel_pos > 0.75 or rel_pos < 0.25:
                            margin_bonus = 20
                        
                        score = -conflicts * 10 + margin_bonus
                        
                        if score > best_score:
                            best_score = score
                            best_pos = rel_pos
                            best_va = va
                    
                    y_pos = y_lim[0] + y_range * best_pos
                    va = best_va
                else:
                    y_pos = y_lim[1] - y_range * 0.03
                    va = 'top'
            else:
                y_pos = y_lim[1] - y_range * 0.03
                va = 'top'
        else:
            y_pos = y_lim[1] - y_range * 0.03
            va = 'top'
        
        ax.text(center_time, y_pos, phase_names[phase_val],
                ha='center', va=va, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.9, linewidth=0.5),
                zorder=10)


def extract_unit(column_name):
    if '__' in column_name:
        unit = column_name.split('__')[1]
        if unit == 'deg':
            return '°'
        elif unit == 'kgm2':
            return 'kg·m²'
        elif unit == 'kgm2_s':
            return 'kg·m²/s'
        elif unit == 'kg_s':
            return 'kg/s'
        elif unit == 'm_s':
            return 'm/s'
        elif unit == 'm_s2':
            return 'm/s²'
        elif unit == '1_s2':
            return '1/s²'
        elif unit == 'rad_s':
            return 'rad/s'
        elif unit == 'rad':
            return 'rad'
        elif unit == 'm2':
            return 'm²'
        elif unit == 'm2_s2':
            return 'm²/s²'
        elif unit == 'm2_s4':
            return 'm²/s⁴'
        elif unit == 'rad2_s2':
            return 'rad²/s²'
        elif unit == 's':
            return 's'
        elif unit == 'K':
            return 'K'
        elif unit == 'Pa':
            return 'Pa'
        elif unit == 'kg_m3':
            return 'kg/m³'
        elif unit == 'N':
            return 'N'
        elif unit == 'Nm':
            return 'N·m'
        elif unit == 'kg':
            return 'kg'
        elif unit == 'm':
            return 'm'
        elif unit == 'percent':
            return '%'
        elif unit == 'V':
            return 'V'
        elif unit == 'A':
            return 'A'
        elif unit == 'As':
            return 'A·s'
        else:
            return unit
    return ''


def generate_labels(columns):
    labels = {}
    for col in columns:
        if '__' in col:
            base_name = col.split('__')[0].replace('_', ' ').title()
            unit = extract_unit(col)
            if unit:
                labels[col] = f"{base_name} [{unit}]"
            else:
                labels[col] = base_name
        else:
            labels[col] = col.replace('_', ' ').title()
    return labels


def plot_record(df, output_dir='plots', max_phase=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
    
    for group_name, columns in tqdm(column_groups.items()):
        if not all(col in df.columns for col in columns):
            continue
            
        n_cols = len(columns)
        fig = plt.figure(figsize=(8, 2 + 1.5*n_cols))
        gs = GridSpec(n_cols, 1, figure=fig, hspace=0.3)
        
        labels = generate_labels(columns)
        
        for i, col in enumerate(columns):
            ax = fig.add_subplot(gs[i, 0])
            
            valid_mask = df[col].notna()
            valid_time = df.loc[valid_mask, 'TIME__s'].values
            valid_data = df.loc[valid_mask, col].values
            
            phase_regions = []
            if phase_data is not None:
                _, phase_regions = add_phase_regions(ax, df['TIME__s'].values, phase_data, max_phase=max_phase)
            
            ax.plot(valid_time, valid_data, linewidth=1.5, color='black')
            ax.set_ylabel(labels[col])
            
            if phase_regions:
                add_phase_labels(ax, phase_regions, valid_time, valid_data)
            
            if len(valid_time) > 0 and len(valid_data) > 0:
                last_time = valid_time[-1]
                last_value = valid_data[-1]
                unit = extract_unit(col)
                
                ax.plot(last_time, last_value, 'ro', markersize=6, zorder=5)
                ax.annotate(f'{last_value:.2f} {unit}', 
                           xy=(last_time, last_value),
                           xytext=(-30, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5),
                           fontsize=9)
            
            if i == n_cols - 1:
                ax.set_xlabel('Time [s]')
                
            ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            
            def format_multiples_of_10(x, pos):
                if x % 10 == 0:
                    return f'{int(x)}'
                return ''
            ax.xaxis.set_major_formatter(FuncFormatter(format_multiples_of_10))
            
            ax.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
            ax.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
            
            ax.margins(x=0.01, y=0.1)
        
        fig.suptitle(f"{group_name}", fontweight='bold')
        plt.subplots_adjust(top=0.95)
        
        output_path = os.path.join(output_dir, f"{group_name.replace(' ', '_').lower()}.png")
        print(f"Saving plot: {output_path}")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        # plt.show()
        plt.close(fig)


def plot_rcs_moment_integral(df, output_dir='plots', reports_dir='reports', max_phase=4):
    try:
        
        time = df['TIME__s'].values
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        rcs_force_x = df['RCS_MOMENT_X__Nm'].values*2.0 / dr
        rcs_force_y = df['RCS_MOMENT_Y__Nm'].values / (length_rv - x_cg_rv)
        rcs_force_z = df['RCS_MOMENT_Z__Nm'].values / (length_rv - x_cg_rv)
        
        impulse_abs_force_x = cumulative_trapezoid(np.abs(rcs_force_x), time, initial=0)
        impulse_abs_force_y = cumulative_trapezoid(np.abs(rcs_force_y), time, initial=0)
        impulse_abs_force_z = cumulative_trapezoid(np.abs(rcs_force_z), time, initial=0)
        
        impulse_total_consumption = impulse_abs_force_x + impulse_abs_force_y + impulse_abs_force_z
        
        sign_changes_x = np.where(np.diff(np.sign(rcs_force_x)))[0]
        sign_changes_y = np.where(np.diff(np.sign(rcs_force_y)))[0]
        sign_changes_z = np.where(np.diff(np.sign(rcs_force_z)))[0]
        
        def calculate_pulse_durations(sign_changes, time_array):
            if len(sign_changes) > 1:
                pulse_durations = []
                for i in range(len(sign_changes) - 1):
                    start_idx = sign_changes[i]
                    end_idx = sign_changes[i + 1]
                    duration = time_array[end_idx] - time_array[start_idx]
                    pulse_durations.append(duration)
                
                min_pulse_duration = min(pulse_durations) if pulse_durations else 0
                min_pulse_idx = pulse_durations.index(min_pulse_duration) if pulse_durations else 0
                min_pulse_time = time_array[sign_changes[min_pulse_idx]] if sign_changes.size > 0 else 0
                return min_pulse_duration, min_pulse_time, pulse_durations
            else:
                return 0, 0, []
        
        min_pulse_duration_x, min_pulse_time_x, pulse_durations_x = calculate_pulse_durations(sign_changes_x, time)
        min_pulse_duration_y, min_pulse_time_y, pulse_durations_y = calculate_pulse_durations(sign_changes_y, time)
        min_pulse_duration_z, min_pulse_time_z, pulse_durations_z = calculate_pulse_durations(sign_changes_z, time)
        
        fig = plt.figure(figsize=(12, 2 + 1.5*4))
        gs = GridSpec(4, 1, figure=fig, hspace=0.4) 
        
        ax1 = fig.add_subplot(gs[0, 0])
        phase_regions = []
        if phase_data is not None:
            _, phase_regions = add_phase_regions(ax1, time, phase_data, max_phase=max_phase)
        ax1.plot(time, rcs_force_x, linewidth=1.5, color='red', label='X')
        ax1.set_ylabel('RCS Force X [N]')
        if phase_regions:
            add_phase_labels(ax1, phase_regions, time, rcs_force_x)
        ax1.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        def format_10(x, pos):
            return f'{int(x)}' if x % 10 == 0 else ''
        ax1.xaxis.set_major_formatter(FuncFormatter(format_10))
        ax1.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax1.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax1.margins(x=0.01, y=0.1)
        
        ax2 = fig.add_subplot(gs[1, 0])
        if phase_data is not None:
            _, phase_regions = add_phase_regions(ax2, time, phase_data, max_phase=max_phase)
        ax2.plot(time, rcs_force_y, linewidth=1.5, color='green', label='Y')
        ax2.set_ylabel('RCS Force Y [N]')
        if phase_regions:
            add_phase_labels(ax2, phase_regions, time, rcs_force_y)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_10))
        ax2.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax2.margins(x=0.01, y=0.1)
        
        ax3 = fig.add_subplot(gs[2, 0])
        if phase_data is not None:
            _, phase_regions = add_phase_regions(ax3, time, phase_data, max_phase=max_phase)
        ax3.plot(time, rcs_force_z, linewidth=1.5, color='blue', label='Z')
        ax3.set_ylabel('RCS Force Z [N]')
        if phase_regions:
            add_phase_labels(ax3, phase_regions, time, rcs_force_z)
        ax3.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax3.xaxis.set_minor_locator(MultipleLocator(1))
        ax3.xaxis.set_major_formatter(FuncFormatter(format_10))
        ax3.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax3.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax3.margins(x=0.01, y=0.1)
        
        ax5 = fig.add_subplot(gs[3, 0])
        if phase_data is not None:
            _, phase_regions = add_phase_regions(ax5, time, phase_data, max_phase=max_phase)
        ax5.plot(time, impulse_total_consumption, linewidth=1.5, color='black', label='Total Consumption')
        ax5.set_ylabel('Cumulative Impulse [N·s]')
        ax5.set_xlabel('Time [s]')
        if phase_regions:
            add_phase_labels(ax5, phase_regions, time, impulse_total_consumption)
        ax5.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax5.xaxis.set_minor_locator(MultipleLocator(1))
        ax5.xaxis.set_major_formatter(FuncFormatter(format_10))
        ax5.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax5.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax5.margins(x=0.01, y=0.1)
        
        fig.suptitle('RCS Force Analysis', fontweight='bold')
        plt.subplots_adjust(top=0.95) 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, 'rcs_force_analysis.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        report_content = []
        report_content.append("="*50)
        report_content.append("RCS FORCE ANALYSIS")
        report_content.append("="*50)
        
        report_content.append(f"\nTotal impulse (propellant consumption):")
        report_content.append(f"  Force X: {impulse_abs_force_x[-1]:.6f} N·s")
        report_content.append(f"  Force Y: {impulse_abs_force_y[-1]:.6f} N·s")
        report_content.append(f"  Force Z: {impulse_abs_force_z[-1]:.6f} N·s")
        report_content.append(f"  Total Consumption: {impulse_total_consumption[-1]:.6f} N·s")
        
        report_content.append(f"\nForce statistics:")
        report_content.append(f"  Max Force X: {np.max(np.abs(rcs_force_x)):.6f} N")
        report_content.append(f"  Max Force Y: {np.max(np.abs(rcs_force_y)):.6f} N")
        report_content.append(f"  Max Force Z: {np.max(np.abs(rcs_force_z)):.6f} N")
        
        report_content.append(f"\nPulse analysis:")
        report_content.append(f"  Force X - Sign changes: {len(sign_changes_x)}")
        if len(sign_changes_x) > 1:
            report_content.append(f"    Min pulse duration: {min_pulse_duration_x:.6f} s")
            report_content.append(f"    Min pulse time: {min_pulse_time_x:.6f} s")
            report_content.append(f"    Total pulses: {len(pulse_durations_x)}")
        else:
            report_content.append("    No pulses detected")
            
        report_content.append(f"  Force Y - Sign changes: {len(sign_changes_y)}")
        if len(sign_changes_y) > 1:
            report_content.append(f"    Min pulse duration: {min_pulse_duration_y:.6f} s")
            report_content.append(f"    Min pulse time: {min_pulse_time_y:.6f} s")
            report_content.append(f"    Total pulses: {len(pulse_durations_y)}")
        else:
            report_content.append("    No pulses detected")
            
        report_content.append(f"  Force Z - Sign changes: {len(sign_changes_z)}")
        if len(sign_changes_z) > 1:
            report_content.append(f"    Min pulse duration: {min_pulse_duration_z:.6f} s")
            report_content.append(f"    Min pulse time: {min_pulse_time_z:.6f} s")
            report_content.append(f"    Total pulses: {len(pulse_durations_z)}")
        else:
            report_content.append("    No pulses detected")
        
        report_content.append(f"\nSimulation time: {time[-1]:.2f} s")
        report_content.append("="*50)
        
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        report_path = os.path.join(reports_dir, 'rcs_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(report_content))
        
        print(f"RCS analysis report saved to: {report_path}")
        
    except FileNotFoundError:
        print("Error: File './records/simulation_data.csv' not found.")
        print("Check if the path is correct and the file exists.")
    except Exception as e:
        print(f"Error processing file: {e}")


def plot_control_performance(df, output_dir='plots', reports_dir='reports', max_phase=4):
    """
    Plot control performance comparing commanded vs measured accelerations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        
        time = df['TIME__s'].values
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        
        normal_cmd = df['NORMAL_ACC_CMD__m_s2'].values  # Normal (Z-axis)
        lateral_cmd = df['LATERAL_ACC_CMD__m_s2'].values  # Lateral (Y-axis)
        
        imu_y = df['IMU_SPECIFIC_FORCE_Y__m_s2'].values  # Lateral measured
        imu_z = df['IMU_SPECIFIC_FORCE_Z__m_s2'].values  # Normal measured
        
        valid_indices = ~(np.isnan(time) | np.isnan(imu_y) | np.isnan(imu_z) | np.isnan(normal_cmd) | np.isnan(lateral_cmd))
        
        time = time[valid_indices]
        imu_y = imu_y[valid_indices]
        imu_z = imu_z[valid_indices]
        normal_cmd = normal_cmd[valid_indices]
        lateral_cmd = lateral_cmd[valid_indices]
        
        time_start = 5.0
        cmd_indices = time >= time_start
        
        fig = plt.figure(figsize=(8, 2 + 1.5*2))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        phase_regions = []
        if phase_data is not None:
            _, phase_regions = add_phase_regions(ax1, time, phase_data, max_phase=max_phase)
        ax1.plot(time, imu_y, linewidth=1.5, color='red', label='Measured')
        ax1.plot(time[cmd_indices], lateral_cmd[cmd_indices], linewidth=1.5, color='blue', label='Commanded')
        ax1.set_ylabel('Lateral Acceleration [m/s²]')
        if phase_regions:
            add_phase_labels(ax1, phase_regions, time, imu_y)
        ax1.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        def format_10_ctrl(x, pos):
            return f'{int(x)}' if x % 10 == 0 else ''
        ax1.xaxis.set_major_formatter(FuncFormatter(format_10_ctrl))
        ax1.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax1.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax1.margins(x=0.01, y=0.1)
        
        legend = ax1.legend(frameon=True, fancybox=True, shadow=False,
                           facecolor='white', edgecolor='black', framealpha=1.0)
        legend.get_frame().set_linewidth(1.0)
        
        ax2 = fig.add_subplot(gs[1, 0])
        if phase_data is not None:
            _, phase_regions = add_phase_regions(ax2, time, phase_data, max_phase=max_phase)
        ax2.plot(time, imu_z, linewidth=1.5, color='red', label='Measured')
        ax2.plot(time[cmd_indices], -normal_cmd[cmd_indices], linewidth=1.5, color='blue', label='Commanded')
        ax2.set_ylabel('Normal Acceleration [m/s²]')
        ax2.set_xlabel('Time [s]')
        if phase_regions:
            add_phase_labels(ax2, phase_regions, time, imu_z)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.xaxis.set_major_formatter(FuncFormatter(format_10_ctrl))
        ax2.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax2.margins(x=0.01, y=0.1)
        
        legend = ax2.legend(frameon=True, fancybox=True, shadow=False,
                           facecolor='white', edgecolor='black', framealpha=1.0)
        legend.get_frame().set_linewidth(1.0)
        
        fig.suptitle('Control Performance Analysis', fontweight='bold')
        plt.subplots_adjust(top=0.95)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, 'control_performance.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        report_content = []
        report_content.append("="*50)
        report_content.append("CONTROL PERFORMANCE ANALYSIS")
        report_content.append("="*50)
        
        lateral_error = imu_y[cmd_indices] - lateral_cmd[cmd_indices]
        normal_error = imu_z[cmd_indices] - (-normal_cmd[cmd_indices])
        
        lateral_rms = np.sqrt(np.mean(lateral_error**2))
        normal_rms = np.sqrt(np.mean(normal_error**2))
        
        report_content.append(f"\nRMS Tracking Error:")
        report_content.append(f"  Lateral:  {lateral_rms:.4f} m/s²")
        report_content.append(f"  Normal:   {normal_rms:.4f} m/s²")
        
        lateral_max_error = np.max(np.abs(lateral_error))
        normal_max_error = np.max(np.abs(normal_error))
        
        report_content.append(f"\nMaximum Tracking Error:")
        report_content.append(f"  Lateral:  {lateral_max_error:.4f} m/s²")
        report_content.append(f"  Normal:   {normal_max_error:.4f} m/s²")
        
        lateral_corr = np.corrcoef(imu_y[cmd_indices], lateral_cmd[cmd_indices])[0, 1]
        normal_corr = np.corrcoef(imu_z[cmd_indices], -normal_cmd[cmd_indices])[0, 1]
        
        report_content.append(f"\nCorrelation Coefficient:")
        report_content.append(f"  Lateral:  {lateral_corr:.4f}")
        report_content.append(f"  Normal:   {normal_corr:.4f}")
        
        time_end = time[-1]
        cmd_time_end = time[cmd_indices][-1]
        report_content.append(f"\nIMU data: 0.0s to {time_end:.1f}s")
        report_content.append(f"Command data: {time_start:.1f}s to {cmd_time_end:.1f}s")
        report_content.append(f"Analysis window: {time_start:.1f}s to {cmd_time_end:.1f}s")
        report_content.append("="*50)
        
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        report_path = os.path.join(reports_dir, 'control_performance_report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(report_content))
        
        print(f"Control performance report saved to: {report_path}")
        
    except FileNotFoundError:
        print("Error: File 'records/simulation_data.csv' not found.")
        print("Check if the path is correct and the file exists.")
    except KeyError as e:
        print(f"Error: Required column not found in CSV: {e}")
        print("Check if the simulation data contains the required control columns.")
    except Exception as e:
        print(f"Error processing control performance data: {e}")


def plot_dynamic_pressure_analysis(df, output_dir='plots', reports_dir='reports', max_phase=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        
        time = df['TIME__s'].values
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        q_dyn = df['DYNAMIC_PRESSURE__Pa'].values
        altitude = df['ALTITUDE__m'].values
        
        valid_indices = ~(np.isnan(time) | np.isnan(q_dyn) | np.isnan(altitude))
        time = time[valid_indices]
        q_dyn = q_dyn[valid_indices]
        altitude = altitude[valid_indices]
        
        time_mask = time <= 100.0
        time = time[time_mask]
        q_dyn = q_dyn[time_mask]
        altitude = altitude[time_mask]
        
        q_dyn_kpa = q_dyn / 1000.0
        
        reference_kpa = 1.0
        intersections = []
        
        for i in range(len(q_dyn_kpa) - 1):
            if (q_dyn_kpa[i] - reference_kpa) * (q_dyn_kpa[i+1] - reference_kpa) < 0:
                t1, t2 = time[i], time[i+1]
                q1, q2 = q_dyn_kpa[i], q_dyn_kpa[i+1]
                alt1, alt2 = altitude[i], altitude[i+1]
                
                t_cross = t1 + (reference_kpa - q1) * (t2 - t1) / (q2 - q1)
                alt_cross = alt1 + (reference_kpa - q1) * (alt2 - alt1) / (q2 - q1)
                
                intersections.append((t_cross, reference_kpa, alt_cross))
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        phase_regions = []
        if phase_data is not None:
            phase_masked = phase_data[time_mask] if len(phase_data) >= len(time_mask) else None
            if phase_masked is not None:
                _, phase_regions = add_phase_regions(ax, time, phase_masked, max_phase=max_phase)
        
        ax.plot(time, q_dyn_kpa, linewidth=1.5, color='black', label='Dynamic Pressure')
        
        if phase_regions:
            add_phase_labels(ax, phase_regions, time, q_dyn_kpa)
        
        ax.axhline(y=reference_kpa, color='red', linestyle='--', linewidth=1.5, 
                   label=f'{reference_kpa} kPa Reference', alpha=0.7)
        
        for t_cross, q_cross, alt_cross in intersections:
            ax.plot(t_cross, q_cross, 'ro', markersize=8, zorder=5)
            ax.annotate(f'Alt: {alt_cross:.0f} m', 
                       xy=(t_cross, q_cross),
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5),
                       fontsize=9)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Dynamic Pressure [kPa]')
        ax.set_title('Dynamic Pressure Analysis', fontweight='bold')
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        
        def format_10_dyn(x, pos):
            return f'{int(x)}' if x % 10 == 0 else ''
        ax.xaxis.set_major_formatter(FuncFormatter(format_10_dyn))
        
        ax.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax.margins(x=0.01, y=0.1)
        
        ax.legend(loc='best')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, 'dynamic_pressure_analysis.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        report_content = []
        report_content.append("="*50)
        report_content.append("DYNAMIC PRESSURE ANALYSIS")
        report_content.append("="*50)
        report_content.append(f"\nIntersections with {reference_kpa} kPa reference line:")
        if intersections:
            for i, (t_cross, q_cross, alt_cross) in enumerate(intersections, 1):
                report_content.append(f"  {i}. Time: {t_cross:.2f} s, Altitude: {alt_cross:.0f} m")
        else:
            report_content.append("  No intersections found")
        report_content.append(f"\nMax dynamic pressure: {np.max(q_dyn_kpa):.3f} kPa at t={time[np.argmax(q_dyn_kpa)]:.2f} s")
        max_q_idx = np.argmax(q_dyn_kpa)
        report_content.append(f"Max Q altitude: {altitude[max_q_idx]:.0f} m")
        report_content.append("="*50)
        
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        report_path = os.path.join(reports_dir, 'dynamic_pressure_report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(report_content))
        
        print(f"Dynamic pressure report saved to: {report_path}")
        
    except FileNotFoundError:
        print("Error: File not found.")
        print("Check if the path is correct and the file exists.")
    except KeyError as e:
        print(f"Error: Required column not found in CSV: {e}")
        print("Check if the simulation data contains the required columns.")
    except Exception as e:
        print(f"Error processing dynamic pressure data: {e}")


def get_apogee_conditions(df):
    """
    Get flight conditions at apogee (maximum altitude).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing flight conditions at apogee:
        - time: Time at apogee [s]
        - altitude: Maximum altitude [m]
        - true_airspeed: True airspeed [m/s]
        - mach: Mach number [-]
        - flight_path_angle: Flight path angle [deg]
        - sideslip_angle: Sideslip angle (beta) [deg]
        - angle_of_attack: Angle of attack (alpha) [deg]
        - latitude: Latitude [deg]
        - longitude: Longitude [deg]
        - velocity_u: Body velocity U [m/s]
        - velocity_v: Body velocity V [m/s]
        - velocity_w: Body velocity W [m/s]
        - dynamic_pressure: Dynamic pressure [Pa]
        - mass: Vehicle mass [kg]
        - heading: Heading angle [deg]
    """
    
    apogee_idx = df['ALTITUDE__m'].idxmax()
    
    apogee_conditions = {
        'time': df.loc[apogee_idx, 'TIME__s'],
        'altitude': df.loc[apogee_idx, 'ALTITUDE__m'],
        'true_airspeed': df.loc[apogee_idx, 'TRUE_AIRSPEED__m_s'],
        'mach': df.loc[apogee_idx, 'MACH_NUMBER'],
        'flight_path_angle': df.loc[apogee_idx, 'FLIGHT_PATH_ANGLE__deg'],
        'sideslip_angle': df.loc[apogee_idx, 'SIDESLIP_ANGLE__deg'],
        'angle_of_attack': df.loc[apogee_idx, 'ANGLE_OF_ATTACK__deg'],
        'latitude': df.loc[apogee_idx, 'LATITUDE__deg'],
        'longitude': df.loc[apogee_idx, 'LONGITUDE__deg'],
        'velocity_u': df.loc[apogee_idx, 'STATE_BODY_VEL_U__m_s'],
        'velocity_v': df.loc[apogee_idx, 'STATE_BODY_VEL_V__m_s'],
        'velocity_w': df.loc[apogee_idx, 'STATE_BODY_VEL_W__m_s'],
        'dynamic_pressure': df.loc[apogee_idx, 'DYNAMIC_PRESSURE__Pa'],
        'mass': df.loc[apogee_idx, 'MASS__kg'],
        'heading': df.loc[apogee_idx, 'HEADING__deg'],
        'temperature': df.loc[apogee_idx, 'TEMPERATURE__K'],
        'stagnation_temperature': df.loc[apogee_idx, 'STAGNATION_TEMPERATURE__K'],
        'pressure': df.loc[apogee_idx, 'PRESSURE__Pa'],
        'density': df.loc[apogee_idx, 'DENSITY__kg_m3'],
    }
    
    return apogee_conditions


def save_apogee_report(df, output_dir='reports'):
    """
    Save a formatted report of flight conditions at apogee to a text file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_dir : str
        Output directory for reports
    """
    conditions = get_apogee_conditions(df)
    
    report_content = []
    report_content.append("="*60)
    report_content.append("FLIGHT CONDITIONS AT APOGEE")
    report_content.append("="*60)
    report_content.append(f"\nTime at Apogee:            {conditions['time']:.3f} s")
    report_content.append(f"\nAltitude (Max):            {conditions['altitude']:.2f} m")
    report_content.append(f"                           {conditions['altitude']/1000:.3f} km")
    report_content.append(f"\nVelocity:")
    report_content.append(f"  True Airspeed:           {conditions['true_airspeed']:.2f} m/s")
    report_content.append(f"  Mach Number:             {conditions['mach']:.4f}")
    report_content.append(f"\nAerodynamic Angles:")
    report_content.append(f"  Flight Path Angle:       {conditions['flight_path_angle']:.3f}°")
    report_content.append(f"  Angle of Attack (α):     {conditions['angle_of_attack']:.3f}°")
    report_content.append(f"  Sideslip Angle (β):      {conditions['sideslip_angle']:.3f}°")
    report_content.append(f"\nBody Velocities:")
    report_content.append(f"  U (axial):               {conditions['velocity_u']:.2f} m/s")
    report_content.append(f"  V (lateral):             {conditions['velocity_v']:.2f} m/s")
    report_content.append(f"  W (normal):              {conditions['velocity_w']:.2f} m/s")
    report_content.append(f"\nPosition:")
    report_content.append(f"  Latitude:                {conditions['latitude']:.6f}°")
    report_content.append(f"  Longitude:               {conditions['longitude']:.6f}°")
    report_content.append(f"  Heading:                 {conditions['heading']:.3f}°")
    report_content.append(f"\nAtmospheric Conditions:")
    report_content.append(f"  Temperature:             {conditions['temperature']:.2f} K")
    report_content.append(f"                           {conditions['temperature']-273.15:.2f} °C")
    report_content.append(f"  Pressure:                {conditions['pressure']:.2f} Pa")
    report_content.append(f"  Density:                 {conditions['density']:.6f} kg/m³")
    report_content.append(f"\nHeating:")
    report_content.append(f"  Stagnation Temperature:  {conditions['stagnation_temperature']:.2f} K")
    report_content.append(f"                           {conditions['stagnation_temperature']-273.15:.2f} °C")
    report_content.append(f"\nOther Parameters:")
    report_content.append(f"  Dynamic Pressure:        {conditions['dynamic_pressure']:.2f} Pa")
    report_content.append(f"  Vehicle Mass:            {conditions['mass']:.2f} kg")
    report_content.append("="*60)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, 'apogee_conditions_report.txt')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(report_content))
    
    print(f"Apogee conditions report saved to: {report_path}")


def plot_roll_phase_plane(df, output_dir='plots', max_phase=4):
    
    roll = df['ROLL_NED__deg'].values
    roll_rate = df['BODY_RATE_P__rad_s'].values * 180.0 / np.pi
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(roll, roll_rate, linewidth=1.5, color='#1F4E78')
    
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    ax.set_xlabel(r'$\theta$ [°]', loc='right', fontsize=12)
    ax.set_ylabel(r'$\dot{\theta}$ [°/s]', fontsize=12, rotation=0, ha='right', va='bottom')
    ax.yaxis.set_label_coords(0.3, 1.01)
    ax.xaxis.set_label_coords(1.01, 0.65)
    
    ax.tick_params(labelsize=11)
    
    ax.grid(False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'roll_phase_plane.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_altitude_with_phases(df, output_csv='altitude_phases.csv', output_dir='plots', records_dir='records', max_phase=4):
    """
    Generate altitude vs time plot with phase regions and export data to CSV.
    Includes all phases present in the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_csv : str
        Output CSV filename (saved in records_dir)
    output_dir : str
        Output directory for plots
    records_dir : str
        Output directory for CSV files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(records_dir):
        os.makedirs(records_dir)

    try:
        
        if 'PHASE' in df.columns:
            df = df[df['PHASE'] <= 3].copy()
        
        time = df['TIME__s'].values
        altitude = df['ALTITUDE__m'].values
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        
        output_data = df[['TIME__s', 'PHASE', 'ALTITUDE__m']].copy()
        output_path_csv = os.path.join(records_dir, output_csv)
        output_data.to_csv(output_path_csv, index=False, float_format='%.6f')
        print(f"Altitude data exported to: {output_path_csv}")
        print(f"  - Total points: {len(output_data)}")
        print(f"  - Time range: {time[0]:.2f} - {time[-1]:.2f} s")
        print(f"  - Altitude range: {altitude.min():.2f} - {altitude.max():.2f} m")
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        phase_regions = []
        if phase_data is not None:
            phases_found, phase_regions = add_phase_regions(ax, time, phase_data, max_phase=max_phase)
        
        ax.plot(time, altitude / 1000.0, linewidth=2.0, color='#1F4E78', label='Altitude')
        
        if phase_regions:
            add_phase_labels(ax, phase_regions, data_x=time, data_y=altitude / 1000.0)
        
        if len(time) > 0 and len(altitude) > 0:
            last_time = time[-1]
            last_altitude = altitude[-1] / 1000.0
            
            ax.plot(last_time, last_altitude, 'ro', markersize=6, zorder=5)
            ax.annotate(f'{last_altitude:.2f} km', 
                       xy=(last_time, last_altitude),
                       xytext=(-30, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5),
                       fontsize=9)
        
        # Configure axes
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Altitude [km]', fontsize=11)
        ax.set_title('Altitude vs Time', fontweight='bold', fontsize=12)
        
        # Grid configuration
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True, min_n_ticks=3))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        def format_time(x, pos):
            return f'{int(x)}'
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_time))
        ax.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax.margins(x=0.01, y=0.05)
        
        # Save figure
        output_path_fig = os.path.join(output_dir, 'altitude_with_phases.png')
        plt.savefig(output_path_fig, dpi=600, bbox_inches='tight')
        print(f"Altitude plot saved to: {output_path_fig}")
        plt.close(fig)
        
        print("\nAltitude Statistics:")
        print(f"  - Maximum altitude: {altitude.max():.2f} m ({altitude.max()/1000.0:.3f} km)")
        print(f"  - Time at max altitude: {time[altitude.argmax()]:.2f} s")
        if phase_data is not None:
            print(f"  - Phases included: {sorted(set(phase_data.astype(int)))}")
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except KeyError as e:
        print(f"Error: Missing required column {e}")
    except Exception as e:
        print(f"Error processing file: {e}")


def plot_altitude_vs_downrange(df, output_dir='plots', max_phase=4):
    """
    Plot altitude vs downrange with phase regions highlighted.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_dir : str
        Output directory for plots
    max_phase : int
        Maximum phase to plot (default: 4)
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        time = df['TIME__s'].values
        altitude = df['ALTITUDE__m'].values
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        
        lat = df['LATITUDE__deg'].values
        lon = df['LONGITUDE__deg'].values
        alt = df['ALTITUDE__m'].values
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        downrange = np.zeros(len(lat))
        for i in range(len(lat)):
            x, y, z = geodetic_to_ecef(lat_rad[i], lon_rad[i], alt[i])
            north, east, down = ecef_to_ned(x, y, z, lat_ref, lon_ref, alt_ref)
            downrange[i] = np.sqrt(north**2 + east**2)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        if phase_data is not None:
            phase_colors = {
                1: '#FFE5E5',
                2: '#E5F5FF',
                3: '#E5FFE5',
                4: '#FFF5E5'
            }
            
            current_phase = None
            start_idx = 0
            
            for i in range(len(phase_data)):
                if phase_data[i] is None or np.isnan(phase_data[i]):
                    continue
                
                phase_val = int(phase_data[i])
                
                if phase_val > max_phase:
                    continue
                
                if phase_val != current_phase:
                    if current_phase is not None and current_phase in phase_colors:
                        ax.fill_between(downrange[start_idx:i+1] / 1000.0, 
                                       0,
                                       altitude[start_idx:i+1] / 1000.0,
                                       facecolor=phase_colors[current_phase],
                                       alpha=0.7,
                                       zorder=0,
                                       edgecolor='none')
                    
                    current_phase = phase_val
                    start_idx = i
            
            # Fill the last segment
            if current_phase is not None and current_phase in phase_colors:
                ax.fill_between(downrange[start_idx:] / 1000.0, 
                               0,
                               altitude[start_idx:] / 1000.0,
                               facecolor=phase_colors[current_phase],
                               alpha=0.7,
                               zorder=0,
                               edgecolor='none')
        
        ax.plot(downrange / 1000.0, altitude / 1000.0, 
               linewidth=2.0, color='#1F4E78', label='Trajectory', zorder=2)
        
        apogee_idx = np.argmax(altitude)
        apogee_alt = altitude[apogee_idx] / 1000.0
        apogee_downrange = downrange[apogee_idx] / 1000.0
        
        ax.plot(apogee_downrange, apogee_alt, 'ro', markersize=8, zorder=5, label='Apogee')
        ax.annotate(f'Apogee\n{apogee_alt:.2f} km\n{apogee_downrange:.2f} km', 
                   xy=(apogee_downrange, apogee_alt),
                   xytext=(20, -40), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', lw=1.5),
                   fontsize=9)
        
        final_alt = altitude[-1] / 1000.0
        final_downrange = downrange[-1] / 1000.0
        
        ax.plot(final_downrange, final_alt, 'go', markersize=8, zorder=5, label='Final Point')
        ax.annotate(f'Final\n{final_alt:.2f} km\n{final_downrange:.2f} km', 
                   xy=(final_downrange, final_alt),
                   xytext=(-60, -30), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', lw=1.5),
                   fontsize=9)
        
        ax.set_xlabel('Downrange [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Altitude [km]', fontsize=12, fontweight='bold')
        ax.set_title('Altitude vs Downrange', fontweight='bold', fontsize=14)
        
        ax.set_aspect('equal', adjustable='box')
        
        ax.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.grid(True, which='minor', linestyle='-', alpha=0.2, linewidth=0.4)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        
        ax.margins(x=0.05, y=0.05)
        
        output_path = os.path.join(output_dir, 'altitude_vs_downrange.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Altitude vs Downrange plot saved to: {output_path}")
        print(f"  - Max altitude: {altitude.max()/1000:.3f} km at downrange {downrange[apogee_idx]/1000:.3f} km")
        print(f"  - Final downrange: {downrange[-1]/1000:.3f} km at altitude {altitude[-1]/1000:.3f} km")
        
    except KeyError as e:
        print(f"Error: Missing required column {e}")
    except Exception as e:
        print(f"Error generating altitude vs downrange plot: {e}")


def generate_mission_summary_report(df, output_dir='reports'):
    """
    Generate comprehensive mission summary report with key flight parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_dir : str
        Output directory for reports
    """
    try:
        
        time = df['TIME__s'].values
        altitude = df['ALTITUDE__m'].values
        velocity_u = df['STATE_BODY_VEL_U__m_s'].values
        mach = df['MACH_NUMBER'].values
        mass = df['MASS__kg'].values
        thrust = df['THRUST__N'].values if 'THRUST__N' in df.columns else None
        q_dyn = df['DYNAMIC_PRESSURE__Pa'].values
        
        report_content = []
        report_content.append("="*70)
        report_content.append("MISSION SUMMARY REPORT")
        report_content.append("="*70)
        
        # Flight duration
        report_content.append(f"\n1. FLIGHT DURATION")
        report_content.append(f"   Total simulation time: {time[-1]:.2f} s ({time[-1]/60:.2f} min)")
        
        # Altitude performance
        max_alt_idx = np.argmax(altitude)
        report_content.append(f"\n2. ALTITUDE PERFORMANCE")
        report_content.append(f"   Maximum altitude: {altitude[max_alt_idx]:.2f} m ({altitude[max_alt_idx]/1000:.3f} km)")
        report_content.append(f"   Time to apogee: {time[max_alt_idx]:.2f} s")
        
        # Velocity and Mach
        max_vel_idx = np.argmax(velocity_u)
        max_mach_idx = np.argmax(mach)
        report_content.append(f"\n3. VELOCITY AND MACH NUMBER")
        report_content.append(f"   Maximum velocity: {velocity_u[max_vel_idx]:.2f} m/s at t={time[max_vel_idx]:.2f} s")
        report_content.append(f"   Maximum Mach: {mach[max_mach_idx]:.4f} at t={time[max_mach_idx]:.2f} s")
        
        # Dynamic pressure
        max_q_idx = np.argmax(q_dyn)
        report_content.append(f"\n4. DYNAMIC PRESSURE")
        report_content.append(f"   Max Q: {q_dyn[max_q_idx]/1000:.3f} kPa at t={time[max_q_idx]:.2f} s")
        report_content.append(f"   Altitude at Max Q: {altitude[max_q_idx]:.0f} m")
        
        # Mass and propulsion
        report_content.append(f"\n5. MASS AND PROPULSION")
        report_content.append(f"   Initial mass: {mass[0]:.2f} kg")
        report_content.append(f"   Final mass: {mass[-1]:.2f} kg")
        report_content.append(f"   Total mass consumed: {mass[0] - mass[-1]:.2f} kg")
        
        if thrust is not None:
            thrust_active = thrust > 1.0
            if np.any(thrust_active):
                burn_time = time[thrust_active][-1] - time[thrust_active][0]
                avg_thrust = np.mean(thrust[thrust_active])
                max_thrust_val = np.max(thrust)
                report_content.append(f"   Burn time: {burn_time:.2f} s")
                report_content.append(f"   Average thrust: {avg_thrust:.2f} N")
                report_content.append(f"   Maximum thrust: {max_thrust_val:.2f} N")
        
        # Phase information
        if 'PHASE' in df.columns:
            phases = df['PHASE'].values
            unique_phases = sorted(set([int(p) for p in phases if not np.isnan(p)]))
            report_content.append(f"\n6. FLIGHT PHASES")
            report_content.append(f"   Number of phases: {len(unique_phases)}")
            
            for phase in unique_phases:
                phase_mask = phases == phase
                phase_times = time[phase_mask]
                if len(phase_times) > 0:
                    phase_duration = phase_times[-1] - phase_times[0]
                    report_content.append(f"   Phase {phase}: {phase_times[0]:.2f} s to {phase_times[-1]:.2f} s (duration: {phase_duration:.2f} s)")
        
        report_content.append("\n" + "="*70)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, 'mission_summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(report_content))
        
        print(f"Mission summary report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error generating mission summary report: {e}")


def generate_critical_events_report(df, output_dir='reports'):
    """
    Generate report of critical flight events and extreme conditions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_dir : str
        Output directory for reports
    """
    try:
        
        time = df['TIME__s'].values
        
        report_content = []
        report_content.append("="*70)
        report_content.append("CRITICAL EVENTS AND EXTREME CONDITIONS REPORT")
        report_content.append("="*70)
        
        # Maximum accelerations
        if 'SPECIFIC_FORCE_X__m_s2' in df.columns:
            acc_x = df['SPECIFIC_FORCE_X__m_s2'].values
            acc_y = df['SPECIFIC_FORCE_Y__m_s2'].values
            acc_z = df['SPECIFIC_FORCE_Z__m_s2'].values
            
            acc_total = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            max_acc_idx = np.argmax(acc_total)
            
            report_content.append(f"\n1. MAXIMUM ACCELERATIONS")
            report_content.append(f"   Max total acceleration: {acc_total[max_acc_idx]:.2f} m/s² ({acc_total[max_acc_idx]/9.81:.2f} g)")
            report_content.append(f"   Time: {time[max_acc_idx]:.2f} s")
            report_content.append(f"   Components:")
            report_content.append(f"     Axial (X): {acc_x[max_acc_idx]:.2f} m/s²")
            report_content.append(f"     Lateral (Y): {acc_y[max_acc_idx]:.2f} m/s²")
            report_content.append(f"     Normal (Z): {acc_z[max_acc_idx]:.2f} m/s²")
        
        # Maximum angular rates
        if 'BODY_RATE_P__rad_s' in df.columns:
            rate_p = df['BODY_RATE_P__rad_s'].values * 180/np.pi
            rate_q = df['BODY_RATE_Q__rad_s'].values * 180/np.pi
            rate_r = df['BODY_RATE_R__rad_s'].values * 180/np.pi
            
            max_p_idx = np.argmax(np.abs(rate_p))
            max_q_idx = np.argmax(np.abs(rate_q))
            max_r_idx = np.argmax(np.abs(rate_r))
            
            report_content.append(f"\n2. MAXIMUM ANGULAR RATES")
            report_content.append(f"   Max roll rate (P): {rate_p[max_p_idx]:.2f} °/s at t={time[max_p_idx]:.2f} s")
            report_content.append(f"   Max pitch rate (Q): {rate_q[max_q_idx]:.2f} °/s at t={time[max_q_idx]:.2f} s")
            report_content.append(f"   Max yaw rate (R): {rate_r[max_r_idx]:.2f} °/s at t={time[max_r_idx]:.2f} s")
        
        # Maximum aerodynamic angles
        if 'ANGLE_OF_ATTACK__deg' in df.columns:
            aoa = df['ANGLE_OF_ATTACK__deg'].values
            sideslip = df['SIDESLIP_ANGLE__deg'].values
            
            max_aoa_idx = np.argmax(np.abs(aoa))
            max_beta_idx = np.argmax(np.abs(sideslip))
            
            report_content.append(f"\n3. MAXIMUM AERODYNAMIC ANGLES")
            report_content.append(f"   Max angle of attack: {aoa[max_aoa_idx]:.3f}° at t={time[max_aoa_idx]:.2f} s")
            report_content.append(f"   Max sideslip angle: {sideslip[max_beta_idx]:.3f}° at t={time[max_beta_idx]:.2f} s")
        
        # Maximum heating
        if 'STAGNATION_TEMPERATURE__K' in df.columns:
            stag_temp = df['STAGNATION_TEMPERATURE__K'].values
            max_temp_idx = np.argmax(stag_temp)
            
            report_content.append(f"\n4. MAXIMUM HEATING")
            report_content.append(f"   Max stagnation temperature: {stag_temp[max_temp_idx]:.2f} K ({stag_temp[max_temp_idx]-273.15:.2f} °C)")
            report_content.append(f"   Time: {time[max_temp_idx]:.2f} s")
        
        # Control deflections
        if 'TVC_DELTA_Q__deg' in df.columns:
            tvc_q = df['TVC_DELTA_Q__deg'].values
            tvc_r = df['TVC_DELTA_R__deg'].values
            
            max_tvc_q_idx = np.argmax(np.abs(tvc_q))
            max_tvc_r_idx = np.argmax(np.abs(tvc_r))
            
            report_content.append(f"\n5. MAXIMUM CONTROL DEFLECTIONS (TVC)")
            report_content.append(f"   Max pitch deflection: {tvc_q[max_tvc_q_idx]:.3f}° at t={time[max_tvc_q_idx]:.2f} s")
            report_content.append(f"   Max yaw deflection: {tvc_r[max_tvc_r_idx]:.3f}° at t={time[max_tvc_r_idx]:.2f} s")
        
        # Battery (if applicable)
        if 'SOC__percent' in df.columns:
            soc = df['SOC__percent'].values
            current = df['CURRENT__A'].values
            
            min_soc_idx = np.argmin(soc)
            max_current_idx = np.argmax(np.abs(current))
            
            report_content.append(f"\n6. BATTERY PERFORMANCE")
            report_content.append(f"   Initial SOC: {soc[0]:.2f}%")
            report_content.append(f"   Final SOC: {soc[-1]:.2f}%")
            report_content.append(f"   Minimum SOC: {soc[min_soc_idx]:.2f}% at t={time[min_soc_idx]:.2f} s")
            report_content.append(f"   Max current draw: {current[max_current_idx]:.2f} A at t={time[max_current_idx]:.2f} s")
        
        report_content.append("\n" + "="*70)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, 'critical_events_report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(report_content))
        
        print(f"Critical events report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error generating critical events report: {e}")


def generate_trajectory_phases_report(df, output_dir='reports'):
    """
    Generate detailed report for each trajectory phase.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_dir : str
        Output directory for reports
    """
    try:
        
        if 'PHASE' not in df.columns:
            print("Warning: PHASE column not found. Skipping trajectory phases report.")
            return
        
        time = df['TIME__s'].values
        altitude = df['ALTITUDE__m'].values
        phases = df['PHASE'].values
        mach = df['MACH_NUMBER'].values
        mass = df['MASS__kg'].values
        
        unique_phases = sorted(set([int(p) for p in phases if not np.isnan(p)]))
        
        report_content = []
        report_content.append("="*70)
        report_content.append("TRAJECTORY PHASES DETAILED REPORT")
        report_content.append("="*70)
        
        for phase in unique_phases:
            phase_mask = phases == phase
            phase_times = time[phase_mask]
            phase_alt = altitude[phase_mask]
            phase_mach = mach[phase_mask]
            phase_mass = mass[phase_mask]
            
            if len(phase_times) == 0:
                continue
            
            duration = phase_times[-1] - phase_times[0]
            alt_start = phase_alt[0]
            alt_end = phase_alt[-1]
            alt_gain = alt_end - alt_start
            
            report_content.append(f"\n{'='*70}")
            report_content.append(f"PHASE {phase}")
            report_content.append(f"{'='*70}")
            report_content.append(f"  Time range: {phase_times[0]:.2f} s to {phase_times[-1]:.2f} s")
            report_content.append(f"  Duration: {duration:.2f} s")
            report_content.append(f"\n  Altitude:")
            report_content.append(f"    Start: {alt_start:.2f} m ({alt_start/1000:.3f} km)")
            report_content.append(f"    End: {alt_end:.2f} m ({alt_end/1000:.3f} km)")
            report_content.append(f"    Gain: {alt_gain:.2f} m ({alt_gain/1000:.3f} km)")
            report_content.append(f"    Maximum in phase: {np.max(phase_alt):.2f} m")
            report_content.append(f"\n  Velocity:")
            report_content.append(f"    Start Mach: {phase_mach[0]:.4f}")
            report_content.append(f"    End Mach: {phase_mach[-1]:.4f}")
            report_content.append(f"    Maximum Mach: {np.max(phase_mach):.4f}")
            report_content.append(f"\n  Mass:")
            report_content.append(f"    Start: {phase_mass[0]:.2f} kg")
            report_content.append(f"    End: {phase_mass[-1]:.2f} kg")
            report_content.append(f"    Consumed: {phase_mass[0] - phase_mass[-1]:.2f} kg")
            
            # Phase-specific analysis
            if 'THRUST__N' in df.columns:
                phase_thrust = df['THRUST__N'].values[phase_mask]
                if np.any(phase_thrust > 1.0):
                    avg_thrust = np.mean(phase_thrust[phase_thrust > 1.0])
                    report_content.append(f"\n  Propulsion:")
                    report_content.append(f"    Average thrust: {avg_thrust:.2f} N")
                    report_content.append(f"    Maximum thrust: {np.max(phase_thrust):.2f} N")
            
            if 'DYNAMIC_PRESSURE__Pa' in df.columns:
                phase_q = df['DYNAMIC_PRESSURE__Pa'].values[phase_mask]
                report_content.append(f"\n  Aerodynamics:")
                report_content.append(f"    Max dynamic pressure: {np.max(phase_q)/1000:.3f} kPa")
        
        report_content.append("\n" + "="*70)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, 'trajectory_phases_report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(report_content))
        
        print(f"Trajectory phases report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error generating trajectory phases report: {e}")


def generate_trajectory_summary_table(df, output_file='trajectory_summary.csv', records_dir='records'):
    """
    Generate trajectory summary table with key flight parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_file : str
        Output CSV filename (saved in records_dir)
    records_dir : str
        Output directory for CSV files
    """
    try:
        
        if 'PHASE' in df.columns:
            df = df[df['PHASE'] <= 3].copy()
        
        time = df['TIME__s'].values
        phase = df['PHASE'].values if 'PHASE' in df.columns else None
        altitude = df['ALTITUDE__m'].values
        dynamic_pressure = df['DYNAMIC_PRESSURE__Pa'].values
        mach = df['MACH_NUMBER'].values
        
        lat = df['LATITUDE__deg'].values
        lon = df['LONGITUDE__deg'].values
        alt = df['ALTITUDE__m'].values
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        downrange = np.zeros(len(lat))
        for i in range(len(lat)):
            x, y, z = geodetic_to_ecef(lat_rad[i], lon_rad[i], alt[i])
            north, east, down = ecef_to_ned(x, y, z, lat_ref, lon_ref, alt_ref)
            downrange[i] = np.sqrt(north**2 + east**2)
        
        summary_df = pd.DataFrame({
            'TIME__s': time,
            'PHASE': phase,
            'ALTITUDE__m': altitude,
            'DOWNRANGE__m': downrange,
            'DYNAMIC_PRESSURE__Pa': dynamic_pressure,
            'MACH_NUMBER': mach
        })
        
        if not os.path.exists(records_dir):
            os.makedirs(records_dir)
        
        output_path = os.path.join(records_dir, output_file)
        summary_df.to_csv(output_path, index=False, float_format='%.6f')
        
        print(f"Trajectory summary table saved to: {output_path}")
        print(f"  - Total points: {len(summary_df)}")
        print(f"  - Time range: {time[0]:.2f} - {time[-1]:.2f} s")
        print(f"  - Max altitude: {altitude.max():.2f} m ({altitude.max()/1000:.3f} km)")
        print(f"  - Max downrange: {downrange.max():.2f} m ({downrange.max()/1000:.3f} km)")
        print(f"  - Max Mach: {mach.max():.4f}")
        print(f"  - Max dynamic pressure: {dynamic_pressure.max()/1000:.3f} kPa")
        
    except Exception as e:
        print(f"Error generating trajectory summary table: {e}")


def plot_navigation_position_analysis(df, output_dir='plots', max_phase=4):
    """
    Analyze navigation system position estimation performance.
    Compares true position (STATE_ECI_POS) with estimated position (NAV_POS_ECI).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        time = df['TIME__s'].values
        
        # True position (ECI)
        pos_true_x = df['STATE_ECI_POS_X__m'].values
        pos_true_y = df['STATE_ECI_POS_Y__m'].values
        pos_true_z = df['STATE_ECI_POS_Z__m'].values
        
        # Estimated position (ECI)
        pos_nav_x = df['NAV_POS_ECI_X__m'].values
        pos_nav_y = df['NAV_POS_ECI_Y__m'].values
        pos_nav_z = df['NAV_POS_ECI_Z__m'].values
        
        # Calculate position errors
        error_x = pos_nav_x - pos_true_x
        error_y = pos_nav_y - pos_true_y
        error_z = pos_nav_z - pos_true_z
        error_magnitude = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        # Get position variances
        var_x = df['NAV_VAR_POS_ECI_X__m2'].values
        var_y = df['NAV_VAR_POS_ECI_Y__m2'].values
        var_z = df['NAV_VAR_POS_ECI_Z__m2'].values
        std_x = np.sqrt(np.abs(var_x))
        std_y = np.sqrt(np.abs(var_y))
        std_z = np.sqrt(np.abs(var_z))
        
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Position components comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, pos_true_x/1000, label='True X', linewidth=2, color='#1976d2')
        ax1.plot(time, pos_nav_x/1000, label='Estimated X', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax1, time, phase_data, max_phase=max_phase)
        ax1.set_xlabel('Time [s]', fontweight='bold')
        ax1.set_ylabel('Position X [km]', fontweight='bold')
        ax1.set_title('Position X Component (ECI)', fontweight='bold', fontsize=11)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, pos_true_y/1000, label='True Y', linewidth=2, color='#1976d2')
        ax2.plot(time, pos_nav_y/1000, label='Estimated Y', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax2, time, phase_data, max_phase=max_phase)
        ax2.set_xlabel('Time [s]', fontweight='bold')
        ax2.set_ylabel('Position Y [km]', fontweight='bold')
        ax2.set_title('Position Y Component (ECI)', fontweight='bold', fontsize=11)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time, pos_true_z/1000, label='True Z', linewidth=2, color='#1976d2')
        ax3.plot(time, pos_nav_z/1000, label='Estimated Z', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax3, time, phase_data, max_phase=max_phase)
        ax3.set_xlabel('Time [s]', fontweight='bold')
        ax3.set_ylabel('Position Z [km]', fontweight='bold')
        ax3.set_title('Position Z Component (ECI)', fontweight='bold', fontsize=11)
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        # Position error magnitude
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time, error_magnitude, label='Position Error', linewidth=2, color='#d32f2f')
        add_phase_regions(ax4, time, phase_data, max_phase=max_phase)
        ax4.set_xlabel('Time [s]', fontweight='bold')
        ax4.set_ylabel('Error Magnitude [m]', fontweight='bold')
        ax4.set_title('Position Error Magnitude', fontweight='bold', fontsize=11)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # Position errors with uncertainty bounds
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, error_x, label='Error X', linewidth=2, color='#1976d2')
        ax5.fill_between(time, -3*std_x, 3*std_x, alpha=0.2, color='#1976d2', label='±3σ bounds')
        add_phase_regions(ax5, time, phase_data, max_phase=max_phase)
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax5.set_xlabel('Time [s]', fontweight='bold')
        ax5.set_ylabel('Position Error X [m]', fontweight='bold')
        ax5.set_title('Position Error X with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time, error_y, label='Error Y', linewidth=2, color='#388e3c')
        ax6.fill_between(time, -3*std_y, 3*std_y, alpha=0.2, color='#388e3c', label='±3σ bounds')
        add_phase_regions(ax6, time, phase_data, max_phase=max_phase)
        ax6.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax6.set_xlabel('Time [s]', fontweight='bold')
        ax6.set_ylabel('Position Error Y [m]', fontweight='bold')
        ax6.set_title('Position Error Y with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(time, error_z, label='Error Z', linewidth=2, color='#7b1fa2')
        ax7.fill_between(time, -3*std_z, 3*std_z, alpha=0.2, color='#7b1fa2', label='±3σ bounds')
        add_phase_regions(ax7, time, phase_data, max_phase=max_phase)
        ax7.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax7.set_xlabel('Time [s]', fontweight='bold')
        ax7.set_ylabel('Position Error Z [m]', fontweight='bold')
        ax7.set_title('Position Error Z with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax7.legend(loc='best', framealpha=0.9)
        ax7.grid(True, alpha=0.3)
        
        # Position error statistics
        ax8 = fig.add_subplot(gs[3, 1])
        error_stats = pd.DataFrame({
            'Component': ['X', 'Y', 'Z', 'Magnitude'],
            'Mean [m]': [np.mean(error_x), np.mean(error_y), np.mean(error_z), np.mean(error_magnitude)],
            'Std [m]': [np.std(error_x), np.std(error_y), np.std(error_z), np.std(error_magnitude)],
            'RMS [m]': [np.sqrt(np.mean(error_x**2)), np.sqrt(np.mean(error_y**2)), 
                       np.sqrt(np.mean(error_z**2)), np.sqrt(np.mean(error_magnitude**2))],
            'Max [m]': [np.max(np.abs(error_x)), np.max(np.abs(error_y)), 
                       np.max(np.abs(error_z)), np.max(error_magnitude)]
        })
        # Format values to 4 decimal places
        error_stats_formatted = error_stats.copy()
        for col in error_stats_formatted.columns:
            if col != 'Component':
                error_stats_formatted[col] = error_stats_formatted[col].apply(lambda x: f'{x:.4f}')
        ax8.axis('tight')
        ax8.axis('off')
        table = ax8.table(cellText=error_stats_formatted.values, colLabels=error_stats_formatted.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(error_stats.columns)):
            table[(0, i)].set_facecolor('#e3f2fd')
            table[(0, i)].set_text_props(weight='bold')
        ax8.set_title('Position Error Statistics', fontweight='bold', fontsize=11)
        
        fig.suptitle('Navigation System - Position Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(output_dir, 'navigation_position_analysis.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print("Navigation position analysis plot saved successfully.")
        
    except Exception as e:
        print(f"Error generating navigation position analysis: {str(e)}")


def plot_navigation_velocity_analysis(df, output_dir='plots', max_phase=4):
    """
    Analyze navigation system velocity estimation performance.
    Compares true velocity (converted to ECI) with estimated velocity (NAV_VEL_ECI).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        time = df['TIME__s'].values
        
        # True velocity in body frame (ECEF)
        vel_body_u = df['STATE_BODY_VEL_U__m_s'].values
        vel_body_v = df['STATE_BODY_VEL_V__m_s'].values
        vel_body_w = df['STATE_BODY_VEL_W__m_s'].values
        
        # True position and quaternion for conversion
        pos_eci_x = df['STATE_ECI_POS_X__m'].values
        pos_eci_y = df['STATE_ECI_POS_Y__m'].values
        pos_eci_z = df['STATE_ECI_POS_Z__m'].values
        
        quat_w = df['STATE_QUAT_W'].values
        quat_x = df['STATE_QUAT_X'].values
        quat_y = df['STATE_QUAT_Y'].values
        quat_z = df['STATE_QUAT_Z'].values
        
        # Convert true velocity from body frame to ECI
        vel_true_eci_x = np.zeros_like(time)
        vel_true_eci_y = np.zeros_like(time)
        vel_true_eci_z = np.zeros_like(time)
        
        wei = np.array([[0], [0], [we]])
        wei_sm = skew_matrix(wei)
        
        for i in range(len(time)):
            q = np.array([[quat_w[i]], [quat_x[i]], [quat_y[i]], [quat_z[i]]])
            LBI = eci_to_body_quaternion(q)
            LIB = np.transpose(LBI)
            
            uvw = np.array([[vel_body_u[i]], [vel_body_v[i]], [vel_body_w[i]]])
            pi = np.array([[pos_eci_x[i]], [pos_eci_y[i]], [pos_eci_z[i]]])
            
            ve = np.matmul(LIB, uvw)
            vi = np.add(ve, np.matmul(wei_sm, pi))
            
            vel_true_eci_x[i] = vi[0, 0]
            vel_true_eci_y[i] = vi[1, 0]
            vel_true_eci_z[i] = vi[2, 0]
        
        # Estimated velocity (ECI)
        vel_nav_x = df['NAV_VEL_ECI_X__m_s'].values
        vel_nav_y = df['NAV_VEL_ECI_Y__m_s'].values
        vel_nav_z = df['NAV_VEL_ECI_Z__m_s'].values
        
        # Calculate velocity errors
        error_vx = vel_nav_x - vel_true_eci_x
        error_vy = vel_nav_y - vel_true_eci_y
        error_vz = vel_nav_z - vel_true_eci_z
        error_v_magnitude = np.sqrt(error_vx**2 + error_vy**2 + error_vz**2)
        
        # Get velocity variances
        var_vx = df['NAV_VAR_VEL_ECI_X__m2_s2'].values
        var_vy = df['NAV_VAR_VEL_ECI_Y__m2_s2'].values
        var_vz = df['NAV_VAR_VEL_ECI_Z__m2_s2'].values
        std_vx = np.sqrt(np.abs(var_vx))
        std_vy = np.sqrt(np.abs(var_vy))
        std_vz = np.sqrt(np.abs(var_vz))
        
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Velocity components comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, vel_true_eci_x, label='True Vx', linewidth=2, color='#1976d2')
        ax1.plot(time, vel_nav_x, label='Estimated Vx', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax1, time, phase_data, max_phase=max_phase)
        ax1.set_xlabel('Time [s]', fontweight='bold')
        ax1.set_ylabel('Velocity X [m/s]', fontweight='bold')
        ax1.set_title('Velocity X Component (ECI)', fontweight='bold', fontsize=11)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, vel_true_eci_y, label='True Vy', linewidth=2, color='#1976d2')
        ax2.plot(time, vel_nav_y, label='Estimated Vy', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax2, time, phase_data, max_phase=max_phase)
        ax2.set_xlabel('Time [s]', fontweight='bold')
        ax2.set_ylabel('Velocity Y [m/s]', fontweight='bold')
        ax2.set_title('Velocity Y Component (ECI)', fontweight='bold', fontsize=11)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time, vel_true_eci_z, label='True Vz', linewidth=2, color='#1976d2')
        ax3.plot(time, vel_nav_z, label='Estimated Vz', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax3, time, phase_data, max_phase=max_phase)
        ax3.set_xlabel('Time [s]', fontweight='bold')
        ax3.set_ylabel('Velocity Z [m/s]', fontweight='bold')
        ax3.set_title('Velocity Z Component (ECI)', fontweight='bold', fontsize=11)
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        # Velocity error magnitude
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time, error_v_magnitude, label='Velocity Error', linewidth=2, color='#d32f2f')
        add_phase_regions(ax4, time, phase_data, max_phase=max_phase)
        ax4.set_xlabel('Time [s]', fontweight='bold')
        ax4.set_ylabel('Error Magnitude [m/s]', fontweight='bold')
        ax4.set_title('Velocity Error Magnitude', fontweight='bold', fontsize=11)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # Velocity errors with uncertainty bounds
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, error_vx, label='Error Vx', linewidth=2, color='#1976d2')
        ax5.fill_between(time, -3*std_vx, 3*std_vx, alpha=0.2, color='#1976d2', label='±3σ bounds')
        add_phase_regions(ax5, time, phase_data, max_phase=max_phase)
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax5.set_xlabel('Time [s]', fontweight='bold')
        ax5.set_ylabel('Velocity Error X [m/s]', fontweight='bold')
        ax5.set_title('Velocity Error X with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time, error_vy, label='Error Vy', linewidth=2, color='#388e3c')
        ax6.fill_between(time, -3*std_vy, 3*std_vy, alpha=0.2, color='#388e3c', label='±3σ bounds')
        add_phase_regions(ax6, time, phase_data, max_phase=max_phase)
        ax6.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax6.set_xlabel('Time [s]', fontweight='bold')
        ax6.set_ylabel('Velocity Error Y [m/s]', fontweight='bold')
        ax6.set_title('Velocity Error Y with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(time, error_vz, label='Error Vz', linewidth=2, color='#7b1fa2')
        ax7.fill_between(time, -3*std_vz, 3*std_vz, alpha=0.2, color='#7b1fa2', label='±3σ bounds')
        add_phase_regions(ax7, time, phase_data, max_phase=max_phase)
        ax7.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax7.set_xlabel('Time [s]', fontweight='bold')
        ax7.set_ylabel('Velocity Error Z [m/s]', fontweight='bold')
        ax7.set_title('Velocity Error Z with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax7.legend(loc='best', framealpha=0.9)
        ax7.grid(True, alpha=0.3)
        
        # Velocity error statistics
        ax8 = fig.add_subplot(gs[3, 1])
        error_stats = pd.DataFrame({
            'Component': ['Vx', 'Vy', 'Vz', 'Magnitude'],
            'Mean [m/s]': [np.mean(error_vx), np.mean(error_vy), np.mean(error_vz), np.mean(error_v_magnitude)],
            'Std [m/s]': [np.std(error_vx), np.std(error_vy), np.std(error_vz), np.std(error_v_magnitude)],
            'RMS [m/s]': [np.sqrt(np.mean(error_vx**2)), np.sqrt(np.mean(error_vy**2)), 
                         np.sqrt(np.mean(error_vz**2)), np.sqrt(np.mean(error_v_magnitude**2))],
            'Max [m/s]': [np.max(np.abs(error_vx)), np.max(np.abs(error_vy)), 
                         np.max(np.abs(error_vz)), np.max(error_v_magnitude)]
        })
        # Format values to 4 decimal places
        error_stats_formatted = error_stats.copy()
        for col in error_stats_formatted.columns:
            if col != 'Component':
                error_stats_formatted[col] = error_stats_formatted[col].apply(lambda x: f'{x:.4f}')
        ax8.axis('tight')
        ax8.axis('off')
        table = ax8.table(cellText=error_stats_formatted.values, colLabels=error_stats_formatted.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(error_stats.columns)):
            table[(0, i)].set_facecolor('#e3f2fd')
            table[(0, i)].set_text_props(weight='bold')
        ax8.set_title('Velocity Error Statistics', fontweight='bold', fontsize=11)
        
        fig.suptitle('Navigation System - Velocity Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(output_dir, 'navigation_velocity_analysis.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print("Navigation velocity analysis plot saved successfully.")
        
    except Exception as e:
        print(f"Error generating navigation velocity analysis: {str(e)}")


def plot_navigation_attitude_analysis(df, output_dir='plots', max_phase=4):
    """
    Analyze navigation system attitude estimation performance.
    Compares true attitude (STATE_QUAT) with estimated attitude (NAV_QUAT) in Euler angles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        time = df['TIME__s'].values
        
        # True attitude quaternions
        quat_true_w = df['STATE_QUAT_W'].values
        quat_true_x = df['STATE_QUAT_X'].values
        quat_true_y = df['STATE_QUAT_Y'].values
        quat_true_z = df['STATE_QUAT_Z'].values
        
        # Estimated attitude quaternions
        quat_nav_w = df['NAV_QUAT_W'].values
        quat_nav_x = df['NAV_QUAT_X'].values
        quat_nav_y = df['NAV_QUAT_Y'].values
        quat_nav_z = df['NAV_QUAT_Z'].values
        
        # Convert quaternions to Euler angles
        roll_true = np.zeros_like(time)
        pitch_true = np.zeros_like(time)
        yaw_true = np.zeros_like(time)
        
        roll_nav = np.zeros_like(time)
        pitch_nav = np.zeros_like(time)
        yaw_nav = np.zeros_like(time)
        
        for i in range(len(time)):
            quat_true = np.array([[quat_true_w[i]], [quat_true_x[i]], [quat_true_y[i]], [quat_true_z[i]]])
            quat_nav_arr = np.array([[quat_nav_w[i]], [quat_nav_x[i]], [quat_nav_y[i]], [quat_nav_z[i]]])
            
            roll_true[i], pitch_true[i], yaw_true[i] = quaternion_to_euler(quat_true)
            roll_nav[i], pitch_nav[i], yaw_nav[i] = quaternion_to_euler(quat_nav_arr)
        
        # Convert to degrees
        roll_true_deg = np.rad2deg(roll_true)
        pitch_true_deg = np.rad2deg(pitch_true)
        yaw_true_deg = np.rad2deg(yaw_true)
        
        roll_nav_deg = np.rad2deg(roll_nav)
        pitch_nav_deg = np.rad2deg(pitch_nav)
        yaw_nav_deg = np.rad2deg(yaw_nav)
        
        # Calculate attitude errors
        error_roll = roll_nav_deg - roll_true_deg
        error_pitch = pitch_nav_deg - pitch_true_deg
        error_yaw = yaw_nav_deg - yaw_true_deg
        
        # Handle wrap-around for angles
        error_roll = np.where(error_roll > 180, error_roll - 360, error_roll)
        error_roll = np.where(error_roll < -180, error_roll + 360, error_roll)
        error_pitch = np.where(error_pitch > 180, error_pitch - 360, error_pitch)
        error_pitch = np.where(error_pitch < -180, error_pitch + 360, error_pitch)
        error_yaw = np.where(error_yaw > 180, error_yaw - 360, error_yaw)
        error_yaw = np.where(error_yaw < -180, error_yaw + 360, error_yaw)
        
        # Get attitude variances and convert to std in degrees
        var_quat_w = df['NAV_VAR_QUAT_W'].values
        var_quat_x = df['NAV_VAR_QUAT_X'].values
        var_quat_y = df['NAV_VAR_QUAT_Y'].values
        var_quat_z = df['NAV_VAR_QUAT_Z'].values
        
        # Approximate std in degrees (simplified conversion)
        std_roll = np.rad2deg(np.sqrt(np.abs(var_quat_x)))
        std_pitch = np.rad2deg(np.sqrt(np.abs(var_quat_y)))
        std_yaw = np.rad2deg(np.sqrt(np.abs(var_quat_z)))
        
        phase_data = df['PHASE'].values if 'PHASE' in df.columns else None
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Roll comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, roll_true_deg, label='True Roll', linewidth=2, color='#1976d2')
        ax1.plot(time, roll_nav_deg, label='Estimated Roll', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax1, time, phase_data)
        ax1.set_xlabel('Time [s]', fontweight='bold')
        ax1.set_ylabel('Roll [deg]', fontweight='bold')
        ax1.set_title('Roll Angle (ECI Frame)', fontweight='bold', fontsize=11)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Pitch comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, pitch_true_deg, label='True Pitch', linewidth=2, color='#1976d2')
        ax2.plot(time, pitch_nav_deg, label='Estimated Pitch', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax2, time, phase_data)
        ax2.set_xlabel('Time [s]', fontweight='bold')
        ax2.set_ylabel('Pitch [deg]', fontweight='bold')
        ax2.set_title('Pitch Angle (ECI Frame)', fontweight='bold', fontsize=11)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Yaw comparison
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time, yaw_true_deg, label='True Yaw', linewidth=2, color='#1976d2')
        ax3.plot(time, yaw_nav_deg, label='Estimated Yaw', linewidth=1.5, linestyle='-', color='#ff6f00')
        add_phase_regions(ax3, time, phase_data)
        ax3.set_xlabel('Time [s]', fontweight='bold')
        ax3.set_ylabel('Yaw [deg]', fontweight='bold')
        ax3.set_title('Yaw Angle (ECI Frame)', fontweight='bold', fontsize=11)
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        # Combined attitude errors
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time, error_roll, label='Roll Error', linewidth=2, color='#1976d2', alpha=0.7)
        ax4.plot(time, error_pitch, label='Pitch Error', linewidth=2, color='#388e3c', alpha=0.7)
        ax4.plot(time, error_yaw, label='Yaw Error', linewidth=2, color='#7b1fa2', alpha=0.7)
        add_phase_regions(ax4, time, phase_data)
        ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax4.set_xlabel('Time [s]', fontweight='bold')
        ax4.set_ylabel('Error [deg]', fontweight='bold')
        ax4.set_title('Attitude Errors', fontweight='bold', fontsize=11)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # Roll error with uncertainty bounds
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, error_roll, label='Roll Error', linewidth=2, color='#1976d2')
        ax5.fill_between(time, -3*std_roll, 3*std_roll, alpha=0.2, color='#1976d2', label='±3σ bounds')
        add_phase_regions(ax5, time, phase_data)
        ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax5.set_xlabel('Time [s]', fontweight='bold')
        ax5.set_ylabel('Roll Error [deg]', fontweight='bold')
        ax5.set_title('Roll Error with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        # Pitch error with uncertainty bounds
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time, error_pitch, label='Pitch Error', linewidth=2, color='#388e3c')
        ax6.fill_between(time, -3*std_pitch, 3*std_pitch, alpha=0.2, color='#388e3c', label='±3σ bounds')
        add_phase_regions(ax6, time, phase_data)
        ax6.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax6.set_xlabel('Time [s]', fontweight='bold')
        ax6.set_ylabel('Pitch Error [deg]', fontweight='bold')
        ax6.set_title('Pitch Error with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(True, alpha=0.3)
        
        # Yaw error with uncertainty bounds
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(time, error_yaw, label='Yaw Error', linewidth=2, color='#7b1fa2')
        ax7.fill_between(time, -3*std_yaw, 3*std_yaw, alpha=0.2, color='#7b1fa2', label='±3σ bounds')
        add_phase_regions(ax7, time, phase_data)
        ax7.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax7.set_xlabel('Time [s]', fontweight='bold')
        ax7.set_ylabel('Yaw Error [deg]', fontweight='bold')
        ax7.set_title('Yaw Error with Uncertainty Bounds', fontweight='bold', fontsize=11)
        ax7.legend(loc='best', framealpha=0.9)
        ax7.grid(True, alpha=0.3)
        
        # Attitude error statistics
        ax8 = fig.add_subplot(gs[3, 1])
        error_stats = pd.DataFrame({
            'Angle': ['Roll', 'Pitch', 'Yaw'],
            'Mean [deg]': [np.mean(error_roll), np.mean(error_pitch), np.mean(error_yaw)],
            'Std [deg]': [np.std(error_roll), np.std(error_pitch), np.std(error_yaw)],
            'RMS [deg]': [np.sqrt(np.mean(error_roll**2)), np.sqrt(np.mean(error_pitch**2)), 
                         np.sqrt(np.mean(error_yaw**2))],
            'Max [deg]': [np.max(np.abs(error_roll)), np.max(np.abs(error_pitch)), 
                         np.max(np.abs(error_yaw))]
        })
        # Format values to 4 decimal places
        error_stats_formatted = error_stats.copy()
        for col in error_stats_formatted.columns:
            if col != 'Angle':
                error_stats_formatted[col] = error_stats_formatted[col].apply(lambda x: f'{x:.4f}')
        ax8.axis('tight')
        ax8.axis('off')
        table = ax8.table(cellText=error_stats_formatted.values, colLabels=error_stats_formatted.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(error_stats.columns)):
            table[(0, i)].set_facecolor('#e3f2fd')
            table[(0, i)].set_text_props(weight='bold')
        ax8.set_title('Attitude Error Statistics', fontweight='bold', fontsize=11)
        
        fig.suptitle('Navigation System - Attitude Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(output_dir, 'navigation_attitude_analysis.png'), dpi=600, bbox_inches='tight')
        plt.close()
        print("Navigation attitude analysis plot saved successfully.")
        
    except Exception as e:
        print(f"Error generating navigation attitude analysis: {str(e)}")


def estimate_process_noise_covariance(df, output_dir='reports'):
    """
    Estimate the process noise covariance matrix Q from flight data.
    Analyzes the temporal growth of state estimation errors.
    
    The Q matrix represents the covariance of process noise in the EKF:
        dx/dt = f(x, u) + w,  where w ~ N(0, Q)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Simulation data DataFrame
    output_dir : str
        Output directory for reports
        
    Returns:
    --------
    dict
        Estimated Q matrix diagonal elements
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Extract time and sampling interval
        time = df['TIME__s'].values
        dt = np.mean(np.diff(time))
        
        # Extract true state
        q_true = np.column_stack([
            df['STATE_QUAT_W'].values,
            df['STATE_QUAT_X'].values,
            df['STATE_QUAT_Y'].values,
            df['STATE_QUAT_Z'].values
        ])
        
        pos_true = np.column_stack([
            df['STATE_ECI_POS_X__m'].values,
            df['STATE_ECI_POS_Y__m'].values,
            df['STATE_ECI_POS_Z__m'].values
        ])
        
        # Convert body velocity to ECI correctly (including Earth rotation effect)
        # Formula from vahsim.py: v_eci = LIB @ v_body + wei_sm @ position
        vel_true_eci = np.zeros((len(df), 3))
        wei = np.array([[0], [0], [we]])
        wei_sm = skew_matrix(wei)
        
        for i in range(len(df)):
            q_i = q_true[i:i+1].T
            LIB = body_to_eci_quaternion(q_i)
            vel_body = np.array([[df['STATE_BODY_VEL_U__m_s'].values[i]],
                                [df['STATE_BODY_VEL_V__m_s'].values[i]],
                                [df['STATE_BODY_VEL_W__m_s'].values[i]]])
            pos_i = pos_true[i:i+1].T
            
            # Correct transformation: v_eci = LIB @ v_body + wei_sm @ position
            vel_eci = np.add(np.matmul(LIB, vel_body), np.matmul(wei_sm, pos_i))
            vel_true_eci[i] = vel_eci.ravel()
        
        # Extract estimated state
        q_nav = np.column_stack([
            df['NAV_QUAT_W'].values,
            df['NAV_QUAT_X'].values,
            df['NAV_QUAT_Y'].values,
            df['NAV_QUAT_Z'].values
        ])
        
        pos_nav = np.column_stack([
            df['NAV_POS_ECI_X__m'].values,
            df['NAV_POS_ECI_Y__m'].values,
            df['NAV_POS_ECI_Z__m'].values
        ])
        
        vel_nav = np.column_stack([
            df['NAV_VEL_ECI_X__m_s'].values,
            df['NAV_VEL_ECI_Y__m_s'].values,
            df['NAV_VEL_ECI_Z__m_s'].values
        ])
        
        ba_nav = np.column_stack([
            df['NAV_ACC_BIAS_X__m_s2'].values,
            df['NAV_ACC_BIAS_Y__m_s2'].values,
            df['NAV_ACC_BIAS_Z__m_s2'].values
        ])
        
        bg_nav = np.column_stack([
            df['NAV_GYRO_BIAS_X__rad_s'].values,
            df['NAV_GYRO_BIAS_Y__rad_s'].values,
            df['NAV_GYRO_BIAS_Z__rad_s'].values
        ])
        
        # Calculate state errors
        q_error = q_nav - q_true
        pos_error = pos_nav - pos_true
        vel_error = vel_nav - vel_true_eci
        
        # Estimate Q from temporal growth of state errors
        # Q represents the covariance of the process noise: dx/dt = f(x) + w
        # We estimate Q from the variance of error derivatives: Q ≈ Var(de/dt)
        
        # Compute error derivatives
        dq_error_dt = np.diff(q_error, axis=0) / dt
        dpos_error_dt = np.diff(pos_error, axis=0) / dt
        dvel_error_dt = np.diff(vel_error, axis=0) / dt
        dba_error_dt = np.diff(ba_nav, axis=0) / dt
        dbg_error_dt = np.diff(bg_nav, axis=0) / dt
        
        # Estimate Q diagonal elements from variance of error growth
        # Quaternion process noise (4 elements)
        q_var = np.var(dq_error_dt, axis=0)
        
        # Position process noise (3 elements)
        pos_var = np.var(dpos_error_dt, axis=0)
        
        # Velocity process noise (3 elements)
        vel_var = np.var(dvel_error_dt, axis=0)
        
        # Accelerometer bias process noise (3 elements)
        ba_var = np.var(dba_error_dt, axis=0)
        
        # Gyroscope bias process noise (3 elements)
        bg_var = np.var(dbg_error_dt, axis=0)
        
        # Construct estimated Q diagonal
        Q_estimated = np.concatenate([
            q_var,      # Quaternion (4)
            pos_var,    # Position (3)
            vel_var,    # Velocity (3)
            ba_var,     # Accelerometer bias (3)
            bg_var      # Gyroscope bias (3)
        ])
        
        # Generate report
        report_content = []
        report_content.append("="*80)
        report_content.append("PROCESS NOISE COVARIANCE MATRIX Q ESTIMATION")
        report_content.append("="*80)
        report_content.append("\nThis analysis estimates the process noise covariance matrix Q used in the")
        report_content.append("Extended Kalman Filter (EKF) for the navigation system.")
        report_content.append("\nThe Q matrix represents the uncertainty in the process model:")
        report_content.append("  dx/dt = f(x, u) + w,  where w ~ N(0, Q)")
        report_content.append("\nQ accounts for:")
        report_content.append("  - IMU sensor noise (accelerometers and gyroscopes)")
        report_content.append("  - Modeling errors in the dynamics")
        report_content.append("  - Unmodeled disturbances")
        report_content.append("  - Numerical discretization errors")
        report_content.append("\nEstimation Method:")
        report_content.append("  Temporal Error Growth Analysis")
        report_content.append("    - Compute state errors: e(t) = x_nav(t) - x_true(t)")
        report_content.append("    - Compute error derivatives: de/dt ≈ [e(t+dt) - e(t)] / dt")
        report_content.append("    - Estimate Q as: Q ≈ Var(de/dt)")
        report_content.append(f"\n  Sampling interval: dt = {dt:.6f} s")
        report_content.append(f"  Data points analyzed: {len(df)}")
        report_content.append(f"  Flight duration: {time[-1]:.2f} s")
        
        # Check for GNSS updates
        gnss_pos_available = ~df['GNSS_POS_ECI_X__m'].isna()
        report_content.append(f"  GNSS updates: {gnss_pos_available.sum()} measurements")
        
        report_content.append("\n" + "-"*80)
        report_content.append("ESTIMATED Q MATRIX (Diagonal Elements)")
        report_content.append("-"*80)
        
        state_names = [
            'q0 (Quaternion W)', 'q1 (Quaternion X)', 'q2 (Quaternion Y)', 'q3 (Quaternion Z)',
            'px (Position X)', 'py (Position Y)', 'pz (Position Z)',
            'vx (Velocity X)', 'vy (Velocity Y)', 'vz (Velocity Z)',
            'bax (Acc Bias X)', 'bay (Acc Bias Y)', 'baz (Acc Bias Z)',
            'bgx (Gyro Bias X)', 'bgy (Gyro Bias Y)', 'bgz (Gyro Bias Z)'
        ]
        
        report_content.append("\n{:<30} {:>25}".format("State Component", "Estimated Q"))
        report_content.append("-"*80)
        
        for i, name in enumerate(state_names):
            report_content.append("{:<30} {:>25.8e}".format(name, Q_estimated[i]))
        
        # Statistics by component groups
        report_content.append("\n" + "-"*80)
        report_content.append("STATISTICS BY STATE COMPONENT GROUP")
        report_content.append("-"*80)
        
        groups = [
            ('Quaternion', 0, 4, ''),
            ('Position', 4, 7, '[m²]'),
            ('Velocity', 7, 10, '[m²/s²]'),
            ('Accelerometer Bias', 10, 13, '[m²/s⁴]'),
            ('Gyroscope Bias', 13, 16, '[rad²/s²]')
        ]
        
        for group_name, start_idx, end_idx, unit in groups:
            q_est_group = Q_estimated[start_idx:end_idx]
            
            report_content.append(f"\n{group_name} {unit}:")
            report_content.append(f"  Mean: {np.mean(q_est_group):.6e}")
            report_content.append(f"  Std:  {np.std(q_est_group):.6e}")
            report_content.append(f"  Min:  {np.min(q_est_group):.6e}")
            report_content.append(f"  Max:  {np.max(q_est_group):.6e}")
        
        # Error statistics
        report_content.append("\n" + "-"*80)
        report_content.append("STATE ERROR STATISTICS")
        report_content.append("-"*80)
        
        report_content.append("\nQuaternion Error:")
        report_content.append(f"  Mean: [{q_error.mean(axis=0)[0]:.6e}, {q_error.mean(axis=0)[1]:.6e}, {q_error.mean(axis=0)[2]:.6e}, {q_error.mean(axis=0)[3]:.6e}]")
        report_content.append(f"  Std:  [{q_error.std(axis=0)[0]:.6e}, {q_error.std(axis=0)[1]:.6e}, {q_error.std(axis=0)[2]:.6e}, {q_error.std(axis=0)[3]:.6e}]")
        
        report_content.append("\nPosition Error [m]:")
        report_content.append(f"  Mean: [{pos_error.mean(axis=0)[0]:.3f}, {pos_error.mean(axis=0)[1]:.3f}, {pos_error.mean(axis=0)[2]:.3f}]")
        report_content.append(f"  Std:  [{pos_error.std(axis=0)[0]:.3f}, {pos_error.std(axis=0)[1]:.3f}, {pos_error.std(axis=0)[2]:.3f}]")
        report_content.append(f"  RMS:  {np.sqrt(np.mean(pos_error**2)):.3f} m")
        
        report_content.append("\nVelocity Error [m/s]:")
        report_content.append(f"  Mean: [{vel_error.mean(axis=0)[0]:.3f}, {vel_error.mean(axis=0)[1]:.3f}, {vel_error.mean(axis=0)[2]:.3f}]")
        report_content.append(f"  Std:  [{vel_error.std(axis=0)[0]:.3f}, {vel_error.std(axis=0)[1]:.3f}, {vel_error.std(axis=0)[2]:.3f}]")
        report_content.append(f"  RMS:  {np.sqrt(np.mean(vel_error**2)):.3f} m/s")
        
        report_content.append("\nAccelerometer Bias [m/s²]:")
        report_content.append(f"  Mean: [{ba_nav.mean(axis=0)[0]:.6e}, {ba_nav.mean(axis=0)[1]:.6e}, {ba_nav.mean(axis=0)[2]:.6e}]")
        report_content.append(f"  Std:  [{ba_nav.std(axis=0)[0]:.6e}, {ba_nav.std(axis=0)[1]:.6e}, {ba_nav.std(axis=0)[2]:.6e}]")
        
        report_content.append("\nGyroscope Bias [rad/s]:")
        report_content.append(f"  Mean: [{bg_nav.mean(axis=0)[0]:.6e}, {bg_nav.mean(axis=0)[1]:.6e}, {bg_nav.mean(axis=0)[2]:.6e}]")
        report_content.append(f"  Std:  [{bg_nav.std(axis=0)[0]:.6e}, {bg_nav.std(axis=0)[1]:.6e}, {bg_nav.std(axis=0)[2]:.6e}]")
        
        # Python code for updated Q
        report_content.append("\n" + "="*80)
        report_content.append("PROPOSED Q MATRIX FOR navigation.py")
        report_content.append("="*80)
        report_content.append("\nself.Q = np.diag([")
        for i in range(0, 16, 4):
            values = ', '.join([f'{Q_estimated[j]:.8e}' for j in range(i, min(i+4, 16))])
            if i + 4 < 16:
                report_content.append(f"    {values},")
            else:
                report_content.append(f"    {values}])")
        
        report_content.append("\n" + "="*80)
        
        # Save report
        report_path = os.path.join(output_dir, 'process_noise_estimation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"Process noise estimation report saved to: {report_path}")
        
        # Return estimated Q
        return {
            'Q_estimated': Q_estimated,
            'state_names': state_names,
            'errors': {
                'quaternion': q_error,
                'position': pos_error,
                'velocity': vel_error,
                'acc_bias': ba_nav,
                'gyro_bias': bg_nav
            }
        }
        
    except Exception as e:
        print(f"Error estimating process noise covariance: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_navigation_performance_report(df, output_dir='reports'):
    """
    Generate a comprehensive text report of navigation system performance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        time = df['TIME__s'].values
        
        # Position errors
        pos_true_x = df['STATE_ECI_POS_X__m'].values
        pos_true_y = df['STATE_ECI_POS_Y__m'].values
        pos_true_z = df['STATE_ECI_POS_Z__m'].values
        pos_nav_x = df['NAV_POS_ECI_X__m'].values
        pos_nav_y = df['NAV_POS_ECI_Y__m'].values
        pos_nav_z = df['NAV_POS_ECI_Z__m'].values
        
        error_pos_x = pos_nav_x - pos_true_x
        error_pos_y = pos_nav_y - pos_true_y
        error_pos_z = pos_nav_z - pos_true_z
        error_pos_mag = np.sqrt(error_pos_x**2 + error_pos_y**2 + error_pos_z**2)
        
        # Velocity errors (need conversion)
        vel_body_u = df['STATE_BODY_VEL_U__m_s'].values
        vel_body_v = df['STATE_BODY_VEL_V__m_s'].values
        vel_body_w = df['STATE_BODY_VEL_W__m_s'].values
        
        quat_w = df['STATE_QUAT_W'].values
        quat_x = df['STATE_QUAT_X'].values
        quat_y = df['STATE_QUAT_Y'].values
        quat_z = df['STATE_QUAT_Z'].values
        
        vel_true_eci_x = np.zeros_like(time)
        vel_true_eci_y = np.zeros_like(time)
        vel_true_eci_z = np.zeros_like(time)
        
        wei = np.array([[0], [0], [we]])
        wei_sm = skew_matrix(wei)
        
        for i in range(len(time)):
            q = np.array([[quat_w[i]], [quat_x[i]], [quat_y[i]], [quat_z[i]]])
            LBI = eci_to_body_quaternion(q)
            LIB = np.transpose(LBI)
            
            uvw = np.array([[vel_body_u[i]], [vel_body_v[i]], [vel_body_w[i]]])
            pi = np.array([[pos_true_x[i]], [pos_true_y[i]], [pos_true_z[i]]])
            
            ve = np.matmul(LIB, uvw)
            vi = np.add(ve, np.matmul(wei_sm, pi))
            
            vel_true_eci_x[i] = vi[0, 0]
            vel_true_eci_y[i] = vi[1, 0]
            vel_true_eci_z[i] = vi[2, 0]
        
        vel_nav_x = df['NAV_VEL_ECI_X__m_s'].values
        vel_nav_y = df['NAV_VEL_ECI_Y__m_s'].values
        vel_nav_z = df['NAV_VEL_ECI_Z__m_s'].values
        
        error_vel_x = vel_nav_x - vel_true_eci_x
        error_vel_y = vel_nav_y - vel_true_eci_y
        error_vel_z = vel_nav_z - vel_true_eci_z
        error_vel_mag = np.sqrt(error_vel_x**2 + error_vel_y**2 + error_vel_z**2)
        
        # Attitude errors
        quat_nav_w = df['NAV_QUAT_W'].values
        quat_nav_x = df['NAV_QUAT_X'].values
        quat_nav_y = df['NAV_QUAT_Y'].values
        quat_nav_z = df['NAV_QUAT_Z'].values
        
        roll_true = np.zeros_like(time)
        pitch_true = np.zeros_like(time)
        yaw_true = np.zeros_like(time)
        roll_nav = np.zeros_like(time)
        pitch_nav = np.zeros_like(time)
        yaw_nav = np.zeros_like(time)
        
        for i in range(len(time)):
            quat_true = np.array([[quat_w[i]], [quat_x[i]], [quat_y[i]], [quat_z[i]]])
            quat_nav_arr = np.array([[quat_nav_w[i]], [quat_nav_x[i]], [quat_nav_y[i]], [quat_nav_z[i]]])
            
            roll_true[i], pitch_true[i], yaw_true[i] = quaternion_to_euler(quat_true)
            roll_nav[i], pitch_nav[i], yaw_nav[i] = quaternion_to_euler(quat_nav_arr)
        
        error_roll = np.rad2deg(roll_nav - roll_true)
        error_pitch = np.rad2deg(pitch_nav - pitch_true)
        error_yaw = np.rad2deg(yaw_nav - yaw_true)
        
        # Generate report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("NAVIGATION SYSTEM PERFORMANCE REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nSimulation Duration: {time[-1]:.2f} s")
        report_lines.append(f"Data Points: {len(time)}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("POSITION ESTIMATION ERRORS (ECI Frame)")
        report_lines.append("="*80)
        report_lines.append(f"\nPosition X:")
        report_lines.append(f"  Mean Error:      {np.mean(error_pos_x):>12.3f} m")
        report_lines.append(f"  Std Deviation:   {np.std(error_pos_x):>12.3f} m")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_pos_x**2)):>12.3f} m")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_pos_x)):>12.3f} m")
        
        report_lines.append(f"\nPosition Y:")
        report_lines.append(f"  Mean Error:      {np.mean(error_pos_y):>12.3f} m")
        report_lines.append(f"  Std Deviation:   {np.std(error_pos_y):>12.3f} m")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_pos_y**2)):>12.3f} m")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_pos_y)):>12.3f} m")
        
        report_lines.append(f"\nPosition Z:")
        report_lines.append(f"  Mean Error:      {np.mean(error_pos_z):>12.3f} m")
        report_lines.append(f"  Std Deviation:   {np.std(error_pos_z):>12.3f} m")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_pos_z**2)):>12.3f} m")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_pos_z)):>12.3f} m")
        
        report_lines.append(f"\nPosition Magnitude:")
        report_lines.append(f"  Mean Error:      {np.mean(error_pos_mag):>12.3f} m")
        report_lines.append(f"  Std Deviation:   {np.std(error_pos_mag):>12.3f} m")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_pos_mag**2)):>12.3f} m")
        report_lines.append(f"  Max Error:       {np.max(error_pos_mag):>12.3f} m")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("VELOCITY ESTIMATION ERRORS (ECI Frame)")
        report_lines.append("="*80)
        report_lines.append(f"\nVelocity X:")
        report_lines.append(f"  Mean Error:      {np.mean(error_vel_x):>12.3f} m/s")
        report_lines.append(f"  Std Deviation:   {np.std(error_vel_x):>12.3f} m/s")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_vel_x**2)):>12.3f} m/s")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_vel_x)):>12.3f} m/s")
        
        report_lines.append(f"\nVelocity Y:")
        report_lines.append(f"  Mean Error:      {np.mean(error_vel_y):>12.3f} m/s")
        report_lines.append(f"  Std Deviation:   {np.std(error_vel_y):>12.3f} m/s")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_vel_y**2)):>12.3f} m/s")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_vel_y)):>12.3f} m/s")
        
        report_lines.append(f"\nVelocity Z:")
        report_lines.append(f"  Mean Error:      {np.mean(error_vel_z):>12.3f} m/s")
        report_lines.append(f"  Std Deviation:   {np.std(error_vel_z):>12.3f} m/s")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_vel_z**2)):>12.3f} m/s")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_vel_z)):>12.3f} m/s")
        
        report_lines.append(f"\nVelocity Magnitude:")
        report_lines.append(f"  Mean Error:      {np.mean(error_vel_mag):>12.3f} m/s")
        report_lines.append(f"  Std Deviation:   {np.std(error_vel_mag):>12.3f} m/s")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_vel_mag**2)):>12.3f} m/s")
        report_lines.append(f"  Max Error:       {np.max(error_vel_mag):>12.3f} m/s")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("ATTITUDE ESTIMATION ERRORS (Euler Angles - ECI Frame)")
        report_lines.append("="*80)
        report_lines.append(f"\nRoll:")
        report_lines.append(f"  Mean Error:      {np.mean(error_roll):>12.3f} deg")
        report_lines.append(f"  Std Deviation:   {np.std(error_roll):>12.3f} deg")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_roll**2)):>12.3f} deg")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_roll)):>12.3f} deg")
        
        report_lines.append(f"\nPitch:")
        report_lines.append(f"  Mean Error:      {np.mean(error_pitch):>12.3f} deg")
        report_lines.append(f"  Std Deviation:   {np.std(error_pitch):>12.3f} deg")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_pitch**2)):>12.3f} deg")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_pitch)):>12.3f} deg")
        
        report_lines.append(f"\nYaw:")
        report_lines.append(f"  Mean Error:      {np.mean(error_yaw):>12.3f} deg")
        report_lines.append(f"  Std Deviation:   {np.std(error_yaw):>12.3f} deg")
        report_lines.append(f"  RMS Error:       {np.sqrt(np.mean(error_yaw**2)):>12.3f} deg")
        report_lines.append(f"  Max Abs Error:   {np.max(np.abs(error_yaw)):>12.3f} deg")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        report_content = '\n'.join(report_lines)
        
        report_path = os.path.join(output_dir, 'navigation_performance_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Navigation performance report saved to {report_path}")
        
    except Exception as e:
        print(f"Error generating navigation performance report: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots and reports from simulation data')
    parser.add_argument('--input', type=str, default='records/simulation_data_0.csv',
                       help='Input CSV file path (default: records/simulation_data_0.csv)')
    parser.add_argument('--plots-dir', type=str, default='plots/simulation_0',
                       help='Output directory for plots (default: plots/simulation_0)')
    parser.add_argument('--reports-dir', type=str, default='reports/simulation_0',
                       help='Output directory for reports (default: reports/simulation_0)')
    parser.add_argument('--records-dir', type=str, default='records/simulation_0',
                       help='Output directory for CSV records (default: records/simulation_0)')
    parser.add_argument('--max-phase', type=int, default=4,
                       help='Maximum phase to plot (default: 4)')
    
    args = parser.parse_args()
    
    file_path = args.input
    plots_dir = args.plots_dir
    reports_dir = args.reports_dir
    records_dir = args.records_dir
    max_phase = args.max_phase
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Generating plots in {plots_dir}...")
    print("  -> plot_record...")
    plot_record(df, output_dir=plots_dir, max_phase=max_phase)
    print("  -> plot_rcs_moment_integral...")
    plot_rcs_moment_integral(df, output_dir=plots_dir, reports_dir=reports_dir, max_phase=max_phase)
    print("  -> plot_control_performance...")
    plot_control_performance(df, output_dir=plots_dir, reports_dir=reports_dir, max_phase=max_phase)
    print("  -> plot_dynamic_pressure_analysis...")
    plot_dynamic_pressure_analysis(df, output_dir=plots_dir, reports_dir=reports_dir, max_phase=max_phase)
    print("  -> plot_roll_phase_plane...")
    plot_roll_phase_plane(df, output_dir=plots_dir, max_phase=max_phase)
    print("  -> plot_altitude_with_phases...")
    plot_altitude_with_phases(df, output_dir=plots_dir, records_dir=records_dir, max_phase=max_phase)
    print("  -> plot_altitude_vs_downrange...")
    plot_altitude_vs_downrange(df, output_dir=plots_dir, max_phase=max_phase)
    
    print(f"\nGenerating reports in {reports_dir}...")
    print("  -> save_apogee_report...")
    save_apogee_report(df, output_dir=reports_dir)
    print("  -> generate_mission_summary_report...")
    generate_mission_summary_report(df, output_dir=reports_dir)
    print("  -> generate_critical_events_report...")
    generate_critical_events_report(df, output_dir=reports_dir)
    print("  -> generate_trajectory_phases_report...")
    generate_trajectory_phases_report(df, output_dir=reports_dir)
    print("  -> generate_trajectory_summary_table...")
    generate_trajectory_summary_table(df, records_dir=records_dir)
    
    # Navigation system analysis
    print(f"\nGenerating navigation system analysis...")
    print("  -> plot_navigation_position_analysis...")
    plot_navigation_position_analysis(df, output_dir=plots_dir, max_phase=max_phase)
    print("  -> plot_navigation_velocity_analysis...")
    plot_navigation_velocity_analysis(df, output_dir=plots_dir, max_phase=max_phase)
    print("  -> plot_navigation_attitude_analysis...")
    plot_navigation_attitude_analysis(df, output_dir=plots_dir, max_phase=max_phase)
    print("  -> generate_navigation_performance_report...")
    generate_navigation_performance_report(df, output_dir=reports_dir)
    print("  -> estimate_process_noise_covariance...")
    estimate_process_noise_covariance(df, output_dir=reports_dir)
    
    print(f"\n[OK] All plots saved to: {plots_dir}")
    print(f"[OK] All reports saved to: {reports_dir}")
    print(f"[OK] CSV records saved to: {records_dir}")
    print("\n[OK] Done!")


