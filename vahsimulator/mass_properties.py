from typing import NamedTuple

import numpy as np
import pandas as pd

# VAHSimulator library
from .interpolate import LinearInterpolator1D
from . import performance_decorator
from .vehicle_state import VehicleState


class MassPropertiesData(NamedTuple):
    mass: np.float64
    Ixx: np.float64
    Iyy: np.float64
    Izz: np.float64
    Ixy: np.float64
    Ixz: np.float64
    Iyz: np.float64
    Ixx_dot: np.float64
    Iyy_dot: np.float64
    Izz_dot: np.float64
    x_cg: np.float64
    y_cg: np.float64
    z_cg: np.float64
    mass_dot: np.float64



class MassProperties:
    
    def __init__(self, dt, config_data: dict, prop_weight_unc=0.0, inert_weight_unc=0.0, payload_weight_unc=0.0, cg_offset_unc=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed + 13)

        self.dt = dt

        # Definining the uncertainties
        self.delta_dry_mass_pc        = np.random.normal(0.0, inert_weight_unc / 100.0)
        self.delta_payload_mass_pc    = np.random.normal(0.0, payload_weight_unc / 100.0)
        self.delta_propellant_mass_pc = np.random.normal(0.0, prop_weight_unc / 100.0)
        self.delta_x_cg_position_pc   = np.random.normal(0.0, cg_offset_unc / 100.0)

        # Getting the mass properties by flight phase
        mass_properties_data = config_data['mass_properties']
        geometric_data = config_data['geometric_data']
        self.mass_properties_table = self._get_mass_properties_data(mass_properties_data, geometric_data)

        # Finding index for first flight phase
        idx = self.mass_properties_table['phase'].index(1)

        dry_mass = self.mass_properties_table['dry_mass_kg'][idx] * (1 + self.delta_dry_mass_pc)
        dry_x_cg = self.mass_properties_table['dry_x_cg_m'][idx] * (1 + self.delta_x_cg_position_pc)
        dry_y_cg = self.mass_properties_table['dry_y_cg_m'][idx] 
        dry_z_cg = self.mass_properties_table['dry_z_cg_m'][idx]
        dry_Ixx = self.mass_properties_table['dry_Ixx_kgm2'][idx]
        dry_Iyy = self.mass_properties_table['dry_Iyy_kgm2'][idx]
        dry_Izz = self.mass_properties_table['dry_Izz_kgm2'][idx]
        dry_Ixy = self.mass_properties_table['dry_Ixy_kgm2'][idx]
        dry_Ixz = self.mass_properties_table['dry_Ixz_kgm2'][idx]
        dry_Iyz = self.mass_properties_table['dry_Iyz_kgm2'][idx]
        
        propellant_data = self.mass_properties_table['propellant_mass_properties'][idx].evaluate([0.])
        propellant_mass = propellant_data[0, 0] * (1 + self.delta_propellant_mass_pc)

        # Initial Values
        # Mass
        self.mass = dry_mass + propellant_mass

        # Initial CG Position
        self.x_cg = (dry_mass * dry_x_cg + propellant_mass * propellant_data[0, 1]) / self.mass
        self.y_cg = (dry_mass * dry_y_cg + propellant_mass * propellant_data[0, 2]) / self.mass
        self.z_cg = (dry_mass * dry_z_cg + propellant_mass * propellant_data[0, 3]) / self.mass
        
        ## Local axes distance wrt to CG
        # Dry Mass
        dx_dry = dry_x_cg - self.x_cg
        dy_dry = dry_y_cg - self.y_cg
        dz_dry = dry_z_cg - self.z_cg

        # Propellant Mass
        dx_prop = propellant_data[0, 1] - self.x_cg
        dy_prop = propellant_data[0, 2] - self.y_cg
        dz_prop = propellant_data[0, 3] - self.z_cg
        
        # Initial Moments of Inertia
        self.Ixx = dry_Ixx + dry_mass * ((dy_dry ** 2) + (dz_dry ** 2)) + \
            propellant_data[0, 4] +  propellant_mass * ((dy_prop  ** 2) + (dz_prop ** 2))

        self.Iyy = dry_Iyy + dry_mass * ((dx_dry ** 2) + (dz_dry ** 2)) + \
            propellant_data[0, 5] +  propellant_mass * ((dx_prop  ** 2) + (dz_prop ** 2))
        
        self.Izz = dry_Izz + dry_mass * ((dx_dry ** 2) + (dy_dry ** 2)) + \
            propellant_data[0, 6] +  propellant_mass * ((dx_prop  ** 2) + (dy_prop ** 2))
        
        # Initial Products of inertia
        self.Ixy = dry_Ixy
        self.Ixz = dry_Ixz
        self.Iyz = dry_Iyz

        # Time derivatives
        self.Ixx_dot = 0.0
        self.Iyy_dot = 0.0
        self.Izz_dot = 0.0
        self.mass_dot = 0.0

    @performance_decorator.time_execution_stats
    def _initialize_propellant_interpolator(self, data, x_top_of_engine):
        if data['type'] == 'table':
            file_name = data['file_name']

            df = pd.read_csv(file_name)

            # Adding columns that are not present in the datafame
            KEYS = ['Ycg_m', 'Zcg_m', 'Ixy_kgm2', 'Ixz_kgm2', 'Iyz_kgm2']
            df = df.assign(**{k: 0.0 for k in KEYS if k not in df.columns})

            df = df[['Time_s', 'Mass_kg', 'Xcg_m', 'Ycg_m', 'Zcg_m', 'Ixx_kgm2', 'Iyy_kgm2', 'Izz_kgm2', 'Ixy_kgm2', 'Ixz_kgm2', 'Iyz_kgm2']]

        else:

            initial_data = data['initial_values']
            final_data = data['final_values']

            propellant_data = {
                "Time_s": [initial_data['time'], final_data['time']],
                "Mass_kg": [initial_data['mass'], final_data['mass']],
                "Xcg_m": [initial_data['x_cg'], final_data['x_cg']],
                "Ycg_m": [initial_data['y_cg'], final_data['y_cg']],
                "Zcg_m": [initial_data['z_cg'], final_data['z_cg']],
                "Ixx_kgm2": [initial_data['Ixx'], final_data['Ixx']],
                "Iyy_kgm2": [initial_data['Iyy'], final_data['Iyy']],
                "Izz_kgm2": [initial_data['Izz'], final_data['Izz']],
                "Ixy_kgm2": [initial_data['Ixy'], final_data['Ixy']],
                "Ixz_kgm2": [initial_data['Ixz'], final_data['Ixz']],
                "Iyz_kgm2": [initial_data['Iyz'], final_data['Iyz']],
            }

            df = pd.DataFrame(propellant_data)

        # Correcting XCG position due to engine position
        df['Xcg_m'] = df['Xcg_m'] +  x_top_of_engine

        return LinearInterpolator1D.from_df(df,'Time_s')

    @performance_decorator.time_execution_stats
    def _get_mass_properties_data(self, mass_properties_data, geometric_data):
        n_phases = len(mass_properties_data)

        mass_properties_table = {
            'phase': [],
            'dry_mass_kg': [],
            'dry_x_cg_m': [],
            'dry_y_cg_m': [],
            'dry_z_cg_m': [],
            'dry_Ixx_kgm2': [],
            'dry_Iyy_kgm2': [],
            'dry_Izz_kgm2': [],
            'dry_Ixy_kgm2': [],
            'dry_Ixz_kgm2': [],    
            'dry_Iyz_kgm2': [],
            'propellant_mass_properties': [],
        }

        for phase in range(n_phases):
            data = mass_properties_data[phase]
            data_geometric = geometric_data[phase]
            phase_id = data['phase_id']

            mass_properties_table['phase'].append(phase_id)
            mass_properties_table['dry_mass_kg'].append(data['zero_fuel_properties']['dry_mass'])
            mass_properties_table['dry_x_cg_m'].append(data['zero_fuel_properties']['x_cg'])
            mass_properties_table['dry_y_cg_m'].append(data['zero_fuel_properties']['y_cg'])
            mass_properties_table['dry_z_cg_m'].append(data['zero_fuel_properties']['z_cg'])
            mass_properties_table['dry_Ixx_kgm2'].append(data['zero_fuel_properties']['dry_Ixx'])
            mass_properties_table['dry_Iyy_kgm2'].append(data['zero_fuel_properties']['dry_Iyy'])
            mass_properties_table['dry_Izz_kgm2'].append(data['zero_fuel_properties']['dry_Izz'])
            mass_properties_table['dry_Ixy_kgm2'].append(data['zero_fuel_properties']['dry_Ixy'])
            mass_properties_table['dry_Ixz_kgm2'].append(data['zero_fuel_properties']['dry_Ixz'])
            mass_properties_table['dry_Iyz_kgm2'].append(data['zero_fuel_properties']['dry_Iyz'])

            x_top_of_engine = data_geometric['other_parameters']['x_top_of_engine']

            # Getting propellant data
            mass_properties_table['propellant_mass_properties'].append(
                self._initialize_propellant_interpolator(data['fuel_properties'], 
                                                         x_top_of_engine))
            
        return mass_properties_table


    @performance_decorator.time_execution_stats
    def evaluate(self, state: VehicleState, phase) -> MassPropertiesData:
        """
        Inertia properties of the propellant at time t.
        """

        mass_k_1 = self.mass
        Ixx_k_1 = self.Ixx
        Iyy_k_1 = self.Iyy
        Izz_k_1 = self.Izz

        # Finding index for first flight phase
        idx = self.mass_properties_table['phase'].index(phase)

        if phase != 4:
            dry_mass = self.mass_properties_table['dry_mass_kg'][idx] * (1 + self.delta_dry_mass_pc)
        else:
            dry_mass = self.mass_properties_table['dry_mass_kg'][idx] * (1 + self.delta_payload_mass_pc)

        dry_x_cg = self.mass_properties_table['dry_x_cg_m'][idx] * (1 + self.delta_x_cg_position_pc)
        dry_y_cg = self.mass_properties_table['dry_y_cg_m'][idx] 
        dry_z_cg = self.mass_properties_table['dry_z_cg_m'][idx]
        dry_Ixx = self.mass_properties_table['dry_Ixx_kgm2'][idx]
        dry_Iyy = self.mass_properties_table['dry_Iyy_kgm2'][idx]
        dry_Izz = self.mass_properties_table['dry_Izz_kgm2'][idx]
        dry_Ixy = self.mass_properties_table['dry_Ixy_kgm2'][idx]
        dry_Ixz = self.mass_properties_table['dry_Ixz_kgm2'][idx]
        dry_Iyz = self.mass_properties_table['dry_Iyz_kgm2'][idx]
        
        propellant_data = self.mass_properties_table['propellant_mass_properties'][idx].evaluate([state.interp_time])
        propellant_mass = max(0.0, propellant_data[0, 0] * (1 + self.delta_propellant_mass_pc))

                # Mass
        self.mass = dry_mass + propellant_mass

        # CG Position
        self.x_cg = (dry_mass * dry_x_cg + propellant_mass * propellant_data[0, 1]) / self.mass
        self.y_cg = (dry_mass * dry_y_cg + propellant_mass * propellant_data[0, 2]) / self.mass
        self.z_cg = (dry_mass * dry_z_cg + propellant_mass * propellant_data[0, 3]) / self.mass
        
        ## Local axes distance wrt to CG
        # Dry Mass
        dx_dry = dry_x_cg - self.x_cg
        dy_dry = dry_y_cg - self.y_cg
        dz_dry = dry_z_cg - self.z_cg

        # Propellant Mass
        dx_prop = propellant_data[0, 1] - self.x_cg
        dy_prop = propellant_data[0, 2] - self.y_cg
        dz_prop = propellant_data[0, 3] - self.z_cg
        
        # Moments of Inertia
        self.Ixx = dry_Ixx + dry_mass * ((dy_dry ** 2) + (dz_dry ** 2)) + \
            propellant_data[0, 4] +  propellant_mass * ((dy_prop  ** 2) + (dz_prop ** 2))

        self.Iyy = dry_Iyy + dry_mass * ((dx_dry ** 2) + (dz_dry ** 2)) + \
            propellant_data[0, 5] +  propellant_mass * ((dx_prop  ** 2) + (dz_prop ** 2))
        
        self.Izz = dry_Izz + dry_mass * ((dx_dry ** 2) + (dy_dry ** 2)) + \
            propellant_data[0, 6] +  propellant_mass * ((dx_prop  ** 2) + (dy_prop ** 2))
        
        # Products of inertia
        self.Ixy = dry_Ixy
        self.Ixz = dry_Ixz
        self.Iyz = dry_Iyz

        self.mass_dot = (self.mass - mass_k_1) / self.dt
        self.Ixx_dot = (self.Ixx - Ixx_k_1) / self.dt
        self.Iyy_dot = (self.Iyy - Iyy_k_1) / self.dt
        self.Izz_dot = (self.Izz - Izz_k_1) / self.dt
     

        mass_properties_data =  MassPropertiesData(
            self.mass,
            self.Ixx, self.Iyy, self.Izz,
            self.Ixy, self.Ixz, self.Iyz,
            self.Ixx_dot, self.Iyy_dot, self.Izz_dot,
            self.x_cg, self.y_cg, self.z_cg,
            self.mass_dot,
        )
        return mass_properties_data
