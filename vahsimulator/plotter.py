import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import Qt

ms2_to_microg = 101971.62
rad_s_to_deg_h = 206264.8
ms2_to_g = 1 / 9.80665
rad_to_deg = 180 / np.pi

# =========================
# CONFIG
# =========================
data_type_config = {
    'longitude': {'title': 'Longitude','unit': '(°)','csv_key': 'LONGITUDE__deg','conv_function': lambda x: x},
    'latitude': {'title': 'Latitude','unit': '(°)','csv_key': 'LATITUDE__deg','conv_function': lambda x: x},
    'altitude': {'title': 'Altitude','unit': '(m)','csv_key': 'ALTITUDE__m','conv_function': lambda x: x},
    'vn': {'title': 'Vel U (Body)','unit': '(m/s)','csv_key': 'STATE_BODY_VEL_U__m_s','conv_function': lambda x: x},
    've': {'title': 'Vel V (Body)','unit': '(m/s)','csv_key': 'STATE_BODY_VEL_V__m_s','conv_function': lambda x: x},
    'vd': {'title': 'Vel W (Body)','unit': '(m/s)','csv_key': 'STATE_BODY_VEL_W__m_s','conv_function': lambda x: x},
    'nx': {'title': 'Load Factor X','unit': '(g)','csv_key': 'SPECIFIC_FORCE_X__m_s2','conv_function': lambda x: x* ms2_to_g},
    'ny': {'title': 'Load Factor Y','unit': '(g)','csv_key': 'SPECIFIC_FORCE_Y__m_s2','conv_function': lambda x: x* ms2_to_g},
    'nz': {'title': 'Load Factor Z','unit': '(g)','csv_key': 'SPECIFIC_FORCE_Z__m_s2','conv_function': lambda x: x* ms2_to_g},
    'roll': {'title': 'Roll','unit': '(°)','csv_key': 'ROLL_NED__deg','conv_function': lambda x: x},
    'pitch': {'title': 'Pitch','unit': '(°)','csv_key': 'PITCH_NED__deg','conv_function': lambda x: x},
    'yaw': {'title': 'Yaw','unit': '(°)','csv_key': 'YAW_NED__deg','conv_function': lambda x: x},
    'roll_eci': {'title': 'Roll','unit': '(°)','csv_key': 'ROLL_ECI__deg','conv_function': lambda x: x},
    'pitch_eci': {'title': 'Pitch','unit': '(°)','csv_key': 'PITCH_ECI__deg','conv_function': lambda x: x},
    'yaw_eci': {'title': 'Yaw','unit': '(°)','csv_key': 'YAW_ECI__deg','conv_function': lambda x: x},
    'p': {'title': 'P','unit': '(°/s)','csv_key': 'BODY_RATE_P__rad_s','conv_function': lambda x: x* rad_to_deg},
    'q': {'title': 'Q','unit': '(°/s)','csv_key': 'BODY_RATE_Q__rad_s','conv_function': lambda x: x* rad_to_deg},
    'r': {'title': 'R','unit': '(°/s)','csv_key': 'BODY_RATE_R__rad_s','conv_function': lambda x: x* rad_to_deg},
    'alpha': {'title': 'Angle of Attack','unit': '(°)','csv_key': 'ANGLE_OF_ATTACK__deg','conv_function': lambda x: x},
    'beta': {'title': 'Sideslip angle','unit': '(°)','csv_key': 'SIDESLIP_ANGLE__deg','conv_function': lambda x: x},
    'mach': {'title': 'Mach','unit': '(-)','csv_key': 'MACH_NUMBER', 'conv_function': lambda x: x,},
    'm_alpha': {'title': 'M_alpha','unit': '(1/s2)','csv_key': 'MA__1_s2', 'conv_function': lambda x: x,},
    'dynamic_pressure': {'title': 'Dynamic Pressure','unit': '(Pa)','csv_key': 'DYNAMIC_PRESSURE__Pa','conv_function': lambda x: x,},
    'static_margin': {'title': 'Static Margin','unit': '(cal)','csv_key': 'STATIC_MARGIN','conv_function': lambda x: x,},
    'gamma': {'title': 'Flight Path Angle','unit': '(°)','csv_key': 'FLIGHT_PATH_ANGLE__deg','conv_function': lambda x: x},
    'mass': {'title': 'Mass','unit': '(kg)','csv_key': 'MASS__kg','conv_function': lambda x: x,},
    'xcg': {'title': 'X_CG','unit': '(m)','csv_key': 'X_CG__m','conv_function': lambda x: x,},
    'Q_tvc': {'title': 'DELTA Q TVC CMD','unit': '(°)','csv_key': 'DELTA_Q_TVC_CMD__deg','conv_function': lambda x: x},
    'R_tvc': {'title': 'DELTA R TVC CMD','unit': '(°)','csv_key': 'DELTA_R_TVC_CMD__deg','conv_function': lambda x: x},
    'L_rcs': {'title': 'RCS MOMENT X','unit': '(Nm)','csv_key': 'RCS_MOMENT_X__Nm','conv_function': lambda x: x},
    'M_rcs': {'title': 'RCS MOMENT Y','unit': '(Nm)','csv_key': 'RCS_MOMENT_Y__Nm','conv_function': lambda x: x},
    'N_rcs': {'title': 'RCS MOMENT Z','unit': '(Nm)','csv_key': 'RCS_MOMENT_Z__Nm','conv_function': lambda x: x},
    'delta_p_control_surface': {'title': 'DELTA P CONTROL SURFACE CMD','unit': '(°)','csv_key': 'DELTA_P_CONTROL_SURFACE_CMD__deg','conv_function': lambda x: x},
    'delta_q_control_surface': {'title': 'DELTA Q CONTROL SURFACE CMD','unit': '(°)','csv_key': 'DELTA_Q_CONTROL_SURFACE_CMD__deg','conv_function': lambda x: x},
    'delta_r_control_surface': {'title': 'DELTA R CONTROL SURFACE CMD','unit': '(°)','csv_key': 'DELTA_R_CONTROL_SURFACE_CMD__deg','conv_function': lambda x: x},
    'phase_id': {'title': 'Phase ID','unit': '(-)','csv_key': 'PHASE','conv_function': lambda x: x},
    'gamma': {'title': 'Gamma','unit': '(°)','csv_key': 'FLIGHT_PATH_ANGLE__deg','conv_function': lambda x: x},
    'heading': {'title': 'Heading','unit': '(°)','csv_key': 'HEADING__deg','conv_function': lambda x: x},
    'downrange': {'title': 'Downrange','unit': '(m)','csv_key': 'DOWNRANGE__m','conv_function': lambda x: x},
    'crossrange': {'title': 'Crossrange','unit': '(m)','csv_key': 'CROSSRANGE__m','conv_function': lambda x: x},
    'displacement': {'title': 'Displacement','unit': '(m)','csv_key': 'DISPLACEMENT__m','conv_function': lambda x: x},
    'wind_x': {'title': 'Wind X (Body)','unit': '(m/s)','csv_key': 'WIND_BODY_X__m_s','conv_function': lambda x: x},
    'wind_y': {'title': 'Wind Y (Body)','unit': '(m/s)','csv_key': 'WIND_BODY_Y__m_s','conv_function': lambda x: x},
    'wind_z': {'title': 'Wind Z (Body)','unit': '(m/s)','csv_key': 'WIND_BODY_Z__m_s','conv_function': lambda x: x},
    'cs_delta_1': {'title': 'CONTROL SURFACE DELTA 1','unit': '(°)','csv_key': 'CONTROL_SURFACE_DELTA_1__deg','conv_function': lambda x: x},
    'cs_delta_2': {'title': 'CONTROL SURFACE DELTA 2','unit': '(°)','csv_key': 'CONTROL_SURFACE_DELTA_2__deg','conv_function': lambda x: x},
    'cs_delta_3': {'title': 'CONTROL SURFACE DELTA 3','unit': '(°)','csv_key': 'CONTROL_SURFACE_DELTA_3__deg','conv_function': lambda x: x},
    'cs_delta_4': {'title': 'CONTROL SURFACE DELTA 4','unit': '(°)','csv_key': 'CONTROL_SURFACE_DELTA_4__deg','conv_function': lambda x: x},
    'cs_type': {'title': 'CONTROL SURFACE CONFIG TYPE','unit': '(-)','csv_key': 'CONTROL_SURFACE_TYPE','conv_function': lambda x: x},
    'tas': {'title': 'TRUE AIRSPEED', 'unit': '(m/s)', 'csv_key': 'TRUE_AIRSPEED__m_s', 'conv_function': lambda x: x},
    'oat': {'title': 'Outside Air Temperature','unit': '(°C)','csv_key': 'TEMPERATURE__K','conv_function': lambda x: x - 273.15},
    'p_atm': {'title': 'Static Pressure','unit': '(Pa)','csv_key': 'PRESSURE__Pa','conv_function': lambda x: x},
    'rho': {'title': 'Air Density','unit': '(kg/m³)','csv_key': 'DENSITY__kg_m3','conv_function': lambda x: x},
    
}

# =========================
# GUI CORE
# =========================
_OPEN_WINDOWS = []

def _ensure_qapp():
    """Create the Qt application if necessary.

    Returns
    -------
    QApplication
        Active Qt application instance.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def _register_window(win):
    _OPEN_WINDOWS.append(win)
    QApplication.processEvents()

def _resize_and_center(win, frac=0.85):
    screen = QGuiApplication.primaryScreen()
    if not screen:
        return
    ag = screen.availableGeometry()
    w = int(ag.width() * frac)
    h = int(ag.height() * frac)
    x = ag.x() + (ag.width() - w) // 2
    y = ag.y() + (ag.height() - h) // 2
    win.resize(w, h)
    win.move(x, y)

class _StandardTab(QWidget):
    def __init__(self, fig_title, data_cfg_list, enable_compare=True):
        super().__init__()
        self.plots = {}
        self.diff_plots = {}

        vbox = QVBoxLayout(self)
        layout = pg.GraphicsLayoutWidget()
        vbox.addWidget(layout)

        for cfg_key in data_cfg_list:
            cfg = data_type_config[cfg_key]
            unit_txt = cfg['unit'].strip('()')

            p_main = layout.addPlot(title=f"{cfg['title']} vs Time")
            p_main.setLabel("bottom", "Time", units="s")
            p_main.setLabel("left", cfg['title'], units=unit_txt)
            p_main.showGrid(x=True, y=True)
            p_main.addLegend()
            self.plots[cfg_key] = p_main

            if enable_compare:
                p_diff = layout.addPlot(title=f"{cfg['title']} Difference")
                p_diff.setLabel("bottom", "Time", units="s")
                p_diff.setLabel("left", "Error", units=unit_txt)
                p_diff.showGrid(x=True, y=True)
                p_diff.addLegend()
                self.diff_plots[cfg_key] = p_diff

            layout.nextRow()

class _MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plots")
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs_dict = {}

    def get_tab(self, name, cfg_list=None, enable_compare=True):
        if name in self.tabs_dict:
            return self.tabs_dict[name]

        tab = _StandardTab(name, cfg_list, enable_compare)
        self.tabs.addTab(tab, name)
        self.tabs_dict[name] = tab
        return tab

_MAIN = None

def _get_main():
    global _MAIN
    _ensure_qapp()
    if _MAIN is None:
        _MAIN = _MainWindow()
        _resize_and_center(_MAIN)
        _MAIN.show()
        _register_window(_MAIN)
    return _MAIN

# =========================
# PLOT CORE
# =========================
def _get_keys(cfg):
    ref_key = cfg.get("ref_key", cfg["csv_key"])
    exp_key = cfg.get("exp_key", cfg["csv_key"])
    return ref_key, exp_key

def _plot(reference_df, experimental_df, name, keys):
    main = _get_main()
    has_exp = experimental_df is not None
    tab = main.get_tab(name, keys, enable_compare=has_exp)

    for k in keys:
        cfg = data_type_config[k]
        ref_key, exp_key = _get_keys(cfg)

        if ref_key not in reference_df.columns:
            continue

        t_ref = reference_df["TIME__s"]
        ref = cfg["conv_function"](reference_df[ref_key])

        tab.plots[k].plot(t_ref, ref, pen="b", name="Ref")

        if not has_exp:
            continue

        if exp_key not in experimental_df.columns:
            continue

        t_exp = experimental_df["TIME__s"]
        exp = cfg["conv_function"](experimental_df[exp_key])

        tab.plots[k].plot(
            t_exp,
            exp,
            pen=pg.mkPen("r", style=Qt.DashLine),
            name="Exp"
        )

        exp_interp = np.interp(t_ref, t_exp, exp)
        diff = ref - exp_interp

        tab.diff_plots[k].plot(t_ref, diff, pen="g", name="Error")

def _validate_dataframe(df, name):
    """Validate basic requirements for plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    name : str
        DataFrame name used in error messages.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame.
    ValueError
        If required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas.DataFrame, "
            f"got {type(df).__name__}."
        )

    if "TIME__s" not in df.columns:
        raise ValueError(
            f"{name} does not contain required column 'TIME__s'."
        )

    if df.empty:
        raise ValueError(f"{name} is empty.")

    time = df["TIME__s"]

    if not pd.api.types.is_numeric_dtype(time):
        raise TypeError(
            f"{name}['TIME__s'] must be numeric."
        )

    if not np.isfinite(time.to_numpy()).all():
        raise ValueError(
            f"{name}['TIME__s'] contains NaN or infinite values."
        )

# =========================
# API
# =========================
def plot_all(ref_df, exp_df=None):
    """Plot all available simulation data.

    Parameters
    ----------
    ref_df : pandas.DataFrame
        Reference simulation data.
    exp_df : pandas.DataFrame, optional
        Experimental simulation data used for comparison.
    """
    _validate_dataframe(ref_df, "ref_df")

    if exp_df is not None:
        _validate_dataframe(exp_df, "exp_df")


    app = _ensure_qapp()


    _plot(ref_df, exp_df, "Position", ["longitude", "latitude", "altitude"])
    _plot(ref_df, exp_df, "Linear Velocity", ["vn", "ve", "vd"])
    _plot(ref_df, exp_df, "Attitude ECI", ["roll_eci", "pitch_eci", "yaw_eci"])
    _plot(ref_df, exp_df, "Attitude", ["roll", "pitch", "yaw"])
    _plot(ref_df, exp_df, "Load Factor", ["nx", "ny", "nz"])
    _plot(ref_df, exp_df, "Angular Velocity", ["p", "q", "r"])
    _plot(ref_df, exp_df, "Aero Angles", ["alpha", "beta"])
    _plot(ref_df, exp_df, "Aero Properties", ["mach", "dynamic_pressure", "static_margin", "m_alpha"])
    _plot(ref_df, exp_df, "Atmos Properties", ["oat", "p_atm", "rho"])
    _plot(ref_df, exp_df, "Trajectory", ["mach", "altitude", "gamma"])
    _plot(ref_df, exp_df, "Range", ["downrange", "crossrange", "displacement"])
    _plot(ref_df, exp_df, "Mass Properties", ["mass", "xcg"])
    _plot(ref_df, exp_df, "RCS Moments", ["L_rcs", "M_rcs", "N_rcs"])
    _plot(ref_df, exp_df, "TVC Command", ["Q_tvc", "R_tvc"])
    _plot(ref_df, exp_df, "Control Surfaces Commands", ["delta_p_control_surface", "delta_q_control_surface", "delta_r_control_surface"])
    _plot(ref_df, exp_df, "Wind Speed", ["wind_x", "wind_y", "wind_z"])
    _plot(ref_df, exp_df, "Control Surfaces Deflections", ["cs_delta_1", "cs_delta_2", "cs_delta_3", "cs_delta_4"])
    _plot(ref_df, exp_df, "Phase", ["phase_id", "cs_type"])
    _plot(ref_df, exp_df, "Speed Comparison", ["vn", "wind_x", "tas"])

    main = _get_main()
    main.show()

    app.exec()
