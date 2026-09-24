import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vahsimulator.aerodynamics import ControlSurfacesSetInterpolator


def plot_coeffs(
        fig_title: str,
        columns: list[list[str]],
    ) -> Figure:

    nrows = len(columns)
    ncols = max(len(row) for row in columns)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        num=fig_title,
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )

    for i, row in enumerate(columns):
        for j, coeff in enumerate(row):
            ax = axes[i, j]

            ax.plot(time, results['total'][coeff], label='TOTAL')
            ax.plot(time, results['fin_1'][coeff], label='Fin 1')
            ax.plot(time, results['fin_2'][coeff], label='Fin 2')
            ax.plot(time, results['fin_3'][coeff], label='Fin 3')
            ax.plot(time, results['fin_4'][coeff], label='Fin 4')

            ax.set_title(coeff)
            ax.set_xlabel('Time (s)')
            ax.grid(True)
            ax.legend()

    # Hide unused subplot positions
    for i in range(nrows):
        for j in range(len(columns[i]), ncols):
            axes[i, j].set_visible(False)

    return fig


### Trajectory
rato_trajectory_df = pd.read_csv('./records/simulation_data_0.csv')
rato_trajectory_df = rato_trajectory_df[rato_trajectory_df['PHASE'] == 1]
time = rato_trajectory_df['TIME__s']

### Control surface aerodeck
aerodeck_df = pd.read_csv('./datasets/aerodecks/control_delta_bank_phase_01.csv')
interpolator_set = ControlSurfacesSetInterpolator(
    fin_keys=['fin_1', 'fin_2', 'fin_3', 'fin_4'],
    aerodeck_df=aerodeck_df,
)

### Interpolation
delta_p = 0.0 * np.ones(len(rato_trajectory_df))
delta_q = 5.0 * np.ones(len(rato_trajectory_df))
delta_r = 0.0 * np.ones(len(rato_trajectory_df))

d1_cmd = -delta_p + delta_q - delta_r
d2_cmd = -delta_p + delta_q + delta_r
d3_cmd = delta_p + delta_q - delta_r
d4_cmd = delta_p + delta_q + delta_r

points = rato_trajectory_df[
    [
        'ANGLE_OF_ATTACK__deg',
        'SIDESLIP_ANGLE__deg',
        'MACH_NUMBER',
    ]
].values.tolist()
points = np.ascontiguousarray(points, dtype=np.float64)

results = interpolator_set.evaluate(
    points=points,
    fin_commands={
        'fin_1': d1_cmd,
        'fin_2': d2_cmd,
        'fin_3': d3_cmd,
        'fin_4': d4_cmd,
    },
)

### Plot aerodynamic coefficients
plot_coeffs(
    fig_title='Force coefficients',
    columns=[['DCA', 'DCY', 'DCN']],
)

plot_coeffs(
    fig_title='Torque coefficients',
    columns=[['DCLL', 'DCM', 'DCLN']],
)

plt.show()
