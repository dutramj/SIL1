import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vahsimulator.aerodynamics import AerodynamicCoreInterpolator


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

            ax.plot(time, results[coeff])

            ax.set_title(coeff)
            ax.set_xlabel('Time (s)')
            ax.grid(True)

    # Hide unused subplot positions
    for i in range(nrows):
        for j in range(len(columns[i]), ncols):
            axes[i, j].set_visible(False)

    return fig


### Trajectory
rato_trajectory_df = pd.read_csv('./records/simulation_data_0.csv')
rato_trajectory_df = rato_trajectory_df[rato_trajectory_df['PHASE'] == 1]
time = rato_trajectory_df['TIME__s']

points = rato_trajectory_df[
    [
        'ANGLE_OF_ATTACK__deg',
        'SIDESLIP_ANGLE__deg',
        'MACH_NUMBER',
        'ALTITUDE__m'
    ]
].values.tolist()
points = np.ascontiguousarray(points, dtype=np.float64)

### Core aerodeck
aerodeck_df = pd.read_csv('./datasets/aerodecks/core_bank_phase_01.csv')
interpolator = AerodynamicCoreInterpolator(aerodeck_df)
results = pd.DataFrame(interpolator.evaluate(points))

### Plot aerodynamic coefficients
plot_coeffs(
    fig_title='Base coefficients',
    columns=[
        ['CA', 'CY', 'CN'],
        ['CLL', 'CM', 'CLN'],
    ],
)

plot_coeffs(
    fig_title='CP/CG',
    columns=[
        ['X-C.P.', 'XCG'],
    ],
)

plot_coeffs(
    fig_title='α static derivatives',
    columns=[
        ['CNA', 'CMA'],
    ],
)

plot_coeffs(
    fig_title='β static derivatives',
    columns=[
        ['CYB', 'CLNB', 'CLLB'],
    ],
)

plot_coeffs(
    fig_title='q dynamic derivatives',
    columns=[
        ['CNQ', 'CMQ'],
    ],
)

plot_coeffs(
    fig_title='𝛼̇ dynamic derivatives',
    columns=[
        ['CNAD', 'CMAD'],
    ],
)

plt.show()
