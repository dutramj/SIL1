from __future__ import annotations

import math
from dataclasses import dataclass
import pandas as pd
import numpy as np

# ==========================================================
# CONSTANTES ISA
# ==========================================================

G0 = 9.80665          # m/s²
R = 287.05287         # J/(kg·K)
GAMMA = 1.4

T0 = 288.15           # K
P0 = 101325.0         # Pa

# ==========================================================
# CAMADAS ISA 1976 ATÉ 86 km (MESOPAUSA)
# ==========================================================
#
# h_base [m], lapse rate [K/m]
#
# 0-11 km      : -6.5 K/km
# 11-20 km     : 0
# 20-32 km     : +1.0 K/km
# 32-47 km     : +2.8 K/km
# 47-51 km     : 0
# 51-71 km     : -2.8 K/km
# 71-86 km     : -2.0 K/km
#
# ==========================================================

@dataclass
class Layer:
    h_base: float
    lapse: float


LAYERS = [
    Layer(0.0,      -0.0065),
    Layer(11000.0,   0.0),
    Layer(20000.0,   0.0010),
    Layer(32000.0,   0.0028),
    Layer(47000.0,   0.0),
    Layer(51000.0,  -0.0028),
    Layer(71000.0,  -0.0020),
]

TOP_ATMOSPHERE = 86000.0


# ==========================================================
# CÁLCULO DAS CONDIÇÕES DE BASE DE CADA CAMADA
# ==========================================================

def build_base_conditions(delta_isa: float = 0.0):
    """
    Calcula temperatura e pressão de base de cada camada
    considerando o Delta ISA informado.
    """

    T_bases = [T0 + delta_isa]
    P_bases = [P0]

    for i in range(len(LAYERS) - 1):

        h0 = LAYERS[i].h_base
        h1 = LAYERS[i + 1].h_base
        L = LAYERS[i].lapse

        Tb = T_bases[i]
        Pb = P_bases[i]

        dh = h1 - h0

        if abs(L) < 1e-12:
            T_next = Tb
            P_next = Pb * math.exp(
                -G0 * dh / (R * Tb)
            )
        else:
            T_next = Tb + L * dh
            P_next = Pb * (T_next / Tb) ** (
                -G0 / (R * L)
            )

        T_bases.append(T_next)
        P_bases.append(P_next)

    return T_bases, P_bases


# ==========================================================
# PROPRIEDADES ATMOSFÉRICAS
# ==========================================================

def atmosphere(h: float, delta_isa: float = 0.0):
    """
    Retorna:
        T [K]
        P [Pa]
        rho [kg/m³]
        a [m/s]
    """

    if h < 0:
        raise ValueError("Altitude deve ser >= 0 m")

    if h > TOP_ATMOSPHERE:
        raise ValueError(
            f"Modelo válido até {TOP_ATMOSPHERE/1000:.0f} km"
        )

    T_bases, P_bases = build_base_conditions(delta_isa)

    layer_idx = len(LAYERS) - 1

    for i in range(len(LAYERS) - 1):
        if LAYERS[i].h_base <= h < LAYERS[i + 1].h_base:
            layer_idx = i
            break

    hb = LAYERS[layer_idx].h_base
    L = LAYERS[layer_idx].lapse

    Tb = T_bases[layer_idx]
    Pb = P_bases[layer_idx]

    dh = h - hb

    if abs(L) < 1e-12:
        T = Tb
        P = Pb * math.exp(
            -G0 * dh / (R * Tb)
        )
    else:
        T = Tb + L * dh
        P = Pb * (T / Tb) ** (
            -G0 / (R * L)
        )

    rho = P / (R * T)

    a = math.sqrt(GAMMA * R * T)

    return T, P, rho, a


# ==========================================================
# GERAÇÃO DA TABELA
# ==========================================================

def generate_table(
    delta_isa: float = 0.0,
    step_m: float = 1000.0
):
    atmosphere_model = {
        "altitude_m": [],
        "temperature_K": [],
        "pressure_Pa": [],
        "density_kg_m3": [],
        "speed_of_sound_m_s": [],
    }

    h = 0.0

    while h <= TOP_ATMOSPHERE:

        T, P, rho, a = atmosphere(
            h,
            delta_isa=delta_isa
        )

        atmosphere_model["altitude_m"].append(h)
        atmosphere_model["temperature_K"].append(T)
        atmosphere_model["pressure_Pa"].append(P)
        atmosphere_model["density_kg_m3"].append(rho)
        atmosphere_model["speed_of_sound_m_s"].append(a)

        h += step_m

    return atmosphere_model



# ==========================================================
# EXEMPLO DE USO
# ==========================================================

if __name__ == "__main__":

    DELTA_ISA = -15.0      # +15°C ISA

    if DELTA_ISA > 0.0:
        output_file_name = f"./datasets/atmosphere_model_DISA_plus{int(DELTA_ISA)}.csv"
    elif DELTA_ISA < 0.0:
        output_file_name = f"./datasets/atmosphere_model_DISA_minus{int(np.abs(DELTA_ISA))}.csv"
    else:
        output_file_name = f"./datasets/atmosphere_model_ISA1976.csv"

    table = generate_table(
        delta_isa=DELTA_ISA,
        step_m=10.0
    )

    print(
        f"\nATMOSFERA ISA + ΔISA = {DELTA_ISA:+.1f} °C\n"
    )

    df = pd.DataFrame(table)
    df.to_csv(output_file_name, index=False)