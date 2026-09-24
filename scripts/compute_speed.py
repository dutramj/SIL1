from vahsimulator.atmosphere import Atmosphere

atmosphere = Atmosphere()

alt = [20000]  # meters
M = [10.0]  # Mach number
for h in alt:
    for m in M:
        speed_of_sound, temperature, pressure, density = atmosphere.step(h)
        print(f"Altitude: {h} m, Mach: {m}, Velocity: {m*speed_of_sound:.4f} m/s. Temperature: {temperature:.4f} K, Pressure: {pressure:.4f} Pa, Density: {density:.4f} kg/m³")
