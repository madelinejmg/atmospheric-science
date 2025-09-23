import numpy as np
import matplotlib.pyplot as plt

class Atmosphere:
    # physical constants (SI)
    g = 9.80665            # m s^-2
    R_d = 287.05           # J kg^-1 K^-1 (dry air)
    p0 = 1013.25 * 100     # Pa

    # (z0_km, z1_km, lapse[K/km], T_base[K at z0])
    layers = [
        (0, 11,  -6.5, 288.15),  # 0-11 km
        (11, 20,  0.0,  216.65), # 11-20
        (20, 32,  1.0,  216.65), # 20-32
        (32, 47,  2.8,  228.65), # 32-47
        (47, 51,  0.0,  270.65), # 47-51
        (51, 71, -2.8,  270.65), # 51-71
    ]

    @staticmethod
    def T_of_z(z_m):
        """Temperature at height z (meters), piecewise-linear in layers."""
        z_km = z_m / 1000.0
        for z0, z1, lapse, T0 in Atmosphere.layers:
            if z0 <= z_km < z1:
                return T0 + lapse * (z_km - z0)
        # clamp to top layer end (70–71 km spec uses 71)
        z0, z1, lapse, T0 = Atmosphere.layers[-1]
        return T0 + lapse * (min(z_km, z1) - z0)

    @staticmethod
    def integrate_pressure(z_top_m=70000, dz=100.0):
        """Return arrays z, T(z), p(z) from 0 to z_top_m."""
        z = np.arange(0.0, z_top_m + dz, dz)
        T = np.array([Atmosphere.T_of_z(zi) for zi in z])
        p = np.empty_like(z)
        p[0] = Atmosphere.p0
        # hydrostatic + ideal gas (forward exponential step)
        factor = -Atmosphere.g * dz / Atmosphere.R_d
        for i in range(1, len(z)):
            p[i] = p[i-1] * np.exp(factor / T[i])
        return z, T, p

    @staticmethod
    def isothermal_pressure(z, T_iso):
        """Isothermal pressure profile for comparison."""
        return Atmosphere.p0 * np.exp(-Atmosphere.g * z / (Atmosphere.R_d * T_iso))


if __name__ == "__main__":
    # compute profiles
    z, T, p = Atmosphere.integrate_pressure(z_top_m=70000, dz=100.0)
    T_mean = T.mean()  # mean over 0–70 km for the comparison
    p_iso = Atmosphere.isothermal_pressure(z, T_mean)

    km = z / 1000.0
    plt.figure()
    plt.plot(T, km)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Height (km)")
    plt.title("Piecewise-Linear Temperature Profile (0–70 km)")
    plt.grid(True, alpha=0.3)

    plt.figure()
    plt.semilogy(p/100.0, km, label="US76 piecewise T(z)")
    plt.semilogy(p_iso/100.0, km, "--", label=f"Isothermal (T̄={T_mean:.1f} K)")
    plt.xlabel("Pressure (hPa)")
    plt.ylabel("Height (km)")
    plt.title("Pressure Profile on Log Scale")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()


        