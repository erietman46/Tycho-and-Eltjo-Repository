import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1) READ EXPERIMENTAL DATA
# ============================================================
df = pd.read_csv("2D Experimental Data.txt", sep=r"\s+")

# Force numeric conversion
df["Alpha"] = pd.to_numeric(df["Alpha"], errors="coerce")
df["Delta_Pb"] = pd.to_numeric(df["Delta_Pb"], errors="coerce")

pressure_cols = [f"P{i:03d}" for i in range(1, 50)]
df[pressure_cols] = df[pressure_cols].apply(pd.to_numeric, errors="coerce")

upper_cols = pressure_cols[:25]
lower_cols = pressure_cols[25:]

# ============================================================
# 2) PRESSURE TAP COORDINATES (NORMALIZED BY CHORD)
# ============================================================

# Upper surface
xu = np.array([
    0.00000, 0.0035626, 0.0133331, 0.0366108, 0.072922,
    0.1135604, 0.1559135, 0.1991328, 0.2428443, 0.2868627,
    0.3310518, 0.3753128, 0.4195991, 0.4638793, 0.508156,
    0.552486, 0.5969223, 0.6413685, 0.68579, 0.7302401,
    0.7747357, 0.8193114, 0.8638589, 0.908108, 1.0
])

yu = np.array([
    0.00000, 0.0077154, 0.0160115, 0.0287759, 0.0415707,
    0.0513022, 0.0585007, 0.0637480, 0.0674148, 0.0697480,
    0.0709219, 0.0710225, 0.0700937, 0.0681628, 0.0652532,
    0.0614225, 0.0568254, 0.0516453, 0.0459453, 0.0397658,
    0.0332133, 0.0263941, 0.0194846, 0.0127669, 0.00000
])

# Lower surface
xl = np.array([
    0.00000, 0.0043123, 0.0147147, 0.0392479, 0.0779506,
    0.120143, 0.1632276, 0.2067013, 0.2503792, 0.2941554,
    0.3379772, 0.3818675, 0.4257527, 0.4696278, 0.5135062,
    0.5573662, 0.6012075, 0.6450502, 0.688901, 0.7328011,
    0.7767783, 0.8207965, 0.8647978, 1.0
])

yl = np.array([
     0.00000,-0.0057176,-0.0109275,-0.0177203,-0.0237270,
    -0.0276684,-0.0302746,-0.0319868,-0.0330615,-0.0336298,
    -0.0337697,-0.0335304,-0.0329378,-0.0320029,-0.0307206,
    -0.0291060,-0.0271424,-0.0248323,-0.0221935,-0.0192575,
    -0.0161034,-0.0128273,-0.0094874, 0.00000
])

# ============================================================
# 3) SURFACE SLOPES
# ============================================================
dyu_dx = np.gradient(yu, xu)
dyl_dx = np.gradient(yl, xl)

# ============================================================
# 4) PRESSURE INTEGRATION
# ============================================================
results = []

for _, r in df.iterrows():

    if pd.isna(r["Alpha"]) or pd.isna(r["Delta_Pb"]) or r["Delta_Pb"] == 0:
        continue

    alpha = np.deg2rad(float(r["Alpha"]))
    q = float(r["P097"]) - float(r["P110"])

    # Pressure coefficients
    Cp_u = r[upper_cols].to_numpy(dtype=float) / q
    Cp_l = r[lower_cols].to_numpy(dtype=float) / q

    # Common x-grid
    x_common = np.sort(np.unique(np.concatenate([xu, xl])))

    Cp_u_c = np.interp(x_common, xu, Cp_u)
    Cp_l_c = np.interp(x_common, xl, Cp_l)

    # Normal and axial force coefficients
    Cn = np.trapezoid(Cp_l_c - Cp_u_c, x_common)

    Ca = (
        np.trapezoid(Cp_u * dyu_dx, xu)
        - np.trapezoid(Cp_l * dyl_dx, xl)
    )

    # Lift and drag
    CL = Cn * np.cos(alpha) - Ca * np.sin(alpha)
    CD = Cn * np.sin(alpha) + Ca * np.cos(alpha)

    # ========================================================
    # Pitching moment coefficients
    # ========================================================

    # About leading edge
    Cm_LE = -np.trapezoid(((Cp_l_c - Cp_u_c) * x_common), x_common)

    Cm_c4 = Cm_LE + 0.25 * Cn


    results.append([np.rad2deg(alpha), CL, CD, Cm_c4])

# ============================================================
# 5) BUILD POLARS DATAFRAME
# ============================================================
polars = pd.DataFrame(
    results,
    columns=["Alpha_deg", "CL", "CD", "Cm_c4"]
).reset_index(drop=True)

# Stall index
i_stall = polars["Alpha_deg"].idxmax()

pre_stall  = polars.loc[:i_stall]
post_stall = polars.loc[i_stall+1:]

# Lift slope (pre-stall)
alpha_rad_pre = np.deg2rad(pre_stall["Alpha_deg"].to_numpy())
CL_pre = pre_stall["CL"].to_numpy()

dCL_dalpha, _ = np.polyfit(alpha_rad_pre, CL_pre, 1)
print(f"dCL/dalpha (pre-stall) = {dCL_dalpha:.3f} 1/rad")

# ============================================================
# 6) PLOTS
# ============================================================

# CL vs alpha
plt.figure()
plt.plot(pre_stall["Alpha_deg"], pre_stall["CL"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha_deg"], post_stall["CL"], "s--", label="Post-stall")
plt.xlabel("Angle of attack α [deg]")
plt.ylabel("$C_L$")
plt.title("$C_L$ vs Angle of Attack")
plt.legend()
plt.grid(True)
plt.show()

# CD vs alpha
plt.figure()
plt.plot(pre_stall["Alpha_deg"], pre_stall["CD"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha_deg"], post_stall["CD"], "s--", label="Post-stall")
plt.xlabel("Angle of attack α [deg]")
plt.ylabel("$C_D$")
plt.title("$C_D$ vs Angle of Attack")
plt.legend()
plt.grid(True)
plt.show()

# Cm_c4 vs alpha
plt.figure()
plt.plot(pre_stall["Alpha_deg"], pre_stall["Cm_c4"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha_deg"], post_stall["Cm_c4"], "s--", label="Post-stall")
plt.xlabel("Angle of attack α [deg]")
plt.ylabel("$C_{m,c4}$")
plt.title("Pitching Moment Coefficient about quarter chord")
plt.legend()
plt.grid(True)
plt.show()

# CL vs CD
plt.figure()
plt.plot(pre_stall["CD"], pre_stall["CL"], "o-", label="Pre-stall")
plt.plot(post_stall["CD"], post_stall["CL"], "s--", label="Post-stall")
plt.xlabel("$C_D$")
plt.ylabel("$C_L$")
plt.title("Aerodynamic Polar")
plt.legend()
plt.grid(True)
plt.show()
