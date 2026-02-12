import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER-DEFINED / UNKNOWN PARAMETERS (LEFT OPEN ON PURPOSE)
# ============================================================

c = 0.16                 # chord [m]
h = 0.40                 # effective tunnel height [m]
t_over_c = 0.104         # max thickness ratio
M = 0.0495               # Mach number
U_inf = 17.0             # freestream velocity [m/s]

# ============================================================
# 1) READ DATA
# ============================================================

df = pd.read_csv("2D Experimental Data.txt", sep=r"\s+")

df["Alpha"] = pd.to_numeric(df["Alpha"], errors="coerce")
df["Delta_Pb"] = pd.to_numeric(df["Delta_Pb"], errors="coerce")

pressure_cols = [f"P{i:03d}" for i in range(1, 50)]
df[pressure_cols] = df[pressure_cols].apply(pd.to_numeric, errors="coerce")

upper_cols = pressure_cols[:25]
lower_cols = pressure_cols[25:]

# ============================================================
# 2) PRESSURE TAP COORDINATES (NORMALIZED BY CHORD)
# ============================================================

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
# 4) WAKE RAKE GEOMETRY
# ============================================================

y_total_mm = np.array([
    0,12,21,27,33,39,45,51,57,63,69,72,75,78,81,84,87,90,93,96,
    99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,
    144,147,150,156,162,168,174,180,186,195,207,219
], dtype=float)

y_static_mm = np.array([
    43.5, 55.5, 67.5, 79.5, 91.5, 103.5,
    115.5, 127.5, 139.5, 151.5, 163.5, 175.5
], dtype=float)

def interp_with_linear_extrap(x_new, x, y):
    y_new = np.interp(x_new, x, y)
    if np.any(x_new < x[0]):
        slope = (y[1]-y[0])/(x[1]-x[0])
        y_new[x_new < x[0]] = y[0] + slope*(x_new[x_new < x[0]]-x[0])
    if np.any(x_new > x[-1]):
        slope = (y[-1]-y[-2])/(x[-1]-x[-2])
        y_new[x_new > x[-1]] = y[-1] + slope*(x_new[x_new > x[-1]]-x[-1])
    return y_new

# ============================================================
# 5) AIRFOIL CROSS-SECTIONAL AREA (FOR BLOCKAGE)
# ============================================================

x_common = np.sort(np.unique(np.concatenate([xu, xl])))
yu_c = np.interp(x_common, xu, yu)
yl_c = np.interp(x_common, xl, yl)

A_norm = np.trapezoid(yu_c - yl_c, x_common)
A = A_norm * c**2

beta = np.sqrt(1 - M**2)

# Shape factor (left open / approximate)
Lambda = 0.18

# ============================================================
# 6) FORCE & WAKE INTEGRATION
# ============================================================

results = []

for _, r in df.iterrows():

    if pd.isna(r["Alpha"]) or r["Delta_Pb"] == 0:
        continue

    alpha_deg = float(r["Alpha"])
    alpha = np.deg2rad(alpha_deg)

    rho = float(r["rho"])
    q_inf = float(r["P097"]) - float(r["P110"])

    Cp_u = r[upper_cols].to_numpy(dtype=float) / q_inf
    Cp_l = r[lower_cols].to_numpy(dtype=float) / q_inf

    Cp_u_c = np.interp(x_common, xu, Cp_u)
    Cp_l_c = np.interp(x_common, xl, Cp_l)

    Cn = np.trapezoid(Cp_l_c - Cp_u_c, x_common)
    Ca = (
        np.trapezoid(Cp_u * dyu_dx, xu)
        - np.trapezoid(Cp_l * dyl_dx, xl)
    )

    CL = Cn*np.cos(alpha) - Ca*np.sin(alpha)
    CD = Cn*np.sin(alpha) + Ca*np.cos(alpha)

    Cm_LE = -np.trapezoid((Cp_l_c - Cp_u_c)*x_common, x_common)
    Cm_c4 = Cm_LE + 0.25*Cn
    x_cp = -Cm_LE/Cn if Cn != 0 else np.nan

    # -------- WIND TUNNEL CORRECTIONS --------

    sigma = (np.pi**2/48)*(c/h)**2
    eps_s = (Lambda*sigma/beta**3)*(1 + 1.1*beta*alpha**2/t_over_c)
    eps_w = 0.25*(c/h)*(1+0.4*M**2)/beta**2 * CD
    eps_b = eps_s + eps_w

    q_ratio = 1 - (2 - M**2)*eps_b

    CL_corr = CL*q_ratio
    CD_corr = CD*q_ratio
    Cm_corr = Cm_c4*q_ratio

    delta_alpha = sigma/(2*np.pi*beta)*(CL + 4*Cm_c4)
    alpha_corr = alpha_deg + np.rad2deg(delta_alpha)

    # -------- WAKE RAKE DRAG (UNCORRECTED) --------

    Pt = r[[f"P{i:03d}" for i in range(50, 97)]].to_numpy(dtype=float)
    Ps = r[[f"P{i:03d}" for i in range(98, 110)]].to_numpy(dtype=float)

    Ps_i = interp_with_linear_extrap(y_total_mm, y_static_mm, Ps)
    q_wake = np.maximum(Pt - Ps_i, 0.0)
    u = np.sqrt(2*q_wake/rho)

    dy = np.diff(y_total_mm)*1e-3
    u_mid = 0.5*(u[:-1] + u[1:])
    p_mid = 0.5 * (Ps_i[:-1] + Ps_i[1:])

    U_inf = 17.0  # freestream velocity [m/s]

    D_mom = rho * np.sum(u_mid * (U_inf - u_mid) * dy)
    D_pres = np.sum((Ps_i[-1] - p_mid) * dy)

    D_wake = D_mom + D_pres

    c = 0.16  # chord length [m]
    CD_wake = D_wake / (0.5 * rho * U_inf**2 * c)

    if (not np.isfinite(CD_wake)) or (CD_wake < 0.0):
        CD_wake = CD

    results.append([
        alpha_deg, CL, CD, Cm_c4, x_cp,
        alpha_corr, CL_corr, CD_corr, Cm_corr,
        CD_wake
    ])

# ============================================================
# 7) POLARS
# ============================================================

polars = pd.DataFrame(results, columns=[
    "Alpha","CL","CD","Cm_c4","x_cp",
    "Alpha_corr","CL_corr","CD_corr","Cm_corr",
    "CD_wake"
])

i_stall = polars["Alpha"].idxmax()
pre_stall = polars.loc[:i_stall]
post_stall = polars.loc[i_stall+1:]

# ============================================================
# 8) PLOTS (RAW + CORRECTED)
# ============================================================

plt.figure()
plt.plot(pre_stall["Alpha"], pre_stall["CL"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha"], post_stall["CL"], "s--", label="Post-stall")
plt.xlabel("α [deg]")
plt.ylabel("CL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(pre_stall["Alpha"], pre_stall["CD"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha"], post_stall["CD"], "s--", label="Post-stall")
plt.xlabel("α [deg]")
plt.ylabel("CD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(pre_stall["Alpha"], pre_stall["Cm_c4"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha"], post_stall["Cm_c4"], "s--", label="Post-stall")
plt.xlabel("α [deg]")
plt.ylabel("Cm c/4")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure()
plt.plot(pre_stall["CD"], pre_stall["CL"], "o-", label="Pre-stall")
plt.plot(post_stall["CD"], post_stall["CL"], "s--", label="Post-stall")
plt.xlabel("CD")
plt.ylabel("CL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(pre_stall["Alpha"], pre_stall["x_cp"], "o-", label="Pre-stall")
plt.plot(post_stall["Alpha"], post_stall["x_cp"], "s--", label="Post-stall")
plt.axhline(0.25, linestyle=":")
plt.xlabel("α [deg]")
plt.ylabel("x_cp / c")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(
    pre_stall["Alpha"],
    pre_stall["CD_wake"],
    "o-",
    label="Wake drag (Pre-stall)"
)
plt.plot(
    post_stall["Alpha"],
    post_stall["CD_wake"],
    "s--",
    label="Wake drag (Post-stall)"
)
plt.xlabel("α [deg]")
plt.ylabel("CD (wake)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()

plt.plot(
    pre_stall["Alpha"],
    pre_stall["CD"],
    "o-",
    label="Surface CD (Pre-stall)"
)
plt.plot(
    post_stall["Alpha"],
    post_stall["CD"],
    "s--",
    label="Surface CD (Post-stall)"
)

plt.plot(
    pre_stall["Alpha"],
    pre_stall["CD_wake"],
    "o:",
    linewidth=2,
    label="Wake CD (Pre-stall)"
)
plt.plot(
    post_stall["Alpha"],
    post_stall["CD_wake"],
    "s:",
    linewidth=2,
    label="Wake CD (Post-stall)"
)

plt.xlabel("α [deg]")
plt.ylabel("Drag coefficient")
plt.title("Surface-Pressure Drag vs Wake-Rake Drag")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()

plt.plot(polars["Alpha"], polars["CL"], "o-", label="CL raw")
plt.plot(polars["Alpha_corr"], polars["CL_corr"], "o:", linewidth=2,
         label="CL corrected")

plt.xlabel("α [deg]")
plt.ylabel("CL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()

plt.plot(polars["Alpha"], polars["CD"], "o-", label="CD raw")
plt.plot(polars["Alpha_corr"], polars["CD_corr"], "o:", linewidth=2,
         label="CD corrected")

plt.xlabel("α [deg]")
plt.ylabel("CD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()

plt.plot(polars["Alpha"], polars["Cm_c4"], "o-", label="Cm raw")
plt.plot(polars["Alpha_corr"], polars["Cm_corr"], "o:", linewidth=2,
         label="Cm corrected")

plt.xlabel("α [deg]")
plt.ylabel("Cm c/4")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()

plt.plot(polars["CD_corr"], polars["CL_corr"], "o-")

plt.xlabel("CD (corrected)")
plt.ylabel("CL (corrected)")
plt.grid(True)
plt.tight_layout()
plt.show()



