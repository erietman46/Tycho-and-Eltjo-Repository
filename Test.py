import numpy as np
from scipy.integrate import quad

# ------------------- Constants -------------------
b = 22.12
C_r = 3.53
lambda_w = 0.32
C_t = 1.11

c_l_alpha = 6.4      # 2D lift curve slope [1/rad]
c_d0 = 0.00525       # profile drag
tau = 0.5            # control surface effectiveness
C_e_over_C_h = 0.28  # Aileron chord ratio

# --- flight condition inputs ---
CL_Max = 1.5
M_Max = 22238.96     # maximum takeoff mass in kg
S = 51.4             # wing area in m^2
T = 288.15
R = 287
P_atm = 101325
g = 9.80665

# ------------------- Utility Functions -------------------
def chord(y):
    """Chord distribution (linear taper wing)."""
    return C_r - (2 * (C_r - C_t)) / b * y

def aileron_area(b1, b2):
    """Reference aileron area between span stations b1 and b2."""
    return quad(chord, b1, b2)[0]

def CL_delta_a(c_l_alpha, tau, S_ref, b, C_r, b1, b2, lambda_w):
    t1 = 0.5 * (b2**2 - b1**2)
    t2 = (2 * (1 - lambda_w) / (3 * b)) * (b2**3 - b1**3)
    return (2 * c_l_alpha * tau / (S_ref * b)) * C_r * (t1 - t2)

def Cl_p(c_l_alpha, c_d0, S_ref, b, C_r, lambda_w):
    t1 = (1 / 3) * (b / 2) ** 3
    t2 = ((1 - lambda_w) / (2 * b)) * (b / 2) ** 4
    return -4 * (c_l_alpha + c_d0) / (S_ref * b**2) * C_r * (t1 - t2)

def roll_rate(CL_da, Clp_val, delta_a, V, b):
    """Roll rate in rad/s."""
    return -CL_da / Clp_val * delta_a * (2 * V / b)

def compute_speeds():
    """Return stall speed and minimum control speed (m/s)."""
    W = M_Max * g
    rho = P_atm / (R * T)
    V_stall = np.sqrt(2 * W / (S * rho * CL_Max))
    V_min_control = 1.13 * V_stall
    return V_stall, V_min_control

# ------------------- Pre-computations -------------------
V_sr, V_mc = compute_speeds()
V = V_mc

# Roll requirements
P_target = np.radians(4)     # 4°/s target
safety_factor = 5
P_req = P_target * safety_factor

# Limits
b1_min, b1_max = 5, 8
b2_min, b2_max = 8.5, 10.5
delta_min, delta_max = 10, 25

# ------------------- Iteration -------------------
best_solution = None
smallest_error = float("inf")

for b1 in np.arange(b1_min, b1_max + 1e-6, 0.1):
    for b2 in np.arange(b2_min, b2_max + 1e-6, 0.1):
        if b2 <= b1:
            continue
        S_ref_val = aileron_area(b1, b2)
        if S_ref_val <= 0 or (b2 - b1) <= 0:
            continue

        Clp_val = Cl_p(c_l_alpha, c_d0, S_ref_val, b, C_r, lambda_w)

        for delta_a in np.radians(np.arange(delta_min, delta_max + 1e-6, 1)):
            CL_da = CL_delta_a(c_l_alpha, tau, S_ref_val, b, C_r, b1, b2, lambda_w)
            P_actual = roll_rate(CL_da, Clp_val, delta_a, V, b)

            if P_actual >= P_req:
                err = abs(P_actual - P_req)
                if err < smallest_error:
                    smallest_error = err
                    best_solution = (b1, b2, np.degrees(delta_a), P_actual, S_ref_val)

# ------------------- Output -------------------
print("Speed summary (computed):")
print(f"  V_sr (stall) = {V_sr:.3f} m/s = {V_sr*1.943844:.2f} kt")
print(f"  V_mc (=1.13*V_sr) = {V_mc:.3f} m/s = {V_mc*1.943844:.2f} kt")
print(f"  Using V = V_mc = {V:.3f} m/s for roll calculations\n")

if best_solution:
    b1, b2, da, P_actual, S_ref_best = best_solution
    print(f"Target roll rate (with safety factor {safety_factor:.1f}×): "
          f"{np.degrees(P_req):.2f}°/s")
    print("\nBest feasible configuration:")
    print(f"  Inboard start  b1 = {b1:.2f} m")
    print(f"  Outboard end   b2 = {b2:.2f} m")
    print(f"  Deflection     δa = {da:.1f}°")
    print(f"  Achieved roll rate = {np.degrees(P_actual):.2f}°/s")
    print(f"  Aileron area = {S_ref_best:.3f} m²")
else:
    print("No feasible aileron configuration found.")
