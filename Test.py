import sympy as sp

import math

def y_MAC_sympy(c_root, c_tip, b):
    """
    Compute the spanwise location of the Mean Aerodynamic Chord (MAC)
    using symbolic integration in sympy.
    
    Parameters:
        c_root : float
            Root chord length
        c_tip : float
            Tip chord length
        b : float
            Full wingspan
            
    Returns:
        y_MAC : float
            Spanwise location of MAC along half-span
    """
    y = sp.symbols('y', real=True)
    
    # chord distribution function
    c_y = c_root - (2*(c_root - c_tip)/b) * y
    
    # integrand for y_MAC
    integrand = y * c_y
    
    # definite integral from 0 to b/2
    integral = sp.integrate(integrand, (y, 0, b/2))
    
    # trapezoidal wing area
    S = (b/2) * (c_root + c_tip)
    
    # y_MAC formula
    y_MAC = (2/S) * integral
    
    # evaluate numerically
    return float(y_MAC.evalf())
c_r = 1.118731233
c_t = 3.53865671
b = 22.12  # example wingspan

y_mac = y_MAC_sympy(c_r, c_t, b)
print(f"y_MAC = {y_mac:.4f}")


# Define symbol
y = sp.Symbol('y', real=True)

def c_MAC(c_r, c_t, b):
    """
    Computes the mean aerodynamic chord (MAC) for a trapezoidal wing
    using the integral definition and Sympy.
    
    Parameters:
        c_r (float): Root chord length
        c_t (float): Tip chord length
        b   (float): Wingspan
        
    Returns:
        sympy expression (exact) and float (approximate)
    """
    # chord distribution
    c_y = c_r - (2*(c_r - c_t)/b) * y
    
    # wing area (trapezoid)
    S = (b/2) * (c_r + c_t)
    
    # semispan
    s = b/2
    
    # integral definition of MAC
    integral_val = sp.integrate(c_y**2, (y, 0, s))
    c_mac_expr = (2/S) * integral_val
    
    return sp.simplify(c_mac_expr), float(c_mac_expr)

# Example usage
print('c_MAC:', c_MAC(3.53865671, 1.118731233, 22.12))


def leading_edge_sweep_and_xLEMAC(lambda_c4_deg, taper_ratio, b, c_root):
    """
    Compute leading edge sweep angle and x_LE of MAC from quarter-chord sweep.
    
    Parameters:
    lambda_c4_deg : float : Quarter-chord sweep angle in degrees
    taper_ratio : float : Tip chord / root chord
    b : float : Wingspan
    c_root : float : Root chord length
    
    Returns:
    tuple : (leading edge sweep in degrees, x_LE_MAC)
    """
    # Convert quarter-chord sweep to radians
    lambda_c4_rad = math.radians(lambda_c4_deg)
    
    # Compute leading edge sweep in radians
    lambda_le_rad = math.atan(math.tan(lambda_c4_rad) - (c_root / (2*b)) * (taper_ratio - 1))
    
    # Convert back to degrees
    lambda_le_deg = math.degrees(lambda_le_rad)
    
    # Compute y_MAC for trapezoidal wing
    y_MAC = (b / 6) * (1 + 2*taper_ratio) / (1 + taper_ratio)
    
    # Compute x_LE_MAC
    x_LE_MAC = y_MAC * math.tan(lambda_le_rad)
    
    return lambda_le_deg, x_LE_MAC

# Given values
lambda_c4 = 24.02  # degrees
taper_ratio = 0.32
b = 22.12  # wingspan
c_r = 3.54  # root chord

# Compute leading edge sweep and x_LE_MAC
lambda_le, x_LE_MAC = leading_edge_sweep_and_xLEMAC(lambda_c4, taper_ratio, b, c_r)

print(f"Leading Edge Sweep: {lambda_le:.2f} degrees")
print(f"x_LE_MAC: {x_LE_MAC:.2f} meters")
