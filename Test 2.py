import numpy as np
import matplotlib.pyplot as plt

def plot_sols(m, k, gamma, T_end):
    t = np.linspace(0, T_end, 500)

    omega_0 = np.sqrt(k/m)
    discriminant = gamma**2 - 4*k*m

    A = 2
    B = 1

    if gamma == 0:
        # Undamped
        y = A*np.cos(omega_0*t) + B*np.sin(omega_0*t)
        color = 'green'
        label = 'Undamped'
    elif discriminant > 0:
        # Overdamped
        r1 = (-gamma + np.sqrt(discriminant))/(2*m)
        r2 = (-gamma - np.sqrt(discriminant))/(2*m)
        y = A*np.exp(r1*t) + B*np.exp(r2*t)
        color = 'blue'
        label = 'Overdamped'
    elif discriminant == 0:
        # Critically damped
        y = (A + B*t)*np.exp(-gamma/(2*m)*t)
        color = 'purple'
        label = 'Critically damped'
    else:
        # Underdamped
        mu = np.sqrt(4*k*m - gamma**2)/(2*m)
        y = np.exp(-gamma/(2*m)*t)*(A*np.cos(mu*t) + B*np.sin(mu*t))
        color = 'red'
        label = 'Underdamped'

    plt.figure(figsize=(8,5))
    plt.plot(t, y, color=color, linewidth=2, label=label)
    plt.title(f"$\gamma^2 = {gamma**2}$, $4km = {4*k*m}$")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test
m = 1.16
k = 0.8
gamma = 2.7
T_end = 5
plot_sols(m, k, gamma, T_end)
