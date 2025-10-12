import matplotlib.pyplot as plt
import numpy as np

# V points
V_points = [0, 40.3, 51.53, 64]
# Positive gust n
n_pos = [1, 2.212, 1.703, 1.728]
# Negative gust n
n_neg = [1, -0.212, 0.297, 0.272]

# Find n_max and corresponding V
n_max = max(n_pos)
V_at_n_max = V_points[n_pos.index(n_max)]

# V axis for ticks
V_ticks = np.arange(0, 71, 5)

plt.figure(figsize=(8,4))

# Plot gust lines
plt.plot(V_points, n_pos, 'b-o', label=f'Positive Gust (n_max={n_max:.2f} at V={V_at_n_max:.1f} m/s)', linewidth=2)
plt.plot(V_points, n_neg, 'r-o', label='Negative Gust', linewidth=2)

# Nominal load factor
plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='Nominal n=1')

# Vertical dashed lines connecting positive and negative points
for V, n1, n2 in zip(V_points[1:], n_pos[1:], n_neg[1:]):
    plt.plot([V,V], [n1,n2], 'k--', linewidth=1)

# Labels for key points
labels_pos = ['A\'','B\'','C\'','D\'']
labels_neg = ['A\'','G\'','F\'','E\'']
for V, n, label in zip(V_points, n_pos, labels_pos):
    plt.text(V, n+0.1, label, fontsize=8, ha='center', color='blue')
for V, n, label in zip(V_points, n_neg, labels_neg):
    plt.text(V, n-0.15, label, fontsize=8, ha='center', color='red')

# Key V points at bottom
V_labels = ['0','V_B','V_C','V_D']
for V, label in zip(V_points, V_labels):
    plt.text(V, -0.4, label, fontsize=8, ha='center')

# Axis limits and ticks
plt.xlim(0,70)
plt.ylim(-0.5,3)
plt.xticks(V_ticks)
plt.xlabel('V [m/s]')
plt.ylabel('n')
plt.title('Gust Load Diagram')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend(fontsize=8)

plt.tight_layout()
plt.show()
