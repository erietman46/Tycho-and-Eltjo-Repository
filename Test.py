import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
V_points = [0, 40.99, 58.18, 87.26]  # speeds in m/s
n_pos = [1, 2.169, 1.753, 1.942]    # positive gust
n_neg = [1, -0.169, 0.247, 0.058]   # negative gust

labels_pos = ["A'", "B'", "C'", "D'"]
labels_neg = ["A'", "G'", "F'", "E'"]

# Max positive load factor
n_max = max(n_pos)
V_at_n_max = V_points[n_pos.index(n_max)]

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10,5))

# Fill gust envelope
plt.fill_between(V_points, n_neg, n_pos, color='lightblue', alpha=0.3, label='Gust Envelope')

# Plot positive and negative gust lines
plt.plot(V_points, n_pos, 'b-o', linewidth=2, label=f'Positive Gust (n_max={n_max:.2f} at V={V_at_n_max:.1f} m/s)')
plt.plot(V_points, n_neg, 'r-o', linewidth=2, label='Negative Gust')

# Nominal load factor
plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='Nominal n=1')

# Vertical dashed lines connecting positive and negative points
for V, n1, n2 in zip(V_points[1:], n_pos[1:], n_neg[1:]):
    plt.plot([V, V], [n1, n2], 'k--', linewidth=1)

# Annotate points
for V, n, label in zip(V_points, n_pos, labels_pos):
    plt.text(V, n + 0.05, label, color='blue', fontsize=10, ha='center')
for V, n, label in zip(V_points, n_neg, labels_neg):
    plt.text(V, n - 0.1, label, color='red', fontsize=10, ha='center')

# Bottom V labels
V_labels = ['0', 'V_B', 'V_C', 'V_D']
for V, label in zip(V_points, V_labels):
    plt.text(V, -0.3, label, fontsize=10, ha='center')

# Axis limits and ticks
plt.xlim(0, 95)
plt.ylim(-0.5, 2.3)
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(-0.5, 2.5, 0.5))

# Labels, title, grid, legend
plt.xlabel('V [m/s]', fontsize=12)
plt.ylabel('n', fontsize=12)
plt.title('Gust Load Diagram', fontsize=14)
plt.grid(True, linestyle=':', linewidth=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
