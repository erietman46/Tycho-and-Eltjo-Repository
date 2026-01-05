import matplotlib.pyplot as plt
# Data (from your mass breakdown table)
labels = ["OEM", "OEM + fuel", "OEM + max payload + fuel", "OEM + max payload"]
x_cg_mac = [0.15, 0.15, 0.548, 0.590]
mass_fraction = [0.57, 0.6732, 1.091, 0.988]

# Create figure
plt.figure(figsize=(8, 6))

# Plot loading path (connect all points and close the loop)
plt.plot(x_cg_mac + [x_cg_mac[0]], mass_fraction + [mass_fraction[0]],
         '-', color='blue', linewidth=2, label='Loading Path')

# Plot larger markers at each point
plt.scatter(x_cg_mac, mass_fraction, color='blue', s=120, zorder=3, edgecolors='black')

# Add MTOM line (red)
plt.axhline(y=1.0, color='red', linewidth=2, label='MTOM')

# Annotate each configuration with larger bold text
for label, x, y in zip(labels, x_cg_mac, mass_fraction):
    plt.text(x + 0.015, y + 0.015, label, fontsize=11, weight='bold',
             ha='left', va='bottom', color='dimgray')

# Axis labels and title
plt.xlabel(r'$x_i / \mathrm{MAC}$ (-)', fontsize=13)
plt.ylabel(r'Mass Fraction, $\dot{m}_i$ (-)', fontsize=13)
plt.title('Class I Loading Diagram', fontsize=15, weight='bold')

# Grid, limits, legend, and style
plt.xlim(0, 0.7)
plt.ylim(0, 1.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
