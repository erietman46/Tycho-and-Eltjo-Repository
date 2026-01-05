import matplotlib.pyplot as plt

# Iteration numbers
iterations = [1, 2, 3, 4]

# MTOW values from each iteration (kg)
MTOW = [13149.4, 13124.8, 13122.8, 13122.7]

# Plot
plt.figure(figsize=(8,5))
plt.plot(iterations, MTOW, marker='o', linestyle='-', color='r')
plt.title("Convergence of MTOW over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MTOW (kg)")
plt.grid(True)
plt.xticks(iterations)
plt.ylim(min(MTOW)-10, max(MTOW)+10)
plt.show()