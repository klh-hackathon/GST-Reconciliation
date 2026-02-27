import matplotlib.pyplot as plt
import numpy as np

# Categories (must be same for all users)
categories = ["Speed", "Power", "Skill", "Accuracy", "Stamina"]
num_vars = len(categories)

# Example raw data for multiple users
data = {
    "User A": [80, 65, 90, 75, 85],
    "User B": [70, 75, 85, 80, 78],
    "User C": [60, 55, 70, 65, 72],
}

# Compute angle for each category
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Create plot
plt.figure()
ax = plt.subplot(111, polar=True)

# Plot each user
for user, values in data.items():
    values = values + values[:1]  # close the loop
    ax.plot(angles, values, label=user)
    ax.fill(angles, values, alpha=0.15)

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Set range
ax.set_ylim(0, 100)

# Add legend
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

plt.title("User Comparison Spider Chart")
plt.show()