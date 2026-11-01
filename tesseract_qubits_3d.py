#!/usr/bin/env python3
"""Tesseract projection with 16 four-qubit basis states as vertices."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# 16 vertices of the tesseract: all binary combinations for 4 qubits
verts4 = np.array(list(product([0, 1], repeat=4)), dtype=float)
# Center them
verts4 -= 0.5

# Simple rotation/projection from 4D to 3D
theta = np.radians(25)
phi = np.radians(35)

def rot_xy(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

def rot_zw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -s],
        [0, 0, s,  c]
    ])

R = rot_xy(theta) @ rot_zw(phi)
verts_rot = verts4 @ R.T

# Project to 3D by dropping the 4th coordinate
verts3 = verts_rot[:, :3]

# Depth for shading (use the dropped coordinate)
depth = verts_rot[:, 3]
dmin, dmax = depth.min(), depth.max()
norm_depth = (depth - dmin) / (dmax - dmin + 1e-12)

# Edges: connect vertices that differ by exactly one bit
edges = []
for i in range(16):
    for j in range(i + 1, 16):
        if np.sum(np.abs(verts4[i] - verts4[j])) == 1.0:
            edges.append((i, j))

fig = plt.figure(figsize=(11, 9), facecolor="black")
ax = fig.add_subplot(111, projection="3d", facecolor="black")

# Plot edges with depth-based color
for i, j in edges:
    avg_depth = (norm_depth[i] + norm_depth[j]) / 2
    color = plt.cm.plasma(avg_depth)
    ax.plot(
        [verts3[i, 0], verts3[j, 0]],
        [verts3[i, 1], verts3[j, 1]],
        [verts3[i, 2], verts3[j, 2]],
        color=color, linewidth=1.5, alpha=0.7
    )

# Plot vertices, size and color by depth
sizes = 40 + 80 * norm_depth
colors = plt.cm.viridis(norm_depth)
ax.scatter(
    verts3[:, 0], verts3[:, 1], verts3[:, 2],
    s=sizes, c=colors, edgecolors="white", linewidths=0.5, zorder=5
)

# Label each vertex with its qubit state
labels = ["".join(str(int(b + 0.5)) for b in v) for v in verts4 + 0.5]
for k, label in enumerate(labels):
    ax.text(
        verts3[k, 0], verts3[k, 1], verts3[k, 2],
        label, color="white", fontsize=7, ha="center", va="bottom"
    )

ax.set_title("Tesseract Projection: 16 Four-Qubit Basis States", color="white", pad=20)
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")
ax.tick_params(colors="white")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Equal aspect
max_range = np.array([verts3.max(axis=0) - verts3.min(axis=0)]).max() / 2
mid = verts3.mean(axis=0)
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

plt.tight_layout()
plt.savefig("tesseract_qubits_3d.png", dpi=150, facecolor="black", bbox_inches="tight")
print("Saved tesseract_qubits_3d.png")
print(f"Vertices: {len(verts4)}, Edges: {len(edges)}")
