#!/usr/bin/env python3
"""Tesseract projection with 4-qubit basis-state labels."""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

verts = np.array(list(product([-1.0, 1.0], repeat=4)))

def project(v):
    x, y, z, w = v
    s = 2.0 - w
    return np.array([x * s, y * s, z * s])

pts3 = np.array([project(v) for v in verts])

def view2d(p):
    a, b = np.radians(25), np.radians(35)
    R = np.array([
        [np.cos(b), 0, np.sin(b)],
        [np.sin(a)*np.sin(b), np.cos(a), -np.sin(a)*np.cos(b)],
        [-np.cos(a)*np.sin(b), np.sin(a), np.cos(a)*np.cos(b)],
    ])
    q = R @ p
    return q[0], q[1], q[2]

scr = np.array([view2d(p) for p in pts3])
xs, ys, zs = scr.T

edges = []
for i, vi in enumerate(verts):
    for j, vj in enumerate(verts):
        if i < j and np.sum(vi != vj) == 1:
            edges.append((i, j))

def qlabel(v):
    bits = ["1" if c > 0 else "0" for c in v]
    return "|" + " ".join(bits) + ">"

fig, ax = plt.subplots(figsize=(11, 11))
for i, j in edges:
    d = (zs[i] + zs[j]) / 2
    alpha = 0.15 + 0.55 * (d - zs.min()) / (zs.max() - zs.min() + 1e-9)
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="slateblue",
            alpha=alpha, lw=1.2, zorder=1)

ax.scatter(xs, ys, s=90, c=zs, cmap="viridis", zorder=3, edgecolors="white", linewidths=0.6)

for i in range(len(verts)):
    ax.text(xs[i], ys[i] + 0.08, qlabel(verts[i]),
            ha="center", va="bottom", fontsize=9, color="#222222", zorder=4)

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Tesseract vertices = 16 four-qubit basis states", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("tesseract_qubits.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.show()
print("saved tesseract_qubits.png")
print("16 vertices, 32 edges")
for v in verts:
    print(qlabel(v), "->", np.round(project(v), 2))
