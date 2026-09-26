import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equations import latency, memory, energy

df = pd.read_csv('results/measurements.csv')
with open('results/theta.json') as f:
    params = json.load(f)
theta = params['theta']
theta_energy = [params['theta_energy']['alpha_J_per_flop'], params['theta_energy']['beta_J_per_byte']]
all_image_sizes = np.sort(df['S'].unique())

d = df[df['memory'].astype(str) != 'OOM'].copy()
d['memory'] = d['memory'].astype(float)
d['is_validation'] = d['is_validation'].astype(bool)

B_line = np.arange(1, 257)
panels = [
    ('latency', 'latency, ms', 1e3, lambda S, B: latency(S, B, theta)),
    ('memory', 'peak memory, MB', 1e-6, memory),
    ('energy', 'energy, mJ', 1e3, lambda S, B: energy(S, B, theta_energy)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (col, ylabel, k, pred) in zip(axes, panels):
    for i, S in enumerate(all_image_sizes):
        c = plt.cm.viridis(i / (len(all_image_sizes) - 1))
        g = d[d['S'] == S]
        ax.plot(B_line, pred(S, B_line) * k, color=c, label=f'S={S}')
        ax.scatter(g.loc[~g['is_validation'], 'B'], g.loc[~g['is_validation'], col] * k, color=c, marker='o', s=18)
        ax.scatter(g.loc[g['is_validation'], 'B'], g.loc[g['is_validation'], col] * k, color=c, marker='x', s=30)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('batch size B')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{col}: measured (o calib, x valid) vs predicted (line)')
    ax.grid(True, which='both', alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
os.makedirs('results/figures', exist_ok=True)
fig.savefig('results/figures/predicted_vs_measured.png', dpi=150)
plt.show()
