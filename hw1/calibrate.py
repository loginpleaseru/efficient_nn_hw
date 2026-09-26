import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from equations import flops, bytes_moved, latency


def calibrate_theta(df):
    d = df.dropna(subset=['latency'])
    S = d['S'].to_numpy(dtype=float)
    B = d['B'].to_numpy(dtype=float)
    ratio = d['latency'].to_numpy() / latency(S, B, 1.0)
    return float(np.exp(np.mean(np.log(ratio)))) #смотрю ср геометрическое тк нужно чтобы тета учитывал отклонение в обе стороны одинаково


def calibrate_theta_energy(df):
    d = df.dropna(subset=['energy'])
    S = d['S'].to_numpy(dtype=float)
    B = d['B'].to_numpy(dtype=float)
    y = d['energy'].to_numpy()
    X = np.column_stack([flops(S, B), bytes_moved(S, B)]) / y[:, None]
    scale = np.linalg.norm(X, axis=0)
    coef, _ = nnls(X / scale, np.ones_like(y))
    alpha, beta = coef / scale
    return [float(alpha), float(beta)]


if __name__ == '__main__':
    df = pd.read_csv('results/measurements.csv')
    df_to_calibrate = df[df['is_validation']==False]

    theta = calibrate_theta(df_to_calibrate)
    theta_energy = calibrate_theta_energy(df_to_calibrate)

    with open('results/theta.json', 'w') as f:
        json.dump({'theta': theta,
                   'theta_energy': {'alpha_J_per_flop': theta_energy[0], 'beta_J_per_byte': theta_energy[1]},
                   'peak_flops': 8.1e12, 'peak_bandwidth': 320e9,
                   'gpu': "Tesla T4"}, f, indent=2)
    print('theta =', theta)
    print('theta_energy =', theta_energy)
