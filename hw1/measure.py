import math
import os

import numpy as np
import pandas as pd
import pynvml
import torch

from equations import memory
from models import LoginModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

image_size_grid = np.array([32, 64, 128, 224, 256, 384, 512])
batch_size_grid = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
validate_image_size_grid = np.array([80, 176, 320, 448])
validate_batch_size_grid = np.array([11, 36, 74])

count_warmup, runs_count = 3, 11

all_image_sizes = np.sort(np.concatenate([image_size_grid, validate_image_size_grid]))
all_batch_sizes = np.sort(np.concatenate([batch_size_grid, validate_batch_size_grid]))
final_grid = np.array([[img, batch] for img in all_image_sizes for batch in all_batch_sizes])

model = LoginModel().to(device)
model = model.cuda().eval()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def measure_energy(x, n_runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
    start.record()
    with torch.inference_mode():
        for _ in range(n_runs):
            model(x)
    end.record()
    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)

    window = start.elapsed_time(end) / 1000.0
    total_j = (e1 - e0) / 1000.0
    return total_j / n_runs, total_j / window


records = []
MIN_ENERGY_WINDOW_S = 1.0

for image_size, batch in final_grid:
    image_size, batch = int(image_size), int(batch)
    is_validation = (image_size in validate_image_size_grid) or (batch in validate_batch_size_grid)
    x = None
    try:
        x = torch.randn(batch, 3, image_size, image_size, device=device)

        for _ in range(count_warmup):
            with torch.inference_mode():
                model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated()

        times = []
        for _ in range(runs_count):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            with torch.inference_mode():
                model(x)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1) / 1000.0)
        lat = float(np.median(times))

        n_energy = max(runs_count, math.ceil(MIN_ENERGY_WINDOW_S / lat))
        e_pass, power_w = measure_energy(x, n_energy)

        records.append(dict(S=image_size, B=batch,
                            latency=lat,
                            memory=mem,
                            energy=e_pass,
                            power_w=power_w,
                            energy_runs=n_energy,
                            is_validation=is_validation))
        print(f"S={image_size:3d} B={batch:3d}  lat={lat*1e3:.2f}ms  mem={mem/1e6:.0f}MB  "
              f"E={e_pass*1e3:.2f}mJ  P={power_w:.1f}W  (n={n_energy})")

    except torch.cuda.OutOfMemoryError:
        records.append(dict(S=image_size, B=batch,
                            latency=None, memory='OOM', energy=None,
                            power_w=None, energy_runs=None,
                            is_validation=is_validation))
        print(f"S={image_size:3d} B={batch:3d}  OOM")
    finally:
        del x
        torch.cuda.empty_cache()

os.makedirs('results', exist_ok=True)
df = pd.DataFrame(records)
df.to_csv('results/measurements.csv', index=False)
print('Saved results/measurements.csv')

# OOM search outside the grid: B = 256, growing S
for image_size, batch in [(s, 256) for s in (512, 640, 736, 752, 896, 1024, 1072, 1200)]:
    image_size, batch = int(image_size), int(batch)
    x = None
    try:
        x = torch.randn(batch, 3, image_size, image_size, device=device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated()
        print(f"S={image_size:4d} B={batch:3d}  mem={mem/1e6:.0f}MB  pred={memory(image_size, batch)/1e6:.0f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"S={image_size:4d} B={batch:3d}  OOM  pred={memory(image_size, batch)/1e6:.0f}MB")
    finally:
        del x
        torch.cuda.empty_cache()
