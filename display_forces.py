#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

H5_PATH = "3D_fan_dataset.h5"
MAX_SAMPLES = 19

OUT_DIR = Path("/home/ubuntu/fan_CFD_dataset/visualization_results_matplotlib")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_datasets(h5):
    d = {}
    def visit(_, obj):
        if isinstance(obj, h5py.Dataset):
            d[obj.name] = obj
    h5.visititems(visit)
    return d

def pick_field(dsets, names):
    for name in names:
        for k in dsets:
            if k.endswith("/" + name) or k == name:
                return np.asarray(dsets[k][...])
    return None

def plot_obj(obj_group, obj_name):
    dsets = collect_datasets(obj_group)

    X = pick_field(dsets, ["X", "Coordinates/X"])
    Y = pick_field(dsets, ["Y", "Coordinates/Y"])
    Z = pick_field(dsets, ["Z", "Coordinates/Z"])
    u = pick_field(dsets, ["solution_u"])
    v = pick_field(dsets, ["solution_v"])
    w = pick_field(dsets, ["solution_w"])

    if any(x is None for x in [X, Y, Z, u, v, w]):
        print(f"Missing fields in {obj_name}, skipping")
        return

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    U = np.sqrt(w.reshape(-1)**2)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X, Y, Z, c=U, s=3, vmin = 0, vmax = 4)
    cb = plt.colorbar(p, ax=ax)
    cb.set_label("|U|")
    ax.set_box_aspect([1,1,1])
    ax.set_title(obj_name)

    out_file = OUT_DIR / f"{obj_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {out_file}")

def main():
    with h5py.File(H5_PATH, "r") as h5:
        if "fan" not in h5:
            raise RuntimeError("expected /fan group inside 3D_fan_dataset.h5")

        g = h5["fan"]
        obj_names = [k for k in g.keys() if k.startswith("object_")]
        if not obj_names:
            raise RuntimeError("no fan/object_* groups found")

        sample = random.sample(obj_names, min(MAX_SAMPLES, len(obj_names)))
        print("Plotting:", sample)

        for name in sample:
            plot_obj(g[name], name)

if __name__ == "__main__":
    main()
