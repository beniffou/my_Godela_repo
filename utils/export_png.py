# utils/export_png.py
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def PNG_exporter(stl_paths, img_path="output.png", opts=None):
    """
    Minimal STL → PNG renderer that can overlay multiple STLs.
    stl_paths: list of file paths
    opts:
      - size: (w, h) inches
      - dpi: int
      - elev, azim: view angles
      - face_colors: list of matplotlib colors or RGBA tuples, one per STL
      - alphas: list of floats in [0,1], one per STL
    """
    if opts is None:
        opts = {}

    figsize = opts.get("size", (6, 6))
    dpi = opts.get("dpi", 300)
    elev = opts.get("elev", 30)
    azim = opts.get("azim", 45)
    face_colors = opts.get("face_colors", ["tab:blue"] * len(stl_paths))
    alphas = opts.get("alphas", [1.0] * len(stl_paths))

    meshes = [mesh.Mesh.from_file(p) for p in stl_paths]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Add each mesh with its own style
    for m, fc, a in zip(meshes, face_colors, alphas):
        coll = Poly3DCollection(m.vectors, facecolors=fc, linewidths=0, alpha=a)
        ax.add_collection3d(coll)

    # Autoscale to all meshes
    all_pts = np.concatenate([m.points for m in meshes], axis=0).flatten()
    ax.auto_scale_xyz(all_pts, all_pts, all_pts)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    plt.savefig(img_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
