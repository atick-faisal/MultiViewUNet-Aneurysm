import os
import numpy as np
import pandas as pd
import pyvista as pv

from tqdm import tqdm

DATASET_PATH = "../../data/dataset/"
VIDEO_DIR = "Videos/"
GEOMETRY_DIR = "Geometry/"
CURVATURE_DIR = "Curvature/"
CFD_DIR = "Geometry_CFD_Aneurysm/"

geometries = os.listdir(os.path.join(DATASET_PATH, GEOMETRY_DIR))
cfd_results = os.listdir(os.path.join(DATASET_PATH, CFD_DIR))

for geometry in tqdm(cfd_results, desc="Processing ... "):
    geometry_path = os.path.join(
        DATASET_PATH,
        CFD_DIR,
        geometry
    )

    video_path = os.path.join(
        DATASET_PATH,
        VIDEO_DIR,
        "CFD",
        geometry[:-4] + ".avi"
    )

    mesh = pv.read(geometry_path)
    # ecap = mesh.active_scalars
    # ecap = (ecap - np.mean(ecap)) / (np.std(ecap))
    # curvature = mesh.curvature(curv_type="mean")
    # curvature = (curvature - np.mean(curvature)) \
    #     / (np.std(curvature))

    pl = pv.Plotter()
    pl.open_movie(video_path)
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        # scalars=curvature,
        cmap="jet",
        # clim=[-1, 1],
        show_scalar_bar=False
    )

    pl.write_frame()
    for i in range(360):
        mesh.rotate_z(1, inplace=True)
        pl.write_frame()

    pl.close()

    # break
