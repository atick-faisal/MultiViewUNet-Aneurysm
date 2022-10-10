import os
import numpy as np
import pandas as pd
import pyvista as pv

from tqdm import tqdm
from matplotlib.colors import ListedColormap
from preprocessing.image_generation import TAWSS_DIR

DATASET_PATH = "../../data/dataset/"
VIDEO_DIR = "Videos/"
GEOMETRY_DIR = "Input/"
CURVATURE_DIR = "Curvature/"
CFD_DIR = "Target/"
TAWSS_DIR = "TAWSS/"

CFD_CMAP = ListedColormap(
    np.array([
        [0, 51, 251, 255],
        [5, 164, 246, 255],
        [2, 244, 255, 255],
        [1, 245, 251, 255],
        [0, 253, 198, 255],
        [4, 250, 122, 255],
        [77, 253, 1, 255],
        [177, 253, 3, 255],
        [247, 254, 1, 255],
        [255, 176, 0, 255],
        [250, 68, 3, 255]
    ], dtype=np.uint8) / 255.0
)


geometries = os.listdir(os.path.join(DATASET_PATH, GEOMETRY_DIR))
geometries = os.listdir(os.path.join(DATASET_PATH, CFD_DIR))
geometries = list(filter(lambda x: "PATIENT1_SYNTHETIC_10" in x, geometries))

for geometry in tqdm(geometries, desc="Processing ... "):
    geometry_path = os.path.join(
        DATASET_PATH,
        GEOMETRY_DIR,
        geometry
    )

    tawss_path = os.path.join(
        DATASET_PATH,
        TAWSS_DIR,
        geometry[:-4] + ".csv"
    )

    video_path = os.path.join(
        DATASET_PATH,
        VIDEO_DIR,
        "Target",
        geometry[:-4] + ".avi"
    )

    mesh = pv.read(geometry_path)

    # ecap = mesh.active_scalars
    # ecap = (ecap - np.mean(ecap)) / (np.std(ecap))
    # curvature = mesh.curvature(curv_type="mean")
    # curvature = (curvature - np.mean(curvature)) \
    #     / (np.std(curvature))

    pl = pv.Plotter()
    pl.enable_anti_aliasing()
    pl.open_movie(video_path)
    pl.set_background("white")
    pl.add_mesh(
        geometry,
        cmap=CFD_CMAP,
        show_scalar_bar=False,
        ambient=0.3,
        smooth_shading=True,
        lighting=True,
        clim=[0, 2],
    )

    pl.write_frame()
    for i in range(360):
        mesh.rotate_z(1, inplace=True)
        pl.write_frame()

    pl.close()

    break
