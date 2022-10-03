import os
import numpy as np
import pandas as pd
import pyvista as pv

from matplotlib import cm

from tqdm import tqdm

DATASET_PATH = "../../data/dataset/"
VIDEO_DIR = "Videos/"
GEOMETRY_DIR = "Input/"
CURVATURE_DIR = "Curvature/"
CFD_DIR = "Target/"
TAWSS_DIR = "TAWSS/"

geometries = os.listdir(os.path.join(DATASET_PATH, GEOMETRY_DIR))[20:]
cfd_results = os.listdir(os.path.join(DATASET_PATH, CFD_DIR))

for geometry in tqdm(geometries[117:], desc="Processing ... "):
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
        "InputMetalic",
        geometry[:-4] + ".avi"
    )

    mesh = pv.read(geometry_path)
    tawss = pd.read_csv(tawss_path, header=None).values
    # tawss = np.append(tawss, [0])
    mesh.point_data["TAWSS"] = tawss

    # ecap = mesh.active_scalars
    # ecap = (ecap - np.mean(ecap)) / (np.std(ecap))
    curvature = mesh.curvature(curv_type="mean")
    # curvature = (curvature - np.mean(curvature)) \
    #     / (np.std(curvature))

    pl = pv.Plotter()
    # pl.enable_anti_aliasing()
    pl.open_movie(video_path)
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        show_scalar_bar=False,
        smooth_shading=True,
        # scalars=tawss,
        cmap=cm.get_cmap("jet", 10),
        # split_sharp_edges=True,
        # pbr=True,
        # metallic=1.0,
        # roughness=0.5,
        clim=[0, 5]
    )

    pl.write_frame()
    for i in range(360):
        mesh.rotate_z(1, inplace=True)
        pl.write_frame()

    pl.close()

    break
