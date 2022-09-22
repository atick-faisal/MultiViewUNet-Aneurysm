# %%
import os
import cv2
from turtle import color
import numpy as np
import pandas as pd
import pyvista as pv

# %%
DATASET_PATH = "../../data/dataset/"
VIDEO_DIR = "Videos/"
GEOMETRY_DIR = "Geometry/"
CFD_DIR = "Geometry_CFD_Aneurysm/"


# %%
geometries = os.listdir(os.path.join(DATASET_PATH, GEOMETRY_DIR))
cfd_results = os.listdir(os.path.join(DATASET_PATH, CFD_DIR))

# %%
geometry = geometries[7][:-4]
geometry_path = os.path.join(DATASET_PATH, GEOMETRY_DIR, geometry + ".stl")
cfd_path = os.path.join(DATASET_PATH, CFD_DIR, geometry + ".vtk")
geometry_video_path = os.path.join(DATASET_PATH, VIDEO_DIR, geometry + ".avi")


# %%
mesh = pv.read(cfd_path)
print(mesh.active_scalars.shape)


# curvature = mesh.curvature()
# curvature = (curvature - np.mean(curvature)) / (np.std(curvature))
# # %%
# pl = pv.Plotter()
# pl.open_movie(geometry_video_path)
# pl.set_background("white")
# # pl.camera_position = 'yz'
# pl.add_mesh(mesh, scalars=curvature, cmap="jet",
#             clim=[-1, 1], show_scalar_bar=False)
# pl.write_frame()
# for i in range(360):
#     mesh.rotate_z(1, inplace=True)
#     cv2.imwrite("../../data/dataset/pp.png", pl.image[:, 128:-128, :])
#     # pl.write_frame()

# pl.close()

# %%
