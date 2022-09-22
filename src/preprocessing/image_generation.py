import os
import numpy as np
import pyvista as pv

from tqdm import tqdm
from PIL import Image

DATASET_PATH = "../../data/dataset/"
IMAGE_DIR = "Images/"
GEOMETRY_DIR = "Geometry/"
CURVATURE_DIR = "Curvature/"
CFD_DIR = "Geometry_CFD_Aneurysm/"

geometries = os.listdir(os.path.join(DATASET_PATH, GEOMETRY_DIR))
cfd_results = os.listdir(os.path.join(DATASET_PATH, CFD_DIR))

for geometry in tqdm(geometries, desc="Processing ... "):
    geometry_path = os.path.join(
        DATASET_PATH,
        GEOMETRY_DIR,
        geometry
    )

    image_path = os.path.join(
        DATASET_PATH,
        IMAGE_DIR,
        CURVATURE_DIR,
        geometry[:-4]
    )

    mesh = pv.read(geometry_path)
    # ecap = mesh.active_scalars
    # ecap = (ecap - np.mean(ecap)) / (np.std(ecap))
    curvature = mesh.curvature(curv_type="mean")
    curvature = (curvature - np.mean(curvature)) \
        / (np.std(curvature))

    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")
    pl.add_mesh(
        mesh,
        scalars=curvature,
        cmap="jet",
        clim=[-1, 1],
        show_scalar_bar=False,
        smooth_shading=True
    )

    for i in range(36):
        mesh.rotate_z(10, inplace=True)
        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        image.save(image_path + "_{:03d}.jpg".format(i))

    pl.close()

    # break
