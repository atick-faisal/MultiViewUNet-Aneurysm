import os
import random
import numpy as np
import pyvista as pv

from tqdm import tqdm
from PIL import Image

random.seed(42)

DATASET_PATH = "../../data/dataset/"
TRAIN_DIR = "Train/"
TEST_DIR = "Test/"
IMAGE_DIR = "Images/"
INPUT_DIR = "Input/"
CURVATURE_DIR = "Curvature/"
TARGET_DIR = "Target/"
TRAIN_PERCENTAGE = 0.8

geometries = os.listdir(os.path.join(DATASET_PATH, INPUT_DIR))

random.shuffle(geometries)
geometries = [filename[:-4] for filename in geometries]
train_size = int(len(geometries) * TRAIN_PERCENTAGE)
train_geometries = geometries[:train_size]
test_geometries = geometries[train_size:]

print("-" * 40)
print(f"TRAIN SIZE: {len(train_geometries)}")
print(f"TEST SIZE: {len(test_geometries)}")
print("-" * 40)

for filename in tqdm(geometries, desc="Processing ... "):
    geometry_path = os.path.join(
        DATASET_PATH,
        INPUT_DIR,
        filename + ".stl"
    )

    cfd_path = os.path.join(
        DATASET_PATH,
        TARGET_DIR,
        filename + ".vtk"
    )

    # --------------------- GEOMETRY -----------------------

    geometry = pv.read(geometry_path)
    image_path = None

    if filename in train_geometries:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TRAIN_DIR,
            INPUT_DIR,
            filename[:-4]
        )
    else:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TEST_DIR,
            INPUT_DIR,
            filename[:-4]
        )

    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")
    pl.add_mesh(
        geometry,
        show_scalar_bar=False,
        smooth_shading=True
    )

    for i in range(36):
        geometry.rotate_z(10, inplace=True)
        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        image.save(image_path + "_{:03d}.jpg".format(i))

    pl.close()

    cfd = pv.read(cfd_path)

    if filename in train_geometries:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TRAIN_DIR,
            TARGET_DIR,
            filename[:-4]
        )
    else:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TEST_DIR,
            TARGET_DIR,
            filename[:-4]
        )

    pl = pv.Plotter(off_screen=True)
    pl.set_background("gray")
    pl.add_mesh(
        cfd,
        cmap="jet",
        show_scalar_bar=False,
        smooth_shading=True
    )

    for i in range(36):
        cfd.rotate_z(10, inplace=True)
        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        image.save(image_path + "_{:03d}.jpg".format(i))

    pl.close()

    # break




# -------------------------- v1 -------------------------------

# for geometry in tqdm(geometries, desc="Processing ... "):
#     geometry_path = os.path.join(
#         DATASET_PATH,
#         INPUT_DIR,
#         geometry
#     )

#     image_path = os.path.join(
#         DATASET_PATH,
#         IMAGE_DIR,
#         CURVATURE_DIR,
#         geometry[:-4]
#     )

#     mesh = pv.read(geometry_path)
#     # ecap = mesh.active_scalars
#     # ecap = (ecap - np.mean(ecap)) / (np.std(ecap))
#     curvature = mesh.curvature(curv_type="mean")
#     curvature = (curvature - np.mean(curvature)) \
#         / (np.std(curvature))

#     pl = pv.Plotter(off_screen=True)
#     pl.set_background("white")
#     pl.add_mesh(
#         mesh,
#         scalars=curvature,
#         cmap="jet",
#         clim=[-1, 1],
#         show_scalar_bar=False,
#         smooth_shading=True
#     )

#     for i in range(36):
#         mesh.rotate_z(10, inplace=True)
#         pl.show(auto_close=False)
#         image = Image.fromarray(pl.image[:, 128:-128, :])
#         image.save(image_path + "_{:03d}.jpg".format(i))

#     pl.close()

#     # break
