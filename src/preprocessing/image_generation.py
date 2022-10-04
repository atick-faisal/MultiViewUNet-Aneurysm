import os
import sys
import random
import numpy as np
import pandas as pd
import pyvista as pv

from tqdm import tqdm
from PIL import Image
from matplotlib import cm

random.seed(42)

DATASET_PATH = "../../data/dataset/"
TRAIN_DIR = "Train/"
TEST_DIR = "Test/"
IMAGE_DIR = "Images/"
INPUT_DIR = "Input/"
CURVATURE_DIR = "Curvature/"
TARGET_DIR = "Target/"
TAWSS_DIR = "TAWSS/"
ROTATION = 30
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

    tawss_path = os.path.join(
        DATASET_PATH,
        TAWSS_DIR,
        filename + ".csv"
    )

    # ======================== GEOMETRY ========================

    geometry = pv.read(geometry_path)
    curvature = geometry.curvature(curv_type="mean")
    # curvature = (curvature - np.mean(curvature)) \
    #     / (np.std(curvature))
    geometry.point_data["Curvature"] = curvature

    # print(np.min(curvature))
    # print(np.max(curvature))

    image_path = None

    if filename in train_geometries:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TRAIN_DIR,
            INPUT_DIR,
            filename
        )
    else:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TEST_DIR,
            INPUT_DIR,
            filename
        )

    # ------------------- AUGMENTATION ----------------------------

    if filename in train_geometries:
        pl = pv.Plotter(off_screen=True)
        pl.enable_anti_aliasing()
        pl.set_background("white")
        pl.add_mesh(
            geometry,
            show_scalar_bar=False,
            smooth_shading=True,
            cmap=cm.get_cmap("jet", 10),
            # split_sharp_edges=True,
            # pbr=True,
            # metallic=1.0,
            # roughness=0.5,
            clim=[0, 300]
        )

        for i in range(360 // ROTATION):
            geometry.rotate_x(ROTATION, inplace=True)
            pl.show(auto_close=False)
            image = Image.fromarray(pl.image[:, 128:-128, :])
            image.save(image_path + "_x_{:03d}.jpg".format(i))

        pl.close()

        pl = pv.Plotter(off_screen=True)
        pl.enable_anti_aliasing()
        pl.set_background("white")
        pl.add_mesh(
            geometry,
            show_scalar_bar=False,
            smooth_shading=True,
            cmap=cm.get_cmap("jet", 10),
            # split_sharp_edges=True,
            # pbr=True,
            # metallic=1.0,
            # roughness=0.5,
            clim=[0, 300]
        )

        for i in range(360 // ROTATION):
            geometry.rotate_y(ROTATION, inplace=True)
            pl.show(auto_close=False)
            image = Image.fromarray(pl.image[:, 128:-128, :])
            image.save(image_path + "_y_{:03d}.jpg".format(i))

        pl.close()

    # --------------------- ORIGINAL -------------------------

    pl = pv.Plotter(off_screen=True)
    pl.enable_anti_aliasing()
    pl.set_background("white")
    pl.add_mesh(
        geometry,
        show_scalar_bar=False,
        smooth_shading=True,
        cmap=cm.get_cmap("jet", 10),
        # split_sharp_edges=True,
        # pbr=True,
        # metallic=1.0,
        # roughness=0.5,
        clim=[0, 300]
    )

    for i in range(360 // (ROTATION // 3)):
        geometry.rotate_z(ROTATION // 3, inplace=True)
        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        image.save(image_path + "_z_{:03d}.jpg".format(i))

    pl.close()

    # ========================= CFD =========================

    geometry = pv.read(cfd_path)
    # geometry = pv.read(geometry_path)
    # tawss = pd.read_csv(tawss_path, header=None).values
    # tawss = (tawss - np.mean(tawss)) \
    #     / (np.std(tawss))

    if filename in train_geometries:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TRAIN_DIR,
            TARGET_DIR,
            filename
        )
    else:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TEST_DIR,
            TARGET_DIR,
            filename
        )

    # ------------------- AUGMENTATION ------------------------

    if filename in train_geometries:
        pl = pv.Plotter(off_screen=True)
        # pl.enable_anti_aliasing()
        pl.set_background("white")
        pl.disable_shadows()
        pl.add_mesh(
            # cfd,
            geometry,
            cmap=cm.get_cmap("rainbow", 10),
            show_scalar_bar=False,
            # smooth_shading=True,
            # scalars=tawss,
            clim=[0, 2]
        )

        for i in range(360 // ROTATION):
            geometry.rotate_x(ROTATION, inplace=True)
            pl.show(auto_close=False)
            image = Image.fromarray(pl.image[:, 128:-128, :])
            image.save(image_path + "_x_{:03d}.jpg".format(i))

        pl.close()

        pl = pv.Plotter(off_screen=True)
        # pl.enable_anti_aliasing()
        pl.set_background("white")
        pl.disable_shadows()
        pl.add_mesh(
            geometry,
            cmap=cm.get_cmap("rainbow", 10),
            show_scalar_bar=False,
            # smooth_shading=True,
            # scalars=tawss,
            clim=[0, 2]
        )

        for i in range(360 // ROTATION):
            geometry.rotate_y(ROTATION, inplace=True)
            pl.show(auto_close=False)
            image = Image.fromarray(pl.image[:, 128:-128, :])
            image.save(image_path + "_y_{:03d}.jpg".format(i))

        pl.close()

    # -------------------- ORIGINAL ---------------------------

    pl = pv.Plotter(off_screen=True)
    # pl.enable_anti_aliasing()
    pl.set_background("white")
    pl.disable_shadows()
    pl.add_mesh(
        geometry,
        cmap=cm.get_cmap("rainbow", 10),
        show_scalar_bar=False,
        # smooth_shading=True,
        # scalars=tawss,
        clim=[0, 2]
    )

    for i in range(360 // (ROTATION // 3)):
        geometry.rotate_z(ROTATION // 3, inplace=True)
        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        image.save(image_path + "_z_{:03d}.jpg".format(i))

    pl.close()

    break


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
