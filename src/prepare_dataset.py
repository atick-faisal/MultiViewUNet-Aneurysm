import os
import random
import pandas as pd
import pyvista as pv

from rich.progress import track
from utils import generate_rotating_snapshots

random.seed(42)

# ---------------- Config --------------------
DATASET_PATH = "../data/dataset/"
TRAIN_DIR = "Train/"
TEST_DIR = "Test/"
IMAGE_DIR = "Images/"
INPUT_DIR = "Input/"
CURVATURE_DIR = "Curvature/"
TARGET_DIR = "Target/"
TAWSS_DIR = "TAWSS/"
ROTATION = 30
TRAIN_PERCENTAGE = 0.8
CURVATURE_CLIM = [0, 300]
TAWSS_CLIM = [0, 5]
ECAP_CLIM = [0, 2]

# --------------------- Train-Test Split  -----------------------------
geometries = os.listdir(os.path.join(DATASET_PATH, INPUT_DIR))
geometries = [filename[:-4] for filename in geometries]
# geometries = list(filter(lambda x: "PATIENT1_SYNTHETIC_10" in x, geometries))

random.shuffle(geometries)
train_size = int(len(geometries) * TRAIN_PERCENTAGE)
train_geometries = geometries[:train_size]
test_geometries = geometries[train_size:]

# -------------------- Generate Dataset -------------------------
for filename in track(geometries, description="Processing ... "):
    geometry_path = os.path.join(
        DATASET_PATH,
        INPUT_DIR,
        filename + ".stl"
    )

    vtk_path = os.path.join(
        DATASET_PATH,
        TARGET_DIR,
        filename + ".vtk"
    )

    result_path = os.path.join(
        DATASET_PATH,
        TAWSS_DIR,
        filename + ".csv"
    )

    # -------------------------- Input Geometry ----------------------------

    geometry = pv.read(geometry_path)
    curvature = geometry.curvature(curv_type="mean")
    geometry.point_data["CURVATURE"] = curvature

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

    # # --------------------- Augmentation --------------------------

    # if filename in train_geometries:
    #     generate_rotating_snapshots(
    #         geometry=geometry,
    #         rotation_step=ROTATION,
    #         rotation_axis="x",
    #         clim=CURVATURE_CLIM,
    #         save_path=image_path,
    #         glossy_rendering=True
    #     )
    #     generate_rotating_snapshots(
    #         geometry=geometry,
    #         rotation_step=ROTATION,
    #         rotation_axis="y",
    #         clim=CURVATURE_CLIM,
    #         save_path=image_path,
    #         glossy_rendering=True
    #     )

    # # --------------------- Original -----------------------

    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION,
        rotation_axis="z",
        clim=CURVATURE_CLIM,
        save_path=image_path,
        glossy_rendering=False
    )

    # -------------------------- Target Geometry ----------------------------

    geometry = pv.read(vtk_path)
    result = pd.read_csv(result_path, header=None).values
    result = geometry.active_scalars
    geometry.point_data["TAWSS"] = result

    image_path = None

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

    # --------------------- Augmentation --------------------------

    # if filename in train_geometries:
    #     generate_rotating_snapshots(
    #         geometry=geometry,
    #         rotation_step=ROTATION,
    #         rotation_axis="x",
    #         clim=TAWSS_CLIM,
    #         save_path=image_path
    #     )
    #     generate_rotating_snapshots(
    #         geometry=geometry,
    #         rotation_step=ROTATION,
    #         rotation_axis="y",
    #         clim=TAWSS_CLIM,
    #         save_path=image_path
    #     )

    # --------------------- Original -----------------------

    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION,
        rotation_axis="z",
        clim=TAWSS_CLIM,
        save_path=image_path
    )

    # break
