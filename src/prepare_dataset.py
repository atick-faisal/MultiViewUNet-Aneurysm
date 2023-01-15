import os, gc
import random
import numpy as np
import pandas as pd
import pyvista as pv

from rich.progress import track
from utils import generate_rotating_snapshots

random.seed(42)

# ---------------- Config --------------------
DATASET_PATH = "../data/dataset/"
TRAIN_DIR = "Train/"
VAL_DIR = "Val/"
TEST_DIR = "Test/"
IMAGE_DIR = "Images/"
INPUT_DIR = "Input/"
TARGET_DIR = "Target/"
ROTATION = 30
TRAIN_PERCENTAGE = 0.9
CURVATURE_CLIM = [0, 300]
# CURVATURE_CLIM = [4, 8] # LOG
TAWSS_CLIM = [0, 5]
# TAWSS_CLIM = [-3, 3] # LOG
ECAP_CLIM = [0, 2]

# --------------------- Train-Test Split  -----------------------------
geometries = os.listdir(os.path.join(DATASET_PATH, INPUT_DIR))
geometries = [filename[:-4] for filename in geometries]

geometries = geometries[150:]

# real_geometries = list(filter(lambda x: "SYNTHETIC" not in x, geometries))
# synthetic_geometries = list(filter(lambda x: "SYNTHETIC" in x, geometries))

# print(f"REAL: {real_geometries}")
# print(f"SYNTHETIC: {synthetic_geometries}")

# sys.exit(0)

# ... Train Val Test
# random.shuffle(synthetic_geometries)
# train_size = int(len(synthetic_geometries) * TRAIN_PERCENTAGE)
# train_geometries = synthetic_geometries[:train_size]
# val_geometries = synthetic_geometries[train_size:]
# test_geometries = real_geometries.copy()

# ... Train Test
random.shuffle(geometries)
train_size = int(len(geometries) * TRAIN_PERCENTAGE)
train_geometries = geometries[:train_size]
test_geometries = geometries[train_size:]
# train_geometries = synthetic_geometries
# test_geometries = real_geometries

# -------------------- Generate Dataset -------------------------
for filename in track(geometries, description="Processing ... "):
    geometry_path = os.path.join(
        DATASET_PATH,
        INPUT_DIR,
        filename + ".stl"
    )

    result_path = os.path.join(
        DATASET_PATH,
        TARGET_DIR,
        filename + ".csv"
    )

    # -------------------------- Input Geometry ----------------------------

    geometry = pv.read(geometry_path)
    curvature = geometry.curvature(curv_type="mean")
    # curvature[curvature < 0.001] = 0.001
    geometry.point_data["CURVATURE"] = curvature

    image_path = None

    if filename in test_geometries:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TEST_DIR,
            INPUT_DIR,
            filename
        )
    # elif filename in val_geometries:
    #     image_path = os.path.join(
    #         DATASET_PATH,
    #         IMAGE_DIR,
    #         VAL_DIR,
    #         INPUT_DIR,
    #         filename
    #     )
    else:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TRAIN_DIR,
            INPUT_DIR,
            filename
        )

    # --------------------- Augmentation --------------------------

    if filename in train_geometries:
        # continue
        generate_rotating_snapshots(
            geometry=geometry,
            rotation_step=ROTATION,
            rotation_axis="x",
            clim=CURVATURE_CLIM,
            save_path=image_path,
            glossy_rendering=False
        )
        generate_rotating_snapshots(
            geometry=geometry,
            rotation_step=ROTATION,
            rotation_axis="y",
            clim=CURVATURE_CLIM,
            save_path=image_path,
            glossy_rendering=False
        )

    # --------------------- Original -----------------------

    # geometry.rotate_x(90, inplace=True)

    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION,
        rotation_axis="z",
        clim=CURVATURE_CLIM,
        save_path=image_path,
        glossy_rendering=False
    )

    del geometry, curvature
    gc.collect()

    # -------------------------- Target Geometry ----------------------------

    geometry = pv.read(geometry_path)
    result = pd.read_csv(result_path)
    geometry.point_data["TAWSS"] = result["TAWSS [Pa]"]
    # geometry.point_data["ECAP"] = result["ECAP [kg^-1 ms^2]"]

    image_path = None

    if filename in test_geometries:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TEST_DIR,
            TARGET_DIR,
            filename
        )
    # elif filename in val_geometries:
    #     image_path = os.path.join(
    #         DATASET_PATH,
    #         IMAGE_DIR,
    #         VAL_DIR,
    #         TARGET_DIR,
    #         filename
    #     )
    else:
        image_path = os.path.join(
            DATASET_PATH,
            IMAGE_DIR,
            TRAIN_DIR,
            TARGET_DIR,
            filename
        )

    # --------------------- Augmentation --------------------------

    if filename in train_geometries:
        # continue
        generate_rotating_snapshots(
            geometry=geometry,
            rotation_step=ROTATION,
            rotation_axis="x",
            clim=TAWSS_CLIM,
            save_path=image_path
        )
        generate_rotating_snapshots(
            geometry=geometry,
            rotation_step=ROTATION,
            rotation_axis="y",
            clim=TAWSS_CLIM,
            save_path=image_path
        )

    # --------------------- Original -----------------------

    # geometry.rotate_x(90, inplace=True)

    generate_rotating_snapshots(
        geometry=geometry,
        rotation_step=ROTATION,
        rotation_axis="z",
        clim=TAWSS_CLIM,
        save_path=image_path
    )

    del geometry, result
    gc.collect()

    # break
