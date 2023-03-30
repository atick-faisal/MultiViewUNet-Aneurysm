import os
import shutil
import random
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
from typing import List, Tuple, Literal

from pv_utils import *

random.seed(42)

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

DATA_DIR = os.path.join(current_dir, "../../data/dataset")
GEOMETRY_DIR = "Geometry"
CFD_DIR = "CFD"
IMAGES_DIR = "Images"
TRAIN_DIR = "Train"
TEST_DIR = "Test"

GEOMETRY_TRANSFORMATIONS = [
    "Raw",
    "Curvature",
    "TAWSS",
    "ECAP",
    "OSI",
    "RRT"
]

RAW_DIR = "Raw/"
CURVATURE_DIR = "Curvature/"
TAWSS_DIR = "TAWSS/"
ECAP_DIR = "ECAP/"
OSI_DIR = "OSI/"
RRT_DIR = "RRT/"

TRAIN_PERCENTAGE = 0.9
ROTATION_STEP = 30


def get_train_test_geometries(
    geometry_files_dir: str,
    train_percentage: float
) -> Tuple[List[str], List[str]]:
    """
    This function returns two lists of geometries for training and testing.

    Args:
        geometry_files_dir (str): The directory containing geometry files.
        train_percentage (float): The percentage of geometries to use for training.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists of geometries for training and testing.
    """

    all_geometries = os.listdir(geometry_files_dir)
    all_geometries = [filename[:-4] for filename in all_geometries]

    random.shuffle(all_geometries)
    train_size = int(len(all_geometries) * train_percentage)
    train_geometries, test_geometries = \
        all_geometries[:train_size], all_geometries[train_size:]

    """ --- Stratified ---
    real_geometries = list(
        filter(lambda x: "SYNTHETIC" not in x, all_geometries))
    synthetic_geometries = list(
        filter(lambda x: "SYNTHETIC" in x, all_geometries))

    random.shuffle(real_geometries)
    random.shuffle(synthetic_geometries)
    train_size_real = int(len(real_geometries) * train_percentage)
    train_size_synthetic = int(len(synthetic_geometries) * train_percentage)

    train_geometries = real_geometries[:train_size_real] + \
        synthetic_geometries[:train_size_synthetic]

    test_geometries = real_geometries[train_size_real:] + \
        synthetic_geometries[train_size_synthetic:]
    """

    return (train_geometries, test_geometries)


def get_clim(transformation) -> List[float]:
    """
    Returns the clim values for a given transformation.

    Parameters:
        transformation (str): The name of the transformation.

    Returns:
        List[float]: The clim values for the given transformation.
    """

    if transformation == "Curvature":
        return [0.0, 150.0]
    elif transformation == "TAWSS":
        return [-1.0, 1.0]
    elif transformation == "ECAP":
        return [0.0, 2.0]
    elif transformation == "OSI":
        return [0.0, 0.5]
    elif transformation == "RRT":
        return [0.0, 10]
    else:
        return [0.0, 0.0]


def get_ambient(transformation: str) -> float:
    """
    This function returns the ambient lighting based on the transformation type.

    Args:
        transformation (str): The type of transformation.

    Returns:
        float: The ambient lighting.
    """

    if transformation == "Raw":
        return 0.1
    else:
        return 0.3


def generate_images_from_geometries(
    geometries: List[str],
    mode: Literal["train", "test"],
    transformation: str
):
    """
    This function generates images from geometries.

    Parameters:
        geometries (List[str]): A list of geometry filenames.
        mode (Literal["train", "test"]): The mode of the images to generate.
        transformation (str): The transformation to apply to the images.

    Returns:
        None
    """

    for filename in geometries:
        geometry_path = os.path.join(DATA_DIR, GEOMETRY_DIR, filename + ".stl")
        cfd_result_path = os.path.join(DATA_DIR, CFD_DIR, filename + ".csv")
        geometry = pv.read(geometry_path)

        cfd_results = pd.read_csv(cfd_result_path)
        if transformation == "Raw":
            pass
        elif transformation == "Curvature":
            curvature = geometry.curvature()
            curvature[curvature < 0.001] = 0.001
            geometry.point_data[transformation] = curvature
        elif transformation == "TAWSS":
            geometry.point_data[transformation] = np.log(cfd_results.filter(
                regex=f".*{transformation}.*"))
        else:
            geometry.point_data[transformation] = cfd_results.filter(
                regex=f".*{transformation}.*")

        save_path = None
        if (mode == "train"):
            save_path = os.path.join(
                DATA_DIR, IMAGES_DIR, TRAIN_DIR, transformation, filename
            )
        else:
            save_path = os.path.join(
                DATA_DIR, IMAGES_DIR, TEST_DIR, transformation, filename
            )

        if mode == "train":
            generate_rotating_snapshots(
                geometry=geometry,
                rotation_step=ROTATION_STEP,
                rotation_axis="x",
                clim=get_clim(transformation),
                ambient=get_ambient(transformation),
                save_path=save_path
            )
            generate_rotating_snapshots(
                geometry=geometry,
                rotation_step=ROTATION_STEP,
                rotation_axis="y",
                clim=get_clim(transformation),
                ambient=get_ambient(transformation),
                save_path=save_path
            )

        generate_rotating_snapshots(
            geometry=geometry,
            rotation_step=ROTATION_STEP,
            rotation_axis="z",
            clim=get_clim(transformation),
            ambient=get_ambient(transformation),
            save_path=save_path
        )

        yield


def clean_dir(path: str):
    try:
        shutil.rmtree(path=path)
        os.mkdir(path)
    except OSError:
        pass


if __name__ == "__main__":
    train_geometries, test_geometries = get_train_test_geometries(
        geometry_files_dir=os.path.join(DATA_DIR, GEOMETRY_DIR),
        train_percentage=TRAIN_PERCENTAGE
    )

    for transformation in GEOMETRY_TRANSFORMATIONS:
        clean_dir(os.path.join(DATA_DIR, IMAGES_DIR, TRAIN_DIR, transformation))
        clean_dir(os.path.join(DATA_DIR, IMAGES_DIR, TEST_DIR, transformation))

        train_generator = generate_images_from_geometries(
            geometries=train_geometries,
            mode="train",
            transformation=transformation
        )
        for _ in tqdm(range(len(train_geometries))):
            next(train_generator)

        test_generator = generate_images_from_geometries(
            geometries=test_geometries,
            mode="test",
            transformation=transformation
        )
        for _ in tqdm(range(len(test_geometries))):
            next(test_generator)
