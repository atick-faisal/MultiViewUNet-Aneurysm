import numpy as np
import pyvista as pv
from PIL import Image
from typing import List
from matplotlib.colors import ListedColormap
from pyvista.core.pointset import PolyData

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


def generate_rotating_snapshots(
    geometry: PolyData,
    rotation_step: int,
    rotation_axis: str,
    clim: List[int],
    save_path: str,
    glossy_rendering: bool = False
):
    pl = pv.Plotter(off_screen=True)
    pl.enable_anti_aliasing()
    pl.set_background("white")

    if glossy_rendering:
        pl.add_mesh(
            geometry,
            cmap=CFD_CMAP,
            show_scalar_bar=False,
            pbr=True,
            metallic=0.0,
            roughness=0.05
        )
    else:
        pl.add_mesh(
            geometry,
            cmap=CFD_CMAP,
            show_scalar_bar=True,
            ambient=0.3,
            smooth_shading=True,
            lighting=True,
            clim=clim,
            scalar_bar_args=dict(height=0.2, vertical=False, color="#212121",
                                 position_x=0.2, position_y=0.05)

        )

    for i in range(1):
        if rotation_axis == "x":
            geometry.rotate_x(rotation_step, inplace=True)
        elif rotation_axis == "y":
            geometry.rotate_y(rotation_step, inplace=True)
        elif rotation_axis == "z":
            geometry.rotate_z(rotation_step, inplace=True)
        else:
            raise ValueError("Roatation axis is not correct")

        pl.show(auto_close=False)
        image = Image.fromarray(pl.image[:, 128:-128, :])
        image.save(save_path + "_{:s}_{:03d}.png".format(rotation_axis, i))

    pl.close()
