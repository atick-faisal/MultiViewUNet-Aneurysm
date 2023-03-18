import pyvista as pv

geometry = pv.read("../data/dataset/Input/PATIENT20_SYNTHETIC_1.stl")

pl = pv.Plotter()
pl.enable_anti_aliasing()
pl.set_background("white")

pl.add_mesh(
    geometry,
    show_scalar_bar=False,
    # lighting=True
    # lighting='three lights',
    # pbr=True,
    # metallic=0.01,
    # roughness=0.0
)

pl.show()