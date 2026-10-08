"""Simple slab geometries for the analytic forward models.

The MATLAB nirs-toolbox builds a slab as a ``nirs.core.Image`` volume and
converts it to a tetrahedral FEM mesh with iso2mesh.  The approximate
(analytic) slab forward model only uses the node positions, so here the slab
is represented directly as a regular grid of nodes using cedalion's
``Voxels`` dataclass.
"""
import numpy as np
import cedalion
import cedalion.dataclasses as cdc

units = cedalion.units


def slab_mesh(origin=(-100.0, -100.0, 0.0), dim=(2.5, 2.5, 2.5), shape=(81, 81, 21),
              crs="pos"):
    """Create a slab of nodes (port of the Slab image used by nirs-toolbox).

    Defaults match the MATLAB example: x,y from -100:2.5:100 mm and depth
    0:2.5:50 mm.

    Args:
        origin: (x, y, z) position (mm) of the first node.
        dim: node spacing (mm) along x, y, z.
        shape: number of nodes along x, y, z.
        crs: coordinate reference system name; must match the probe geo3d crs.

    Returns:
        cedalion.dataclasses.Voxels with node positions in mm.  Depth is along
        +z, so the slab surface is at z = origin[2].
    """
    axes = [o + d * np.arange(n) for o, d, n in zip(origin, dim, shape)]
    grid = np.meshgrid(*axes, indexing="ij")
    nodes = np.column_stack([g.ravel() for g in grid])
    return cdc.Voxels(nodes, crs, units.mm)


def combine_meshes(meshes):
    """Concatenate the nodes of several meshes (port of ApproxSlab.combinemesh)."""
    if isinstance(meshes, cdc.Voxels):
        return meshes
    meshes = list(meshes)
    crs = {m.crs for m in meshes}
    if len(crs) != 1:
        raise ValueError("all meshes must share the same crs")
    nodes = np.vstack([m.voxels * (1 * m.units).to(units.mm).magnitude for m in meshes])
    return cdc.Voxels(nodes, meshes[0].crs, units.mm)
