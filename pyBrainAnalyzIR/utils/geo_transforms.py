import numpy as np
import pint

class geo_rotation:
    def __init__(self,translation2D:np.ndarray=np.zeros(2),rotation2D:np.ndarray=np.eye(2),flip2D:np.ndarray=np.ones(2),
                 translation3D:np.ndarray=np.zeros(3),rotation3D:np.ndarray=np.eye(3),flip3D:np.ndarray=np.ones(3)):
        
        self.translation2D = translation2D
        self.rotation2D = rotation2D
        self.flip2D = flip2D
        self.translation3D = translation3D
        self.rotation3D = rotation3D
        self.flip3D = flip3D

    def set_rotation2D(self,angles:np.ndarray):
        self.rotation2D = np.array([[np.cos(angles[0]),-np.sin(angles[0])],[np.sin(angles[0]),np.cos(angles[0])]])

    def set_rotation3D(self,angles:np.ndarray):
        self.rotation3D = np.array([[np.cos(angles[0])*np.cos(angles[1]),np.cos(angles[0])*np.sin(angles[1])*np.sin(angles[2])-np.sin(angles[0])*np.cos(angles[2]),np.cos(angles[0])*np.sin(angles[1])*np.cos(angles[2])+np.sin(angles[0])*np.sin(angles[2])],
                                    [np.sin(angles[0])*np.cos(angles[1]),np.sin(angles[0])*np.sin(angles[1])*np.sin(angles[2])+np.cos(angles[0])*np.cos(angles[2]),np.sin(angles[0])*np.sin(angles[1])*np.cos(angles[2])-np.cos(angles[0])*np.sin(angles[2])],
                                    [-np.sin(angles[1]), np.cos(angles[1])*np.sin(angles[2]), np.cos(angles[1])*np.cos(angles[2])]])

    def transform(self, geo):
        """Apply the configured rotation/translation to a 2D or 3D geo Xarray object.

        Keeps the original coordinates, indexes, and unit metadata while rotating the
        underlying point values in the last axis. Handles pint-quantified data (e.g.
        via ``.pint.quantify()``) without triggering a UnitStrippedWarning by operating
        on the magnitude and reattaching the units afterwards.
        """
        data = geo.data
        units = None
        if isinstance(data, pint.Quantity):
            units = data.units
            values = data.magnitude
        else:
            values = np.asarray(data)

        last_dim = values.shape[-1] if values.ndim > 0 else 0

        if last_dim == 2:
            transformed = (values * self.flip2D) @ self.rotation2D.T + self.translation2D
        elif last_dim == 3:
            transformed = (values * self.flip3D) @ self.rotation3D.T + self.translation3D
        else:
            raise ValueError(f"Expected a 2D or 3D geometry array, got shape {values.shape!r}")

        if units is not None:
            transformed = pint.Quantity(transformed, units)

        geoNew = geo.copy(deep=True)
        geoNew.data = transformed
        if hasattr(geo, "attrs"):
            geoNew.attrs = dict(geo.attrs)

        return geoNew

        