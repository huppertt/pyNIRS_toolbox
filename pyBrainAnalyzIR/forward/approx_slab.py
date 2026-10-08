"""Approximate (analytic) slab forward model (port of nirs.forward.ApproxSlab)."""
import numpy as np
import xarray as xr
import cedalion
import cedalion.dataclasses as cdc

from pyBrainAnalyzIR.forward.mesh import combine_meshes
from pyBrainAnalyzIR.media.spectra import getspectra

units = cedalion.units


class ApproxSlab:
    """Analytic slab forward model based on infinite-medium Green's functions.

    The sensitivity of a channel to an absorption change at node r is
    approximated by ``phiS(r) * phiD(r)`` with ``phi(r) = exp(-K r) / r``.

    Attributes:
        mesh: cedalion ``Voxels`` (or list of them) holding the node positions.
        prop: optical properties (e.g. ``pyBrainAnalyzIR.media.tissues.brain``);
            its wavelengths define the wavelengths of the model.
        geo3d: cedalion LabeledPointCloud with the optode positions.
        channel: xr.DataArray of channel labels with ``source`` and ``detector``
            coordinates (e.g. ``rec['amp'].channel``).
        Fm: modulation frequency in MHz (0 for continuous wave).
    """

    def __init__(self, mesh=None, prop=None, geo3d=None, channel=None, Fm=0.0):
        self.mesh = mesh
        self.prop = prop
        self.geo3d = geo3d
        self.channel = channel
        self.Fm = Fm

    @classmethod
    def from_recording(cls, rec, mesh, prop, ts_name="amp", Fm=0.0):
        """Build the model from the probe (geo3d + channels) of a cedalion Recording."""
        return cls(mesh=mesh, prop=prop, geo3d=rec.geo3d, channel=rec[ts_name].channel, Fm=Fm)

    # ---- helpers ------------------------------------------------------------
    @property
    def wavelengths(self):
        return np.asarray(self.prop.wavelengths, dtype=float)

    def _optode_positions(self, labels):
        geo = self.geo3d
        if hasattr(geo, "pint") and geo.pint.units is not None:
            geo = geo.pint.to("mm").pint.dequantify()
        return geo.sel(label=list(labels)).values

    def _nodes(self):
        mesh = combine_meshes(self.mesh)
        crs = self.geo3d.dims[1]
        if mesh.crs != crs:
            raise ValueError(f"mesh crs '{mesh.crs}' does not match probe crs '{crs}'")
        return mesh.voxels * (1 * mesh.units).to(units.mm).magnitude

    def _wavenumber(self):
        """Complex diffusion wavenumber K (1/mm) for each wavelength."""
        mua = self.prop.mua.pint.to("1/mm").pint.dequantify().values
        musp = self.prop.musp.pint.to("1/mm").pint.dequantify().values
        v = self.prop.v.to("mm/s").magnitude
        V = v / self.prop.ri
        D = V / (3 * (musp + mua))
        return np.sqrt(V * mua / D - 1j * (2 * np.pi * self.Fm * 1e6) / D)

    def _channel_coords(self):
        ch = self.channel
        return {
            "channel": ("channel", ch.channel.values),
            "source": ("channel", ch.source.values),
            "detector": ("channel", ch.detector.values),
            "wavelength": ("wavelength", self.wavelengths),
        }

    @staticmethod
    def _as_real(values):
        return values.real if np.all(np.isreal(values)) else values

    # ---- API ------------------------------------------------------------------
    def measurement(self):
        """Modeled baseline measurement for each channel/wavelength.

        Returns:
            xr.DataArray with dims ("channel", "wavelength").
        """
        src = self._optode_positions(self.channel.source.values)
        det = self._optode_positions(self.channel.detector.values)
        dist = np.linalg.norm(src - det, axis=1)
        K = self._wavenumber()
        meas = np.exp(-np.outer(dist, K)) / dist[:, None]
        return xr.DataArray(self._as_real(meas), dims=["channel", "wavelength"],
                            coords=self._channel_coords())

    def jacobian(self, type="standard"):
        """Sensitivity (Jacobian) of each channel to absorption at each node.

        Args:
            type: 'standard' for d(meas)/d(mua) or 'spectral' for
                d(meas)/d(HbO, HbR).

        Returns:
            xr.DataArray with dims ("channel", "vertex", "wavelength") following
            cedalion's sensitivity (Adot) layout.  For 'spectral' an extra
            "chromo" dimension (HbO, HbR) is added.
        """
        if type.lower() not in ("standard", "spectral"):
            raise ValueError("Jacobian can either be 'standard' or 'spectral'.")

        nodes = self._nodes()
        src_labels = self.channel.source.values
        det_labels = self.channel.detector.values

        usrc, isrc = np.unique(src_labels, return_inverse=True)
        udet, idet = np.unique(det_labels, return_inverse=True)
        src_r = np.linalg.norm(self._optode_positions(usrc)[:, None, :] - nodes[None], axis=2)
        det_r = np.linalg.norm(self._optode_positions(udet)[:, None, :] - nodes[None], axis=2)

        K = self._wavenumber()
        J = np.zeros((len(src_labels), nodes.shape[0], len(K)), dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            for iw, k in enumerate(K):
                phiS = np.where(src_r == 0, 1.0, np.exp(-k * src_r) / src_r)
                phiD = np.where(det_r == 0, 1.0, np.exp(-k * det_r) / det_r)
                J[:, :, iw] = phiS[isrc] * phiD[idet]
        J = self._as_real(J)

        coords = self._channel_coords()
        coords["is_brain"] = ("vertex", np.ones(nodes.shape[0], dtype=bool))
        Jda = xr.DataArray(J, dims=["channel", "vertex", "wavelength"], coords=coords)

        if type.lower() == "spectral":
            ext = getspectra(self.wavelengths).sel(chromo=["HbO", "HbR"])
            Jda = Jda * ext

        return Jda
