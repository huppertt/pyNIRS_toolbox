from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import copy
import re
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
import matplotlib.colors as mcolors


import cedalion  # noqa: F401  (namespace + accessor registration)
from cedalion.typing import LabeledPointCloud  # noqa: F401  (documented type)
from pyBrainAnalyzIR.utils.geo_transforms import geo_rotation

from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt


def to_string(arr):
    # Convert to list of integers
    return arr.values.astype(str).tolist()


# columns of Connectivity.coef that hold values (everything else identifies a connection)
VALUE_COLUMNS = ('Coeff', 'Pvalue', 'Condition', 'dfe', 'ZstdErr')
Z_MAX = 6.0


def _fisher_z(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.arctanh(np.asarray(r, dtype=float))
    return np.clip(z, -Z_MAX, Z_MAX)


@dataclass
class Connectivity:
    """Main container for connectivity results.

    The `Connectivity` class holds connectivity adjunct objects. It can hold several connectivity
    matrices, one per condition (e.g. the different task / rest blocks of a recording when the
    connectivity is computed with ``divide_events``), and conditions can be compared with
    :meth:`ttest` (modeled on the MATLAB ``nirs.core.sFCStats``).

    Attributes:
        coef: (Pandas DataFrame) one row per (condition, connection) with the columns ``Coeff``
            (correlation R), ``Pvalue``, ``Condition``, ``Channel_to``, ``Channel_from``,
            ``Type_to``, ``Type_from``, ``geo_to``, ``geo_from`` (``Subject_to``/``Subject_from``
            for hyperscanning), ``dfe`` (degrees of freedom) and ``ZstdErr`` (standard error of
            the Fisher-Z transformed coefficient).
        geo3d (LabeledPointCloud): A labeled point cloud representing the 3D geometry of
            the recording.
        geo2d (LabeledPointCloud): A labeled point cloud representing the 2D geometry of
            the recording.
        head_model (Optional[Any]): A head model object.
        meta_data (OrderedDict[str, Any]): A dictionary of meta data.
    """

    description: str = field(default_factory=str)

    coef: pd.DataFrame = field(default_factory=pd.DataFrame)

    geo3d: OrderedDict[str, LabeledPointCloud] = field(default_factory=OrderedDict)
    geo2d: OrderedDict[str, LabeledPointCloud] = field(default_factory=OrderedDict)
    geo2d_rotation_data: OrderedDict[str, geo_rotation] = field(default_factory=OrderedDict)
    geo3d_rotation_data: OrderedDict[str, geo_rotation] = field(default_factory=OrderedDict)

    head_model: Optional[Any] = None
    meta_data: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def __repr__(self):
        """Return a string representation of the Recording object."""
        return (
            f"<Connectivity | "
            f"{self.description}, "
        )

    def __init__(self):
        self.description = "Connectivity statistics"
        self.geo2d = OrderedDict()
        self.geo3d = OrderedDict()
        self.geo2d_rotation_data = OrderedDict()
        self.geo3d_rotation_data = OrderedDict()
        self.head_model = None
        self.meta_data = OrderedDict()

        return

    def __copy__(self) -> "Connectivity":
        """Return a shallow copy of this Connectivity object."""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo=None) -> "Connectivity":
        """Return a deep copy of this Connectivity object."""
        if memo is None:
            memo = {}
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, copy.deepcopy(value, memo))
        return new

    def copy(self, deep: bool = True) -> "Connectivity":
        """Return a copy of this Connectivity object.

        Args:
            deep (bool): If True (default), return a deep copy where nested
                mutable attributes (geometries, meta data, etc.) are
                independent of the original. If False, return a shallow copy.

        Returns:
            Connectivity: The copied object.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def geo_rotated2D(self, name: str) -> LabeledPointCloud:
        """Return the rotation data for a given geometry name.

        Args:
            name (str): The name of the geometry.

        Returns:
            LabeledPointCloud: The rotated 2D geometry.

        """
        if name not in self.geo2d or name not in self.geo2d_rotation_data:
            raise ValueError(f"Geometry '{name}' not found in geo2d.")

        rotator = self.geo2d_rotation_data[name]
        geo = self.geo2d[name].copy()

        return rotator.transform(geo)

    def geo_rotated3D(self, name: str) -> LabeledPointCloud:
        """Return the rotation data for a given geometry name.

        Args:
            name (str): The name of the geometry.
        Returns:
            LabeledPointCloud: The rotated 3D geometry.
        """
        if name not in self.geo3d or name not in self.geo3d_rotation_data:
            raise ValueError(f"Geometry '{name}' not found in geo3d.")

        rotator = self.geo3d_rotation_data[name]
        geo = self.geo3d[name].copy()

        return rotator.transform(geo)

    # ------------------------------------------------------------------
    # conditions
    # ------------------------------------------------------------------

    @property
    def conditions(self):
        """Condition names, in the order they appear in ``coef``."""
        if 'Condition' not in self.coef:
            return []
        return [str(c) for c in pd.unique(self.coef['Condition'])]

    def list_conditions(self):
        return np.array(self.conditions)

    def remove_condition(self, cond_name):
        names = [cond_name] if isinstance(cond_name, str) else list(cond_name)
        self.coef = self.coef[~self.coef['Condition'].isin(names)].reset_index(drop=True)

    def keep_condition(self, cond_name):
        names = [cond_name] if isinstance(cond_name, str) else list(cond_name)
        self.coef = self.coef[self.coef['Condition'].isin(names)].reset_index(drop=True)

    def _key_columns(self):
        return [c for c in self.coef.columns if c not in VALUE_COLUMNS]

    @property
    def Z(self):
        """Fisher-Z transform of the coefficients (capped at +/-6 as in MATLAB)."""
        return pd.Series(_fisher_z(self.coef['Coeff']), index=self.coef.index, name='Z')

    def _zstderr(self):
        if 'ZstdErr' in self.coef:
            se = self.coef['ZstdErr'].astype(float).to_numpy()
        else:
            se = np.full(len(self.coef), np.nan)
        if 'dfe' in self.coef:
            # Fisher-Z standard error 1/sqrt(n-3) = 1/sqrt(dfe-1) when it was not stored
            fallback = 1.0 / np.sqrt(np.maximum(self.coef['dfe'].astype(float).to_numpy() - 1, 1))
            se = np.where(np.isnan(se), fallback, se)
        return se

    def table(self):
        """``coef`` with the Fisher Z, T-statistic (Z / ZstdErr) and FDR corrected q-values."""
        tbl = self.coef.copy()
        tbl['Z'] = _fisher_z(tbl['Coeff'])
        with np.errstate(divide='ignore', invalid='ignore'):
            tbl['Tstat'] = tbl['Z'] / self._zstderr()
        tbl['Qvalue'] = np.nan
        for cond in self.conditions:
            lst = (tbl['Condition'] == cond).to_numpy() & tbl['Pvalue'].notna().to_numpy()
            if lst.any():
                _, q, _, _ = multipletests(tbl.loc[lst, 'Pvalue'], alpha=0.05, method='fdr_bh')
                tbl.loc[lst, 'Qvalue'] = q
        return tbl

    def matrix(self, condition=None, type_to=None, type_from=None, value='Coeff'):
        """Connectivity matrix (DataFrame, rows = ``from``, columns = ``to``) for one condition.

        Args:
            condition: condition name (default: the first condition).
            type_to, type_from: data type (e.g. 'HbO'); default: the first type.
            value: column of :meth:`table` to return ('Coeff', 'Pvalue', 'Z', 'Tstat', 'Qvalue').
        """
        tbl = self.table()
        if condition is None:
            condition = self.conditions[0]
        tbl = tbl[tbl['Condition'] == condition]
        if type_to is None:
            type_to = tbl['Type_to'].iloc[0]
        if type_from is None:
            type_from = type_to
        tbl = tbl[(tbl['Type_to'] == type_to) & (tbl['Type_from'] == type_from)]
        hyper = 'Subject_to' in tbl and tbl['Subject_to'].nunique() + tbl['Subject_from'].nunique() > 2
        if hyper:
            tbl = tbl.assign(To=tbl['Subject_to'].astype(str) + ':' + tbl['Channel_to'].astype(str),
                             From=tbl['Subject_from'].astype(str) + ':' + tbl['Channel_from'].astype(str))
        else:
            tbl = tbl.assign(To=tbl['Channel_to'], From=tbl['Channel_from'])
        order_to = list(pd.unique(tbl['To']))
        order_from = list(pd.unique(tbl['From']))
        mat = tbl.pivot_table(index='From', columns='To', values=value, aggfunc='first')
        return mat.reindex(index=order_from, columns=order_to)

    # ------------------------------------------------------------------
    # contrasts / t-test
    # ------------------------------------------------------------------

    def _parse_contrast(self, text):
        """Parse e.g. ``'task - rest'`` or ``'0.5*A + 0.5*B - rest'`` into a weight vector over
        :attr:`conditions`. Condition names are matched ignoring spaces (longest name first)."""
        conds = self.conditions
        squashed = [c.replace(' ', '') for c in conds]
        order = sorted(range(len(conds)), key=lambda i: -len(squashed[i]))
        s = str(text).replace(' ', '')
        weights = np.zeros(len(conds))
        pos, found = 0, False
        number = re.compile(r'([+-]?)(\d*\.?\d+(?:[eE][+-]?\d+)?\*)?')
        while pos < len(s):
            m = number.match(s, pos)
            sign = -1.0 if m.group(1) == '-' else 1.0
            scale = float(m.group(2)[:-1]) if m.group(2) else 1.0
            if not m.group(1) and found:
                raise ValueError(f"Could not parse contrast '{text}' at '{s[pos:]}'")
            pos = m.end()
            for i in order:
                if squashed[i] and s.startswith(squashed[i], pos):
                    weights[i] += sign * scale
                    pos += len(squashed[i])
                    found = True
                    break
            else:
                raise ValueError(f"Unknown condition in contrast '{text}' at '{s[pos:]}'. "
                                 f"Available conditions: {conds}")
        if not found:
            raise ValueError(f"Empty contrast '{text}'")
        return weights

    def _contrast_matrix(self, contrast, names):
        conds = self.conditions
        if isinstance(contrast, str):
            contrast = [contrast]
        if (isinstance(contrast, (list, tuple)) and len(contrast) > 0
                and all(isinstance(c, str) for c in contrast)):
            C = np.vstack([self._parse_contrast(c) for c in contrast])
            default_names = [str(c) for c in contrast]
        else:
            C = np.atleast_2d(np.asarray(contrast, dtype=float))
            if C.shape[1] != len(conds):
                raise ValueError(f"The contrast has {C.shape[1]} columns but there are {len(conds)} "
                                 f"conditions ({conds}).")
            default_names = []
            for row in C:
                terms = [(w, c) for w, c in zip(row, conds) if w != 0]
                text = ''
                for w, c in terms:
                    sign = '-' if w < 0 else ('+' if text else '')
                    mag = '' if abs(w) == 1 else f'{abs(w):g}*'
                    text += f'{sign}{mag}{c}'
                default_names.append(text or '0')
        if names is None:
            names = default_names
        names = [names] if isinstance(names, str) else list(names)
        if len(names) != C.shape[0]:
            raise ValueError("The number of names must match the number of contrasts.")
        return C, names

    def _apply_contrast(self, contrast, b=None, names=None):
        C, names = self._contrast_matrix(contrast, names)
        b = np.zeros(C.shape[0]) if b is None else np.broadcast_to(np.asarray(b, dtype=float),
                                                                   (C.shape[0],))
        conds = self.conditions
        keys = self._key_columns()
        coef = self.coef.copy()
        coef['_Z'] = _fisher_z(coef['Coeff'])
        coef['_se'] = self._zstderr()
        coef['_dfe'] = coef['dfe'].astype(float) if 'dfe' in coef else np.nan

        # one row per connection, one column per condition
        template = coef[coef['Condition'] == conds[0]][keys].reset_index(drop=True)
        idx = pd.MultiIndex.from_frame(template.astype(object))

        def per_condition(col):
            out = np.full((len(template), len(conds)), np.nan)
            for j, cond in enumerate(conds):
                sub = coef[coef['Condition'] == cond]
                sub_idx = pd.MultiIndex.from_frame(sub[keys].astype(object))
                out[:, j] = pd.Series(sub[col].to_numpy(), index=sub_idx).reindex(idx).to_numpy()
            return out

        Z, SE, DFE = per_condition('_Z'), per_condition('_se'), per_condition('_dfe')
        R, P = per_condition('Coeff'), per_condition('Pvalue')

        frames = []
        for c, bb, name in zip(C, b, names):
            used = c != 0
            if not used.any():
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                z = Z[:, used] @ c[used] - bb
                se = np.sqrt((SE[:, used] ** 2) @ (c[used] ** 2))
                dfe = (DFE[:, used] @ np.abs(c[used])) / np.abs(c[used]).sum()
                tstat = z / se
            pval = 2 * stats.t.sf(np.abs(tstat), dfe)
            coeff = np.tanh(z)
            if used.sum() == 1 and c[used][0] == 1 and bb == 0:
                # a plain condition: keep the original estimate and p-value
                j = np.where(used)[0][0]
                coeff, pval = R[:, j], P[:, j]
            df = template.copy()
            df.insert(0, 'Coeff', coeff)
            df.insert(1, 'Pvalue', pval)
            df.insert(2, 'Condition', name)
            df['dfe'] = dfe
            df['ZstdErr'] = se
            frames.append(df[[c for c in self.coef.columns if c in df.columns]
                             + [c for c in df.columns if c not in self.coef.columns]])
        if not frames:
            raise ValueError("None of the contrasts use any of the conditions.")
        return pd.concat(frames, ignore_index=True)

    def ttest(
            self,
            contrast=None,
            b=None,
            names=None,
            condnames=None,
            types_to=None,
            type_from=None,
            geo_to=None,
            geo_from=None,
            remove_cross_types=False,
            remove_cross_geo=False,
            remove_same_types=False,
            remove_same_geo=False,
            remove_self_connections=False):
        """T-test of the null hypothesis ``contrast * Z = b`` and/or selection of connections.

        Modeled on ``nirs.core.sFCStats.ttest``: the contrast is applied to the Fisher-Z
        transformed coefficients of each connection, the standard errors are propagated
        (``ZstdErr``) and the degrees of freedom are the ``|contrast|``-weighted average of those of
        the conditions. A new Connectivity object is returned (``Coeff = tanh(Z)``).

        Args:
            contrast: None (no contrast, only the selection below), a string or list of strings
                such as ``'task - rest'`` / ``['A', 'B', 'A-B']``, or a numeric vector/matrix
                (one row per contrast, one column per entry of :attr:`conditions`).
            b: (optional) mean of the null distribution (Fisher-Z units), scalar or one per contrast.
            names: (optional) names of the new conditions (default: the contrast strings).
            condnames: list of condition names to include (applied before the contrast).
            types_to / type_from: data types to include. geo_to / geo_from: geometries to include.
            remove_cross_types / remove_cross_geo: keep only connections within a type / geometry.
            remove_same_types / remove_same_geo: keep only connections across types / geometries.
            remove_self_connections: drop the connections of a channel with itself.

        Example:
            conn.ttest('task - rest')          # compare two resting-state conditions
            conn.ttest([[1, -1], [1, 1]])      # numeric contrasts over conn.conditions
        """

        coef_pruned = self.coef.copy()
        if condnames is not None:
            coef_pruned = coef_pruned[coef_pruned['Condition'].isin(condnames)]
        if types_to is not None:
            coef_pruned = coef_pruned[coef_pruned['Type_to'].isin(types_to)]
        if type_from is not None:
            coef_pruned = coef_pruned[coef_pruned['Type_from'].isin(type_from)]
        if geo_to is not None:
            coef_pruned = coef_pruned[coef_pruned['geo_to'].isin(geo_to)]
        if geo_from is not None:
            coef_pruned = coef_pruned[coef_pruned['geo_from'].isin(geo_from)]

        if remove_cross_types:
            coef_pruned = coef_pruned[coef_pruned['Type_to'] == coef_pruned['Type_from']]
        if remove_cross_geo:
            coef_pruned = coef_pruned[coef_pruned['geo_to'] == coef_pruned['geo_from']]
        if remove_same_types:
            coef_pruned = coef_pruned[coef_pruned['Type_to'] != coef_pruned['Type_from']]
        if remove_same_geo:
            coef_pruned = coef_pruned[coef_pruned['geo_to'] != coef_pruned['geo_from']]
        if remove_self_connections:
            coef_pruned = coef_pruned[(coef_pruned['Channel_to'] != coef_pruned['Channel_from']) |
                                      (coef_pruned['Type_to'] != coef_pruned['Type_from']) |
                                      (coef_pruned['geo_to'] != coef_pruned['geo_from'])]

        coef_pruned.reset_index(drop=True, inplace=True)

        newConn = self.copy()
        newConn.coef = coef_pruned

        if contrast is not None:
            newConn.coef = newConn._apply_contrast(contrast, b, names)
            newConn.description = 'T-test'

        return newConn

    def show(
            self,
            fdr_correct_for_all=False,
            type="r",
            threshold_type='p',
            thres=0.05,
            vrange=None,
            show_cross_types=False,
            condnames=None):
        """Draw the significant connections on the 2D probe layout.

        One row of panels is drawn per condition (``condnames``; default all conditions) and one
        panel per data type (type pair with ``show_cross_types``).

        Args:
            fdr_correct_for_all: with ``threshold_type='q'``, FDR-correct over all shown panels
                rather than per panel.
            type: value that is drawn: 'r' (correlation), 'z' (Fisher Z) or 't' (Z / ZstdErr).
            threshold_type: 'p' or 'q'. thres: threshold.
            vrange: color range; default symmetric around the largest drawn value.
            show_cross_types: also draw the connections between different data types.
            condnames: conditions to draw.
        """
        coef = self.table()

        coef = coef[(coef['Channel_to'] != coef['Channel_from']) |
                    (coef['Type_to'] != coef['Type_from']) |
                    (coef['geo_to'] != coef['geo_from'])]

        if not show_cross_types:
            coef = coef[(coef.Type_to == coef.Type_from)]

        if condnames is None:
            condnames = self.conditions
        elif isinstance(condnames, str):
            condnames = [condnames]
        coef = coef[coef['Condition'].isin(condnames)].reset_index(drop=True)

        if type.lower() == "z":
            coef['Coeff'] = coef['Z']
        elif type.lower() == "t":
            coef['Coeff'] = coef['Tstat']
        elif type.lower() != "r":
            raise ValueError("type must be 'r', 'z' or 't'")

        if threshold_type.lower() == 'p':
            coef['_thr'] = coef['Pvalue']
        elif threshold_type.lower() == 'q':
            if fdr_correct_for_all:
                _, qvalue, _, _ = multipletests(coef['Pvalue'], alpha=0.05, method='fdr_bh')
                coef['_thr'] = qvalue
            else:
                coef['_thr'] = np.nan
        else:
            raise ValueError("threshold_type must be 'p' or 'q'")

        types = np.unique(np.concatenate([coef.Type_to, coef.Type_from]))
        pairs = [(t, t) for t in types] if not show_cross_types else \
            [(tto, tfrom) for tfrom in types for tto in types]
        ncols = len(types)
        nrows_per_cond = 1 if not show_cross_types else len(types)

        panels = []
        for cond in condnames:
            for typeto, typefrom in pairs:
                lst = np.where((coef.Condition == cond) & (coef.Type_to == typeto)
                               & (coef.Type_from == typefrom))[0]
                if threshold_type.lower() == 'q' and not fdr_correct_for_all and len(lst) > 0:
                    _, qvalue, _, _ = multipletests(coef.Pvalue[lst], alpha=0.05, method='fdr_bh')
                    coef.loc[lst, '_thr'] = qvalue
                panels.append((cond, typeto, typefrom, lst))

        shown = np.concatenate([lst[coef['_thr'].to_numpy()[lst] < thres] for *_, lst in panels]) \
            if panels else np.array([], dtype=int)
        if vrange is None:
            vmax = np.nanmax(np.abs(coef['Coeff'].to_numpy()[shown])) if len(shown) else 1.0
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
            vrange = [-vmax, vmax]

        cmap = mpl.colormaps['jet']
        val_range = np.array((vrange[0], vrange[1]))

        norm = mcolors.Normalize(vmin=val_range[0], vmax=val_range[1])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Important for the colorbar to work correctly

        nrows = max(1, len(condnames) * nrows_per_cond)
        fig, ax = plt.subplots(nrows, max(1, ncols), figsize=(4 * max(1, ncols), 4 * nrows),
                               squeeze=False)
        ax = ax.flatten()

        def channel_ends(geo2d, chan):
            source = chan[:chan.find("D")]
            detector = chan[chan.find("D"):]
            return (geo2d[geo2d.label == source].to_numpy(),
                    geo2d[geo2d.label == detector].to_numpy(), source, detector)

        for axIdx, (cond, typeto, typefrom, lst) in enumerate(panels):
            a = ax[axIdx]
            thr = coef['_thr'].to_numpy()
            for idx in lst[thr[lst] < thres]:
                srcpos_to, detpos_to, _, _ = channel_ends(self.geo_rotated2D(coef.geo_to[idx]),
                                                          coef.Channel_to[idx])
                srcpos_from, detpos_from, _, _ = channel_ends(self.geo_rotated2D(coef.geo_from[idx]),
                                                              coef.Channel_from[idx])
                pos_to = (srcpos_to + detpos_to) / 2
                pos_from = (srcpos_from + detpos_from) / 2

                ll, = a.plot([pos_to[0, 0], pos_from[0, 0]], [pos_to[0, 1], pos_from[0, 1]], 'k')
                ll.set_color(cmap(norm(coef.Coeff[idx])))

            optodes = []
            for geo_col, chan_col in (('geo_to', 'Channel_to'), ('geo_from', 'Channel_from')):
                for name in np.unique(self.coef[geo_col]):
                    geo2d = self.geo_rotated2D(name)
                    optodes.append(geo2d.to_numpy())
                    uchan = np.unique(self.coef[self.coef[geo_col] == name][chan_col])
                    for chan in uchan:
                        srcpos, detpos, source, detector = channel_ends(geo2d, chan)
                        a.plot([srcpos[0, 0], detpos[0, 0]], [srcpos[0, 1], detpos[0, 1]],
                               color=[0.8, 0.8, 0.8])
                        a.text(srcpos[0, 0], srcpos[0, 1], source, fontsize=12, ha='center', va='center')
                        a.text(detpos[0, 0], detpos[0, 1], detector, fontsize=12, ha='center', va='center')
            optodes = np.vstack(optodes)

            s = (optodes.max() - optodes.min()) / 10
            a.set_ylim(optodes[:, 1].min() - s, optodes[:, 1].max() + s)
            a.set_xlim(optodes[:, 0].min() - s, optodes[:, 0].max() + s)
            a.set_aspect('equal')
            a.set_axis_off()

            typename = typeto if typeto == typefrom else f"{typeto}-{typefrom}"
            a.set_title(f"{cond}: {typename}", fontsize=14)

            plt.colorbar(sm, ax=a, shrink=0.5)

        for a in ax[len(panels):]:
            a.set_axis_off()

        fig.tight_layout()
