from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import copy
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.colors as mcolors
from cedalion.dataclasses.geometry import PointType


import cedalion  # noqa: F401  (namespace + accessor registration)
import cedalion.dataclasses as cdc
from cedalion.typing import LabeledPointCloud  # noqa: F401  (documented type)
from pyBrainAnalyzIR.utils.geo_transforms import geo_rotation

from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt

def to_string(arr):
    # Convert to list of integers
    return arr.values.astype(str).tolist()


@dataclass
class Connectivity: 
    """Main container for connectivity results.

    The `Connectivity` class holds connectivity adjunct objects.

    Attributes:
        coef: (Pandas DataFrame)
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

    def ttest(self,condnames=None,types_to=None,type_from=None,geo_to=None,geo_from=None,
              remove_cross_types=False, remove_cross_geo=False, remove_same_types=False, remove_same_geo=False, remove_self_connections=False):
        
        """Perform a t-test on the connectivity coefficients.

            condnames: list of condition names to include in the t-test. If None, all conditions will be included.
            types_to: list of types to include in the t-test. If None, all types will be included.
            type_from: list of types to include in the t-test. If None, all types will be included.
            geo_to: list of geometry names to include in the t-test. If None, all geometries will be included.
            geo_from: list of geometry names to include in the t-test. If None, all geometries will be included.    
        
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

        return newConn

    def show(self,fdr_correct_for_all=False,type="r",threshold_type='p',thres=0.05, vrange=None, show_cross_types=False):
        coef = self.coef.copy()

        coef = coef[(coef['Channel_to'] != coef['Channel_from']) | 
                    (coef['Type_to'] != coef['Type_from']) |
                    (coef['geo_to'] != coef['geo_from'])]
        coef.reset_index(drop=True, inplace=True)

        if not show_cross_types:
            coef = coef[(coef.Type_to == coef.Type_from)]
            coef.reset_index(drop=True, inplace=True)

        types = np.unique(np.concatenate([coef.Type_to, coef.Type_from]))

        if not show_cross_types:
            n=1
        else:
            n=len(types)

        if type.lower() == "z":
            coef['Coeff'] = np.arctanh(coef['Coeff'])

        if vrange is None:
            vrange = [-coef['Coeff'].abs().max(), coef['Coeff'].abs().max()]

        cmap = mpl.colormaps['jet']
        val_range = np.array((vrange[0], vrange[1]))
        
        norm = mcolors.Normalize(vmin=val_range[0], vmax=val_range[1])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Important for the colorbar to work correctly

        fig, ax = plt.subplots(n,len(types), figsize=(8, 4*len(types)))

        if len(types) == 1:
            ax = [ax]
        ax = ax.flatten()

        axIdx=0
        for typefrom in types:
            for typeto in types:
                lst = np.where((coef.Type_to == typeto) & (coef.Type_from == typefrom))
                
                if(len(lst[0]) > 0):

                    if threshold_type.lower() == 'p':
                        pvalue = coef.Pvalue[lst[0]]
                        lst = (lst[0][pvalue < thres],)
                    elif threshold_type.lower() == 'q':
                        if fdr_correct_for_all:
                            _, qvalue, _, _ = multipletests(coef.Pvalue, alpha=0.05, method='fdr_bh')
                            qvalue = qvalue[lst[0]]
                            lst = (lst[0][qvalue < thres],)
                        else:
                            pvalues = coef.Pvalue[lst[0]]
                            _, qvalue, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
                            lst = (lst[0][qvalue < thres],)
                    else:
                        raise ValueError("threshold_type must be 'p' or 'q'")    
                    
                    for idx in lst[0]:

                        chan_to = coef.Channel_to[idx]
                        source = chan_to[:chan_to.find("D")]
                        detector = chan_to[chan_to.find("D"):]
                        geo2d = self.geo_rotated2D(self.coef.geo_to[idx])
                        srcpos_to = geo2d[geo2d.label == source].to_numpy()
                        detpos_to = geo2d[geo2d.label == detector].to_numpy()

                        geo2d = self.geo_rotated2D(self.coef.geo_from[idx])
                        chan_from = coef.Channel_from[idx]
                        source = chan_from[:chan_from.find("D")]
                        detector = chan_from[chan_from.find("D"):]
                        srcpos_from = geo2d[geo2d.label == source].to_numpy()
                        detpos_from = geo2d[geo2d.label == detector].to_numpy()

                        pos_to = (srcpos_to + detpos_to) / 2
                        pos_from = (srcpos_from + detpos_from) / 2
                        
                        ll, = ax[axIdx].plot([pos_to[0,0], pos_from[0, 0]], [pos_to[0, 1], pos_from[0, 1]],'k')
                        ll.set_color(cmap(norm(coef.Coeff[idx])))


                    optodes=[]

                    geonames = np.unique(self.coef.geo_to)
                    for name in geonames:
                        geo2d = self.geo_rotated2D(name)
                        if len(optodes)==0:
                            optodes = geo2d.to_numpy()
                        else:
                            optodes = np.vstack((optodes, geo2d.to_numpy()))

                        uchan=np.unique(self.coef.Channel_to)
                        for chan in uchan:
                            source = chan[:chan .find("D")]
                            detector = chan[chan .find("D"):]

                            srcpos = geo2d[geo2d.label == source].to_numpy()
                            detpos = geo2d[geo2d.label == detector].to_numpy()
                            ll, = ax[axIdx].plot([srcpos[0, 0], detpos[0, 0]], [srcpos[0, 1], detpos[0, 1]],color=[0.8,0.8,0.8])
                            ax[axIdx].text(srcpos[0, 0], srcpos[0, 1], source, fontsize=12, ha='center', va='center')
                            ax[axIdx].text(detpos[0, 0], detpos[0, 1], detector, fontsize=12, ha='center', va='center')


                    geonames = np.unique(self.coef.geo_from)
                    for name in geonames:
                        geo2d = self.geo_rotated2D(name)    
                        if len(optodes)==0:
                            optodes = geo2d.to_numpy()
                        else:
                            optodes = np.vstack((optodes, geo2d.to_numpy()))

                        uchan=np.unique(self.coef.Channel_from)
                        for chan in uchan:
                            source = chan[:chan .find("D")]
                            detector = chan[chan .find("D"):]

                            srcpos = geo2d[geo2d.label == source].to_numpy()
                            detpos = geo2d[geo2d.label == detector].to_numpy()
                            ll, = ax[axIdx].plot([srcpos[0, 0], detpos[0, 0]], [srcpos[0, 1], detpos[0, 1]],color=[0.8,0.8,0.8])
                            ax[axIdx].text(srcpos[0, 0], srcpos[0, 1], source, fontsize=12, ha='center', va='center')
                            ax[axIdx].text(detpos[0, 0], detpos[0, 1], detector, fontsize=12, ha='center', va='center')


                    s = (optodes.max() - optodes.min()) / 10
                    ax[axIdx].set_ylim(optodes[:, 1].min() - s, optodes[:, 1].max() + s)
                    ax[axIdx].set_xlim(optodes[:, 0].min() - s, optodes[:, 0].max() + s)
                    ax[axIdx].set_axis_off()
                    
                    if typeto == typefrom:
                        ax[axIdx].set_title(f"Connectivity for type: {typeto}", fontsize=16)
                    else:
                        ax[axIdx].set_title(f"Connectivity for type: {typeto}-{typefrom}", fontsize=16)

                    plt.colorbar(sm, ax=ax[axIdx], shrink=0.5)

                    axIdx += 1
            
            
        