import numpy as np
import pyBrainAnalyzIR.math
from pyBrainAnalyzIR.math.corrcoef_robust import corrcoef as robust_corrcoef

import pandas as pd

from cedalion.math.ar_model import ar_filter
from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity

def compute_connectivity(data,robust=True,AR=None,stimMasks=None):


    if (hasattr(data, 'wavelength')):
        data = data.transpose('time', 'channel', 'wavelength')
    else:
        data = data.transpose('time', 'channel', 'chromo')

    data=data-data.mean("time")
    data=data-data.mean("time")

    if AR is not None and AR>0:
        data=ar_filter(data,pmax=AR)
        # d.pint.dequalify()
            
    shp = data.shape
    d2 = np.reshape(data.as_numpy().data, (shp[0], shp[1] * shp[2]))

    if stimMasks is None:
        stimName = ['rest']
        stimList = [np.arange(0, shp[0])]

    df_all = pd.DataFrame()

    for stimName, stimList in zip(stimName, stimList):

        if robust:
            r, p = robust_corrcoef(d2[stimList, :], verbose=False)
        else:
            r, p = np.corrcoef(d2[stimList, :], rowvar=False)

        nchan = shp[1]
        ntype = shp[2]

        if (hasattr(data, 'wavelength')):
            types=np.tile(data.wavelength,(nchan,1)).T.flatten()
        else:
            types=np.tile(data.chromo,(nchan,1)).T.flatten()

        chans=np.tile(data.channel,(ntype,1)).flatten()

        chan_from=[]
        chan_to=[]
        type_to=[]
        type_from=[]
        coef =[]
        pval =[]

        for i in range(0,len(chans)):
            for j in range(0,len(chans)):
                chan_from.append(chans[i])
                chan_to.append(chans[j])
                type_from.append(types[i])
                type_to.append(types[j])
                coef.append(r[i,j])
                pval.append(p[i,j])

        df=pd.DataFrame({'Coeff':coef,'Pvalue': pval,'Condition':stimName, 
                        'Channel_to': chan_to, 'Channel_from': chan_from, 
                        'Type_to': type_to, 'Type_from': type_from})
        
        df_all = pd.concat([df_all, df], ignore_index=True)

    # Compute connectivity matrix from the input data
    return df_all
