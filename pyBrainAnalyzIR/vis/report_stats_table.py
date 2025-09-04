import numpy as np
import pandas as pd

def report_stats_table(stats):

    params=stats.sm.params

    channels=params.channel.to_numpy().flatten()
    types=params.chromo.to_numpy().flatten()
    conditions=params.regressor.to_numpy().flatten()

    return pd.DataFrame({'Channels':np.matlib.repmat(channels,len(types)*len(conditions),1).T.flatten(),
            'Types':np.matlib.repmat(np.matlib.repmat(types,len(channels),1).T.flatten(),
                                        1,len(conditions)).flatten(),
                'Conditions':np.matlib.repmat(conditions,1,len(types)*len(channels)).flatten(),
                'Beta':stats.sm.params.to_numpy().flatten(),
                'StdErr':stats.sm.bse.to_numpy().flatten(),
                'T-stat':stats.sm.tvalues.to_numpy().flatten(),
                'p-value':stats.sm.__getattr__('pvalues').to_numpy().flatten(),
                'dof':np.matlib.repmat(stats.sm.__getattr__('df_resid').to_numpy().flatten(),1,len(conditions)).flatten()})
