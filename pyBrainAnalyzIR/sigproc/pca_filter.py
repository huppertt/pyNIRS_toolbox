import numpy as np


def pca_filter(data, ncomp=.8, split_types=True):

    if (hasattr(data, 'wavelength')):
        data = data.transpose('time', 'channel', 'wavelength')
    else:
        data = data.transpose('time', 'channel', 'chromo')
    shp = data.shape
    units = data.pint.units

    if (split_types):
        for tp in range(0, shp[2]):
            d = np.reshape(data[:, :, tp].as_numpy().data, (shp[0], shp[1]))

            m = np.expand_dims(np.mean(d, axis=0), axis=0)
            d = d - np.ones((d.shape[0], 1)) @ m

            U, S, V = np.linalg.svd(d)
            if (ncomp >= 1):
                S[:ncomp] = 0
            else:
                ss = np.cumsum(S)
                ss = ss / np.sum(S)
                S[ss < ncomp] = 0

            U = U[:, :len(S)]
            S = np.diag(S)
            d = U @ S @ V.T
            d = d + np.ones((d.shape[0], 1)) @ m
            data.data[:, :, tp] = np.reshape(d, [shp[0], shp[1]]) * units
    else:
        d = np.reshape(data.as_numpy().data, (shp[0], shp[1] * shp[2]))

        m = np.expand_dims(np.mean(d, axis=0), axis=0)
        d = d - np.ones((d.shape[0], 1)) @ m

        U, S, V = np.linalg.svd(d)
        if (ncomp >= 1):
            S[:ncomp] = 0
        else:
            ss = np.cumsum(S)
            ss = ss / np.sum(S)
            S[ss < ncomp] = 0

        U = U[:, :len(S)]
        S = np.diag(S)
        d = U @ S @ V.T
        d = d + np.ones((d.shape[0], 1)) @ m
        data.data = np.reshape(d, shp)

    return data
