import pywt
import numpy as np


def __mad(arr):
    return np.median(np.abs(arr - np.median(arr)))


def Wavelet(data, Fs, sthresh=5, wbasis='sym8', removeScaling=True):

    if (hasattr(data, 'wavelength')):
        data = data.transpose('time', 'channel', 'wavelength')
    else:
        data = data.transpose('time', 'channel', 'chromo')

    # units=data.pint.units
    # d.pint.dequalify()
    shp = data.shape
    d = np.reshape(data.as_numpy().data, (shp[0], shp[1] * shp[2]))

    d = local_wavelet(d, Fs, sthresh=sthresh, wbasis=wbasis, removeScaling=removeScaling)

    data.data = np.reshape(d, shp)
    return data


def local_wavelet(signal, Fs, sthresh=5, wbasis='sym8', removeScaling=True):

    signalout = signal.copy()

    # max level
    for ch in range(0, signalout.shape[1]):
        y = signal[:, ch]
        n = pywt.dwt_max_level(len(y), wbasis)

        # decomposition
        coef = pywt.wavedec(y, wbasis, level=n, mode='symmetric')

        # remove lowest freq components
        if removeScaling:
            coef[0] = np.zeros(coef[0].shape)

        # thresholding
        for lev in range(1, n + 1):
            # selection
            mad_val = __mad(coef[lev]) / 0.6745
            if mad_val == 0:
                continue
            lst = np.abs(coef[lev]) / mad_val > sthresh
            coef[lev][lst] = 0

        # estimated signal
        # waverec can return one extra sample for odd-length signals
        yy = pywt.waverec(coef, wavelet=wbasis, mode='symmetric')
        signalout[:, ch] = yy[0:signalout.shape[0]]

    return signalout
