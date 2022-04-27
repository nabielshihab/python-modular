import numpy as np
import pandas as pd
import math


def round_up(x):
    """
    a function to round up a float to 1 decimal place.

    :param x (float): float to be rounded
    :return (float): rounded float
    """

    math.ceil(x*10)/10
    return math.ceil(x*10)/10


def fmd(mag, mbin):
    """
    a function to create frequency-magnitude distribution from a series of earthquake magnitudes.

    :param mag (series): a series of eartquake magnitudes
    :param mbin (float): magnitude bin width
    :return:
    m (array) : magnitude bins
    cum (array) : cumulative magnitude frequency
    noncum (array) : noncumulative magnitude frequency
    """

    m = np.arange(min(round(mag/mbin)*mbin), (max(round(mag/mbin)*mbin)+mbin), mbin)
    nbm = len(m)
    cum = np.zeros(nbm)
    for i in range(nbm):
        cum[i] = len(np.where(mag > round(m[i],1))[0])
    cumnbmagtmp = list(cum)
    cumnbmagtmp.append(0)
    noncum = abs(np.diff(cumnbmagtmp))
    return m, cum, noncum


def maxc(mag, mbin):
    """
    a function to estimate magnitude of completeness using maximum curvature method

    :param mag (series): a series of eartquake magnitudes
    :param mbin (float): magnitude bin width
    :return:
    Mc (float): magnitude of completeness

    """
    
    FMD = fmd(mag, mbin)
    m = FMD[0]
    noncum = FMD[2]
    Mc = m[np.argmax(noncum)]
    return Mc, m


def gft(mag, mbin):
    """
    a function to estimate magnitude of completeness using Goodness-of-fit test method

    :param mag (series): a series of eartquake magnitudes
    :param mbin (float): magnitude bin width
    :return:
    Mc (float): magnitude of completeness
    Best (string): best confidence level
    Mco (array): magnitude cutoff
    R (array): residual

    """
    
    FMD = fmd(mag, mbin)
    Mcbound = maxc(mag, mbin)[0]
    Mco = Mcbound - 0.4 + (np.arange(1,16,1) - 1) / 10
    Mco = [round(elem, 1) for elem in Mco]
    R = np.zeros(len(Mco))
    for i in range(len(R)):
        indmag = np.where( mag > Mco[i]- mbin/2 )
        b = math.log10(math.exp(1))/(np.mean(np.array(mag)[indmag])-(Mco[i] - mbin/2))
        a = math.log10(len(np.array(indmag)[0]))+b*Mco[i]
        fmd0 = [round(elem,1) for elem in FMD[0]]
        FMDcum_model = 10**(a-[b*elem for elem in fmd0])
        indmi = np.where(fmd0 >= Mco[i])
        R[i] = sum(sum(abs(np.reshape(FMD[1][indmi],(1,len(FMDcum_model[indmi])))-FMDcum_model[indmi])))/sum(FMD[1][indmi])*100
    indgft = np.array(list(np.where(R <= 5)))
    if (len(indgft[0]) != 0):
        Mc = Mco[indgft[0,0]]
        best = "95%"
    else:
        indgft = np.array(list(np.where(R <= 10)))
        if (len(indgft[0]) != 0):
            Mc = Mco[indgft[0,0]]
            best = "90%"
        else:
            Mc = Mcbound
            best = "MAXC"
    return Mc, best, Mco, R


def mbs(mag, mbin):
    """
    a function to estimate magnitude of completeness using Mc by b-value stability method

    :param mag (series): a series of eartquake magnitudes
    :param mbin (float): magnitude bin width
    :return:
    Mc (float): magnitude of completeness
    Mco (array): magnitude cutoff
    bi (array): b-value for each Mco
    unc (array): b-value uncertainty for each Mco
    bave (array) b-value average

    """
    
    Mcbound = round(maxc(mag, mbin)[0],1)
    Mco = Mcbound - 0.7 + (np.arange(1,21,1) - 1) / 10
    Mco = [round(elem,1) for elem in Mco]
    bi = np.zeros(20)
    unc = np.zeros(20)
    for i in range(20):
        indmag = np.where( mag > Mco[i]-mbin/2 )
        nbev = len(np.array(indmag[0]))
        bi[i] = math.log10(math.exp(1))/(np.mean(np.array(mag)[indmag])-(Mco[i]-mbin/2))
        if nbev > 1:
            unc[i] = 2.3*bi[i]**2*(sum((np.array(mag)[indmag]-np.mean(np.array(mag)[indmag]))**2)/(nbev*(nbev-1)))**(1/2)
        else: unc[i] = np.nan
    
    n = 15    
    bave = np.zeros(n)
    for i in range(n):
        bave[i] = np.mean(bi[i:i+(len(bi)-n)])
    dbi = abs(bave[0:n]-bi[0:n])
    indmbs = np.where(dbi <= unc[0:n])
    if len(list(indmbs[0])) != 0:
        Mc = Mco[list(indmbs[0])[0]]
    else: Mc = np.nan    
    return Mc, Mco, bi, unc, bave


def fmd_bvalue(mag, mc, maxmag, mbin):
    """
    a function to create frequency-magnitude distribution from a series of earthquake magnitudes with Mc as its minimum
    boundary and maxmag as its maximum boundary.

    :param mag (series): a series of eartquake magnitudes
    :param mbin (float): magnitude bin width
    :param mc (float): magnitude of completeness
    :param maxmag (float): maximum magnitude
    :return:
    m (array) : magnitude bins
    cum (array) : cumulative magnitude frequency
    y (array) : noncumulative magnitude frequency
    """
    
    x = np.arange(mc, maxmag, mbin)
    nbm = len(x)
    cum = np.zeros(nbm)
    for i in range(nbm):
        cum[i] = len(np.where(mag > round(x[i],1))[0])
    cumnbmagtmp = list(cum)
    cumnbmagtmp.append(0)
    y = abs(np.diff(cumnbmagtmp))
    return x, cum, y


def fmd_details(mag, mc_method, mbin=0.1, nbsample=200):
    """
    a function to generate FMD details including Mc and maximum magnitude that will be used for calculating b-value

    :param mag: a series of earthquake magnitudes
    :param mc: magnitude of completeness
    :param mbin: magnitude bin width
    :param nbsample: number of bootstrap sample
    :return:
    fmd_data (dict) : informations that will be used for calculating b-value and visualization
    """
    # pembentukan FMD
    FMD = fmd(mag, mbin)
    m = FMD[0]
    cum = FMD[1]
    noncum = FMD[2]

    
    # estimasi Mc
    if mc_method == 'maxc':
        mc_method = maxc
    elif mc_method == 'mbs':
        mc_method = mbs
    else:
        mc_method = gft

    mc_bootstrap = np.zeros(nbsample)
    for i in range(nbsample):
        magbs = pd.Series(np.random.choice(mag, size=len(mag)))
        mc_bootstrap[i] = mc_method(magbs, mbin)[0]

    mc_mean = np.nanmean(mc_bootstrap)
    mc_sd = np.nanstd(mc_bootstrap)
    mc_sdl = mc_mean - mc_sd
    mc_sdr = mc_mean + mc_sd

    mc_test = round_up(mc_mean)
    if mc_test <= mc_mean + mc_sd:
        mc = mc_test
    else:
        mc = round(mc_mean, 1)

    # estimasi maximum magnitude
    sorted_mag = sorted(mag, reverse=True)
    max1 = sorted_mag[0]
    for i in range(len(sorted_mag)):
        if sorted_mag[i] != max1:
            max2 = sorted_mag[i]
            break
    maxmag = round(max1 + (max1 - max2), 1)
    # maxmag = round(max1, 1)

    
    mag_bvalue = mag[(mag >= mc) & (mag <= maxmag)]
    FMD_bvalue = fmd_bvalue(mag_bvalue, mc, maxmag, mbin)
    x = FMD_bvalue[0]
    cum_bvalue = FMD_bvalue[1]
    y = FMD_bvalue[2]
    d1 = pd.DataFrame({"x": np.array(x), "y": np.array(y)})

    # membentuk variabel xo dan yo
    yo = y[list(np.where(y != 0)[0])]
    xo = x[list(np.where(y != 0)[0])]
    log_yo = [math.log10(elem) for elem in yo]
    d2 = pd.DataFrame({"xo": np.array(xo), "yo": np.array(yo), "log_yo": np.array(log_yo)})

    fmd_data = {
        'm': m,
        'noncum': noncum,
        'cum': cum,
        'mc_mean': mc_mean,
        'mc_sdl': mc_sdl,
        'mc_sdr': mc_sdr,
        'mc': mc,
        'maxmag': maxmag,
        'mag_bvalue': mag_bvalue,
        'cum_bvalue':cum_bvalue,
        'd1': d1,
        'x': x,
        'y': y,
        'd2': d2,
        'yo': yo,
        'xo': xo,
        'log_yo': log_yo
    }

    return fmd_data