import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy


def fit_data_with_poisson(data, x, xo, log_yo):
    """
    a function to calculate b-value, its uncertainty and its 95% confidence interval 
    using GLM with Poisson residual curve fitting method (b-value pois)

    :param data (dataframe): dataframe containing x and y
    :param x (array): x
    :param xo (array): x without zero value
    :param log_yo (array): log10(y) without zero value
    :return:
    cv_data (dictionary): a dictionary containing b-value, its uncertainty, its 95% confidence interval, etc.
    """
    
    pois=smf.glm(formula='y~x',family=sm.families.Poisson(),data=data).fit()  
    bpois=round(-1*pois.params['x']/math.log(10),2)

    cipois2=round(100*(pois.conf_int().loc['x',0])/math.log(10),3)/100
    cipois4=round(100*(pois.conf_int().loc['x',1])/math.log(10),3)/100

    uncpois=round(100*(cipois4-cipois2)/2)/100 

    qhi = 0.975
    qlo = 0.025

    ppois = pois.fittedvalues
    nhpois = scipy.stats.poisson.ppf(qhi, ppois)
    nlpois = scipy.stats.poisson.ppf(qlo, ppois)
    nlpoiso = nlpois[list(np.where(np.array(nlpois) != 0)[0])]
    xonlpois = x[list(np.where(np.array(nlpois) != 0)[0])]
    nhpoiso = nhpois[list(np.where(np.array(nhpois) != 0)[0])]
    xonhpois = x[list(np.where(np.array(nhpois) != 0)[0])]

    cv_data = {
        'pois'      : pois,
        'bvalue'     : bpois,
        'cipois2'   : cipois2,
        'cipois4'   : cipois4,
        'unc'        : uncpois,
        'ppois'     : ppois,
        'nhpois'    : nhpois,
        'nlpois'    : nlpois,
        'nlpoiso'   : nlpoiso,
        'xonlpois'  : xonlpois,
        'nhpoiso'   : nhpoiso,
        'xonhpois'  : xonhpois,
        'x'         : x,
        'xo'        : xo,
        'log_yo'       : log_yo
    }
    return cv_data


def fit_data_with_gaussian_glm(data, x, xo, log_yo):
    """
    a function to calculate b-value, its uncertainty and its 95% confidence interval 
    GLM with Gaussian residual curve fitting method (b-value glin)

    :param data (dataframe): dataframe containing xo and yo
    :param x (array): x
    :param xo (array): x without zero value
    :param log_yo (array): log10(y) without zero value
    :return:
    cv_data (dictionary): a dictionary containing b-value, its uncertainty, its 95% confidence interval, etc.
    """

    glin=smf.glm(formula='yo~xo',family=sm.families.Gaussian(link=sm.genmod.families.links.log), data=data).fit()
    bglin=round(-1*glin.params['xo']/math.log(10),2)

    ciglin2=round(100*(glin.conf_int().loc['xo',0])/math.log(10),3)/100
    ciglin4=round(100*(glin.conf_int().loc['xo',1])/math.log(10),3)/100

    uncglin=round(100*(ciglin4-ciglin2)/2)/100

    qhi = 0.975
    qlo = 0.025

    pglin=glin.fittedvalues
    nhglin=scipy.stats.norm.ppf(qhi, pglin, np.std(log_yo-pglin))
    nlglin=scipy.stats.norm.ppf(qlo, pglin, np.std(log_yo-pglin))
    nlglinp = nlglin[list(np.where(np.array(nlglin)>0)[0])]
    xop = xo[tuple([list(np.where(np.array(nlglin)>0)[0])])]

    cv_data = {
        'glin'      : glin,
        'bvalue'    : bglin,
        'ciglin2'   : ciglin2,
        'ciglin4'   : ciglin4,
        'unc'        : uncglin,
        'pglin'     : pglin,
        'nhglin'    : nhglin,
        'nlglin'    : nlglin,
        'nlglinp'   : nlglinp,
        'xop'       : xop,
        'x'         : x,
        'xo'        : xo,
        'log_yo'    : log_yo
    }
    return cv_data


def fit_data_with_gaussian_lm(data, x, xo, log_yo):
    """
    a function to calculate b-value, its uncertainty and its 95% confidence interval 
    LM with Gaussian residual curve fitting method (b-value glog)

    :param data (dataframe): dataframe containing xo and log10(yo)
    :param x (array): x
    :param xo (array): x without zero value
    :param log_yo (array): log10(y) without zero value
    :return:
    cv_data (dictionary): a dictionary containing b-value, its uncertainty, its 95% confidence interval, etc.
    """
    
    glog=smf.ols(formula='log_yo~xo', data=data).fit()
    bglog=round(-1*glog.params['xo'],2)

    ciglog2=glog.conf_int().loc['xo',0]
    ciglog4=glog.conf_int().loc['xo',1]

    uncglog=round(100*(ciglog4-ciglog2)/2)/100

    qhi = 0.975
    qlo = 0.025

    pglog=glog.fittedvalues
    nhglog=scipy.stats.norm.ppf(qhi, pglog, np.std(log_yo-pglog))
    nlglog=scipy.stats.norm.ppf(qlo, pglog, np.std(log_yo-pglog))

    cv_data = {
        'glog'      : glog,
        'bvalue'     : bglog,
        'ciglog2'   : ciglog2,
        'ciglog4'   : ciglog4,
        'unc'    : uncglog,
        'pglog'     : pglog,
        'nhglog'    : nhglog,
        'nlglog'    : nlglog,
        'x'         : x,
        'xo'        : xo,
        'log_yo'       : log_yo
    }
    return cv_data


def generate_autobvalue(cv_method, fmd_data):
    """
    a function to calculate b-value, its uncertainty and its 95% confidence interval 
    based on what curve fitting method that we use.
    
    :param cv_method (string): curve fitting method. options are 'b-value-pois', 'b-value glin', and 'b-value glog'.
    :param fmd_data (dictionary): a dictionary containing x, xo, and log10(yo) and one of these dataframes:
     1. dataframe containing x and y
     2. dataframe containing x, xo, and log10(yo)
    :return:
    cv_data (dictionary): a dictionary containing b-value, its uncertainty, its 95% confidence interval, etc.
    """
    
    if cv_method == "b-value pois":
        cv_data = fit_data_with_poisson(fmd_data['d1'], fmd_data['x'], fmd_data['xo'], fmd_data['log_yo'])
    elif cv_method == "b-value glin":
        cv_data = fit_data_with_gaussian_glm(fmd_data['d2'], fmd_data['x'], fmd_data['xo'], fmd_data['log_yo'])
    elif cv_method == "b-value glog":
        cv_data = fit_data_with_gaussian_lm(fmd_data['d2'], fmd_data['x'], fmd_data['xo'], fmd_data['log_yo'])
    return cv_data
