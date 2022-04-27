import matplotlib.pyplot as plt
import math
from matplotlib import cm


def plot_eq_distribution(catalog, coor='utm'):
    """
    plotting earthquake distributions (epicenters)

    :param catalog (dataframe): seismic catalog containing earthquake locations (x,y,z) and magnitudes (m)
    :param coor (str): coordinate system, it is only used for labelling x and y axes.
    options are 'lon/lat' or 'utm'.
    :return: none
    """

    x = catalog['X']
    y = catalog['Y']
    z = catalog['Z']
    m = catalog['M']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = cm.colors.Normalize(vmin=min(z), vmax=max(z))
    cmap = cm.Oranges

    ax.scatter(x, y, c = z, cmap= cmap, edgecolors = 'k', s=m**3.5, alpha=0.8)
    fig.colorbar(cm.ScalarMappable(cmap = cmap, norm = norm), ax = ax, label = 'Depth')
    ax.set_title('Earthquake Distributions')
    if coor == 'lon/lat':
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    elif coor == 'utm':
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
    fig.show()
    
    return fig, ax


def plot_bvalue(data, cv_method):
    """
    plotting log(10)frequency-magnitude distribution with the fitted line and its 95% confidence interval.
    it also tells us the b-value and its uncertainty based on what curve fitting method that we use.

    :param data (dict): a dictionary containing log(10)frequency, magnitudes, fitted line and its 95% confidence interval,
     b-value and its uncertainty.
    :param cv_method (str): curve fitting method. options are 'b-value pois', 'b-value glin', or 'b-value glog'
    :return: none
    """

    if cv_method == 'b-value pois':
        plt.figure()
        plt.scatter(data['xo'],data['log_yo'],color='black', marker = ".",label='data')
        plt.plot(data['x'],[math.log10(elem) for elem in data['ppois']], color='blue', label=f"Pois b = {data['bvalue']} \u00B1 {data['unc']}", linewidth = 0.7)
        plt.plot(data['xonhpois'],[math.log10(elem) for elem in data['nhpoiso']], linestyle='--', color='blue', label='95% CI', linewidth = 0.7)
        plt.plot(data['xonlpois'],[math.log10(elem) for elem in data['nlpoiso']],linestyle='--', color='blue', linewidth = 0.7)
        plt.xlabel("Magnitude (Mw)")
        plt.ylabel("log10(Frequency)")
        plt.legend()
        plt.grid(axis='x')
        plt.ylim(-0.1)

    elif cv_method == 'b-value glin':
        plt.figure()
        plt.scatter(data['xo'],data['log_yo'],color='black',marker='.', label='data')
        plt.plot(data['xo'],[math.log10(elem) for elem in data['pglin']],color='green',lw = 0.7,label=f"glin b = {data['bvalue']} \u00B1 {data['unc']}")
        plt.plot(data['xo'],[math.log10(elem) for elem in data['nhglin']],linestyle='--',lw = 0.7, color='green', label='95% CI')
        plt.plot(data['xop'],[math.log10(elem) for elem in data['nlglinp']],linestyle='--', color='green',lw = 0.7)
        plt.xlabel("Magnitude (Mw)")
        plt.ylabel("log10(Frequency)")
        plt.legend()
        plt.grid(axis='x')
        plt.ylim(-0.1)

    elif cv_method == 'b-value glog':
        plt.figure()
        plt.scatter(data['xo'],data['log_yo'],color='black',label='data', marker='.')
        plt.plot(data['xo'],data['pglog'],color='red',label=f"glog b = {data['bvalue']} \u00B1 {data['unc']}",lw = 0.7)
        plt.plot(data['xo'],data['nhglog'],linestyle='--', color='red', label='95% CI',lw = 0.7)
        plt.plot(data['xo'],data['nlglog'],linestyle='--', color='red',lw = 0.7)
        plt.xlabel("Magnitude (Mw)")
        plt.ylabel("log10(Frequency)")
        plt.legend()
        plt.grid(axis='x')
        plt.ylim(-0.1)

    plt.show()

def plot_noncum_fmd(data):
    """
    plotting noncumulative frequency-magnitude distribution, mean of Mc bootstrap, std of Mc bootstrap, fixed Mc,
    and maximum magnitude that is used for estimating b-value

    :param data (dict): a dictionary containing mean of Mc bootstrap, std of Mc bootstrap, fixed Mc, and maximum magnitude
    :return: none
    """

    plt.figure()
    plt.scatter(data['m'], data['noncum'], marker = ".", color = "k", label = "noncumulative FMD")
    plt.axvline(data['mc_mean'], linestyle = "solid", color = "green", linewidth = 0.7, label = "Mc mean")
    plt.axvline(data['mc_sdl'], linestyle = "dashed", color = "green", linewidth = 0.7, label = "sd Mc")
    plt.axvline(data['mc_sdr'], linestyle = "dashed", color = "green", linewidth = 0.7)
    plt.axvline(data['mc'], linestyle = "solid", color = "blue", linewidth = 0.7, label = "Mc ")
    plt.axvline(data['maxmag'], linestyle = "solid", color = "red", linewidth = 0.7, label = "Max magnitude")
    plt.title("Frequency-Magnitude Distribution")
    plt.xlabel("Magnitude (Mw)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()
