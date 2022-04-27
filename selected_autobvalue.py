from library.curve_fitting_method import *
from library.visualization import *
from library.mag_of_completeness import *
from matplotlib import path

# reading catalog file
file_name = 'catalogs/earthquakes.csv'
catalog = pd.read_csv(file_name)

# b-value parameters
mc_method    = 'maxc'
cv_method    = 'b-value glog'

# plotting EQ epicenters distribution
plot_eq_distribution(catalog, coor='lon/lat')

# defining the polygon
poly = plt.ginput(8)
poly = np.array(poly)

# searching events that are included in the polygon
p = path.Path(poly)
p_index = list(np.where(p.contains_points(np.transpose(np.array([catalog['X'], catalog['Y']]))) == 1)[0])

# extracting event coordinates and magnitudes that are included in polygon
mag         = catalog['M'][p_index]
x_poly      = catalog['X'][p_index]
y_poly      = catalog['Y'][p_index]
z_poly      = catalog['Z'][p_index]

# plotting events that are included in polygon
ax = plt.subplot(111)
ax.scatter(x_poly, y_poly, c=z_poly, cmap='Greens', edgecolors='k', s=mag**3.5, alpha=0.8)

# calculating Mc & maxmag
fmd_data        = fmd_details(mag, mc_method)
mc              = fmd_data['mc']
maxmag          = fmd_data['maxmag']

# b-value and its uncertainty calculation
cv_data     = generate_autobvalue(cv_method, fmd_data)
bvalue      = cv_data['bvalue']
sd_bvalue   = cv_data['unc']

# writing b-value and its standard deviation in figure
text = [(128.1, -3.28)]
ax.text(text[0][0], text[0][1],
        f'b-value = {bvalue} \u00B1 {sd_bvalue}',
        style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

# plotting FMD, Mc, maxmag, and b-value
plot_noncum_fmd(fmd_data)
plot_bvalue(cv_data, cv_method)
