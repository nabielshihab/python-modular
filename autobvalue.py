from library.curve_fitting_method import *
from library.visualization import *
from library.mag_of_completeness import *

# reading catalog file
file_name = 'catalogs/earthquakes.csv'
catalog = pd.read_csv(file_name)

# b-value parameters
mc_method    = 'mbs'
cv_method    = 'b-value pois'

# plotting EQ epicenters distribution
plot_eq_distribution(catalog, coor='lon/lat')

# calculating Mc & maxmag
mag             = catalog['M']
fmd_data        = fmd_details(mag, mc_method)
mc              = fmd_data['mc']
maxmag          = fmd_data['maxmag']

# b-value and its uncertainty calculation
cv_data     = generate_autobvalue(cv_method, fmd_data)
bvalue      = cv_data['bvalue']
sd_bvalue   = cv_data['unc']

# plotting FMD, Mc, maxmag, and b-value
plot_noncum_fmd(fmd_data)
plot_bvalue(cv_data, cv_method)
