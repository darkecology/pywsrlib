import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d, RegularGridInterpolator

import pyart

import boto3
import urllib

import matplotlib.pyplot as plt
import matplotlib.colors as pltc

import warnings
from more_itertools.more import always_iterable
import os.path
import tempfile
import re

def aws_parse(name):
    '''
    Parse AWS key into constituent parts

    s = aws_parse(name)
    
    Parameters
    ----------
    name: string
        The name part of a key, e.g., KBGM20170421_025222 or KBGM20170421_025222_V06 
        or KBGM20170421_025222_V06.gz
        
    Returns
    -------
    s: dict
        A dictionary with fields: station, year, month, day, hour, minute, second. 


    See Also
    --------
    aws_key

    Note: the suffix (e.g., '_V06' or '_V06.gz') is deduced from the portion
    of the key that is given and may not be the actual file suffix. 
    '''

    name = os.path.basename(name)
    name, ext = os.path.splitext(name)

    if not re.match('[A-Z]{4}[0-9]{8}_[0-9]{6}.*', name):
        raise ValueError(f'invalid key: {name}')
        
    # example: KBGM20170421_025222
    return {
        'station': name[0:4],
        'year':    int(name[4:8]),
        'month':   int(name[8:10]),
        'day':     int(name[10:12]),
        'hour':    int(name[13:15]),
        'minute':  int(name[15:17]),
        'second':  int(name[17:19]),
        'suffix':  name[19:] + ext
    }

def aws_key(s, suffix=''):

    '''
    Get key for scan

    key, path, name = aws_key(s, suffix)

    Parameters
    ----------
    s: string or struct
        The short name, e.g., KBGM20170421_025222. This can also be a
        dictionary returned by aws_parse
    suffix: string
        Optionally append this to the returned name and key

    Returns
    -------
    key: string
        The full key, e.g., 2017/04/21/KBGM/KBGM20170421_025222
    path: string
        The path, e.g., 2017/04/21/KBGM
    name: string
        The name, e.g., KBGM20170421_025222
    
    See Also
    --------
    aws_parse
    '''
    
    if isinstance(s, str):
        s = aws_parse(s)

    path = '%4d/%02d/%02d/%s' % (s['year'], 
                                 s['month'], 
                                 s['day'], 
                                 s['station'])
    
    name = '%s%04d%02d%02d_%02d%02d%02d' % (s['station'], 
                                            s['year'], 
                                            s['month'], 
                                            s['day'], 
                                            s['hour'], 
                                            s['minute'], 
                                            s['second']);
    

    suff = suffix or s['suffix']
    
    key = '%s/%s%s' % (path, name, suff)
    
    return key

    
def prefix2key(bucket, prefix):
    """
    Map prefix to a unique object
    
    Returns first match if there are multiple

    Parameters
    ----------
    bucket: string
        The bucket
    prefix: string
        The object prefix
        
    Returns
    -------
    obj: string
        The name of the object
    """
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket = bucket,
                                  Prefix = prefix,
                                  MaxKeys = 1)
        
    key = None

    try:
        for obj in response["Contents"]:
            key = obj["Key"]
    except KeyError:
        raise KeyError(f'AWS key with prefix {prefix} in bucket {bucket} not found')
        
    return key

def get_s3(name, localfile=None):
    if localfile is None:
        localfile = os.path.basename(name) 
    with open(localfile, 'wb') as f:
        get_s3_fileobj(name, f)
    
def get_s3_fileobj(name, fileobj):
    bucket = 'noaa-nexrad-level2'
    key = aws_key(name)
    key = prefix2key(bucket, key)
    boto3.client('s3').download_fileobj(bucket, key, fileobj)

def read_s3(key, fun=pyart.io.read_nexrad_archive, **kwargs):
    with tempfile.NamedTemporaryFile() as temp:
        get_s3_fileobj(key, temp)
        radar = fun(temp.name, **kwargs)    
    return radar

def read_http(name, fun=pyart.io.read_nexrad_archive, **kwargs):
    with tempfile.NamedTemporaryFile() as temp:
        key = aws_key(name)
        key = prefix2key('noaa-nexrad-level2', key)
        url = f"http://noaa-nexrad-level2.s3.amazonaws.com/{key}"
        urllib.request.urlretrieve(url, temp.name)
        radar = fun(temp.name, **kwargs)    
    return radar

def db(x):

    ''' 
    Compute decibel transform

    dbx = db( x )

    dbz = 10.*log10(z)
    '''

    return 10 * np.log10(x)

def idb(dbx):
    '''
    Inverse decibel (convert from decibels to linear units)

    x = idb( dbx )

    x = 10**(dbx/10)
    '''
    return 10 ** (dbx/10)


def z_to_refl( z, wavelength=0.1071):
    '''
    Convert reflectivity factor (Z) to reflectivity (eta)
    
    eta, db_eta = z_to_refl( z, wavelength )
    
    Parameters
    ----------
    z: array
        Vector of Z values (reflectivity factor; units: mm^6/m^3)
    wavelength: scalar
        Radar wavelength (units: meters; default = 0.1071 )

    Returns
    -------
    eta: vector
        Reflectivity values (units: cm^2/km^3 )
    db_eta: vector
        Decibels of eta (10.^(eta/10))
        
    See Also
    --------
    refl_to_z

    Reference: 
      Chilson, P. B., W. F. Frick, P. M. Stepanian, J. R. Shipley, T. H. Kunz, 
      and J. F. Kelly. 2012. Estimating animal densities in the aerosphere 
      using weather radar: To Z or not to Z? Ecosphere 3(8):72. 
      http://dx.doi.org/10.1890/ ES12-00027.1


    UNITS
        Z units = mm^6 / m^3   
                = 1e-18 m^6 / m^3
                = 1e-18 m^3

        lambda units = m

        eta units = cm^2 / km^3  
                  = 1e-4 m^2 / 1e9 m^3 
                  = 1e-13 m^-1

    Equation is

               lambda^4
       Z_e = -------------- eta    (units 1e-18 m^3)
              pi^5 |K_m|^2


              pi^5 |K_m|^2
       eta = -------------- Z_e    (units 1e-13 m^-1)
               lambda^4
    '''
    

    K_m_squared = 0.93
    log_eta = np.log10(z) + 5*np.log10(np.pi) + np.log10(K_m_squared) - 4*np.log10(wavelength)


    '''
    Current units: Z / lambda^4 = 1e-18 m^3 / 1 m^4 
                                = 1e-18 m^3 / 1 m^4
                                = 1e-18 m^-1
                                
    Divide by 10^5 to get units 1e-13
    '''
    
    log_eta = log_eta - 5 # Divide by 10^5

    db_eta = 10*log_eta
    eta    = 10**(log_eta)
    
    return eta, db_eta


def refl_to_z(eta, wavelength=0.1071):
    
    '''    
    Convert reflectivity (eta) to reflectivity factor (Z)
    
    z, dbz = refl_to_z( eta, wavelength )
    
    Parameters
    ----------
    eta: vector
        Reflectivity values (units: cm^2/km^3 )
    wavelength: scalar
        Radar wavelength (units: meters; default = 0.1071 )

    Returns
    -------
    z: array
        Vector of Z values (reflectivity factor; units: mm^6/m^3)
    dbz: vector
        Decibels of z (10.^(z/10))

    For details of conversion see refl_to_z documentation

    See Also
    --------
    refl_to_z
    '''
    
    K_m_squared = 0.93

    log_z = np.log10(eta) + 4*np.log10(wavelength) - 5*np.log10(np.pi) - np.log10(K_m_squared)

    '''
    Current units: eta * lambda^4 = 1e-13 m^-1 * 1 m^4 
                                  = 1e-13 m^3 
    Multiply by 10^5 to get units 1e-18
    '''
    
    log_z = log_z + 5 # Multiply by 10^5

    dbz = 10*log_z
    z   = 10**(log_z)
    
    return z, dbz

    
def cart2pol(x, y):
    '''
    Convert from Cartesian coordinates to polar coordinate

    theta, rho = cart2pol( x, y)

    Parameters
    ----------
    x, y: array-like
        Horizontal coordinate and vertical coordinate

    Returns
    -------
    theta, rho: array-like 
        Input arrays: angle in radians, distance from origin

    See Also
    --------
    pol2cart    
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    '''Convert from polar coordinate to Cartesian coordinates

    Parameters
    ----------
    theta, rho: array-like 
        Input arrays: angle in radians, distance from origin

    Returns
    -------
    x, y: array-like
        Horizontal coordinate and vertical coordinate

    See Also
    --------
    cart2pol
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def pol2cmp( theta ):
    '''Convert from mathematical angle to compass bearing

    Parameters
    ----------
    theta: array-like
        angle in radians counter-clockwise from positive x-axis

    Returns
    -------
    bearing: array-like
        angle in degrees clockwise from north

    See Also
    --------
    cmp2pol
    '''
    bearing = np.rad2deg(np.pi/2 - theta)
    bearing = np.mod(bearing, 360)
    return bearing

def cmp2pol(bearing):
    '''Convert from compass bearing to mathematical angle

    Parameters
    ----------
    bearing: array-like
        Angle measured in degrees clockwise from north

    Returns
    -------
    theta: array-like
        angle in radians counter-clockwise from positive x-axis

    See Also
    --------
    pol2cmp
    '''
    theta = np.deg2rad(90 - bearing)
    theta = np.mod(theta, 2*np.pi)
    return theta



def slant2ground( r, theta ):
    '''
    Convert from slant range and elevation to ground range and height.
    
    Parameters
    ----------
    r: array
        Range along radar path in m
    theta: array
        elevation angle in degrees
    
    Returns
    -------
    s: array
        Range along ground (great circle distance) in m
    h: array
        Height above earth in m

    Uses spherical earth with radius 6371.2 km
    
    From Doviak and Zrnic 1993 Eqs. (2.28b) and (2.28c)
    
    See also
    https://github.com/deeplycloudy/lmatools/blob/master/lmatools/coordinateSystems.py#L258
    pyart.core.antenna_to_cartesian
    '''
    
    earth_radius = 6371200.0           # from NARR GRIB file
    multiplier = 4.0 / 3.0

    r_e = earth_radius * multiplier    # earth effective radius

    theta = np.deg2rad(theta)          # convert to radians

    z = np.sqrt( r**2 + r_e**2 + (2 * r_e * r * np.sin(theta))) - r_e
    s = r_e * np.arcsin( r * np.cos(theta) / ( r_e + z ) )
    
    return s, z


def ground2slant(s, h):
    '''
    Convert from slant range and elevation to ground range and height.
    
    r: array
        Range along radar path in m

    Parameters
    ----------
    s: array
        Range along ground (m, great circle distance)
    h: array
        height above earth (m)

    Returns
    -------
    r: array
        range along radar path (m)
    thet: array
        elevation angle in degrees


    Uses spherical earth with radius 6371.2 km
    
    From Doviak and Zrnic 1993 Eqs. (2.28b) and (2.28c)
    
    See also https://github.com/deeplycloudy/lmatools/blob/master/lmatools/coordinateSystems.py#L286
    '''

    earth_radius = 6371200.0           # from NARR GRIB file
    multiplier = 4.0 / 3.0

    r_e = earth_radius * multiplier    # earth effective radius

    
    '''
    Law of cosines of triangle ABC
    
    A = center of earth
    B = radar station
    C = pulse volume
    
    d(A,B) = r_e
    d(B,C) = r
    d(A,C) = r_e + h
    thet(AB,AC) = s/r_e
    
    If the elevation angle is zero, the ray BC is exactly perpendicular to
    AB, so ABC is a right triangle.
    
    If the elevation angle is positive, ABC is obtuse.
    
    If the elevation angle is negative, ABC is acute. 
    
    We can use this to detect negative elevation angles.
    '''

    r  = np.sqrt(r_e**2 + (r_e+h)**2 - 2*(r_e+h) * r_e * np.cos(s/r_e))
    thet = np.arccos((r_e + h) * np.sin(s/r_e) / r)
    thet = np.array(np.rad2deg(thet))
    
    is_acute = (r_e + h)**2 < r_e**2 + r**2
    
    thet = np.array(thet)
    thet[is_acute] *= -1
    thet = thet + 0.  # this converts it back from array to scalar if it was originally scalar
    
    return r, thet


def xyz2radar( x, y, z ):
    '''
    Convert (x, y, z) to (elev, rng, az)
    
    Parameters
    ----------
    x: array
        x coordinates in meters from radar
    y: array
        y coordinates in meters from radar
    z: array
        z coordinates in meters above radar
        
    Returns
    -------
    elev: array
        elevation angle in degrees
    rng: array
        range along radar path in metersx
    az: array
        azimuth in degrees
    '''

    # Get azimuth and ground (great circle) distance
    az, s = cart2pol(x, y)
    az = pol2cmp(az)
    
    # Now get slant range of each pixel on this elevation
    rng, elev = ground2slant(s, z)

    return elev, rng, az    
    

def radar2xyz( elev, rng, az ):
    '''
    Convert (elev, rng, az) to (x, y, z)

    Parameters
    ----------
    elev: array
        elevation angle in degrees
    rng: array
        range along radar path in meters
    az: array
        azimuth in degrees

    Returns
    ----------
    x: array
        x coordinates in meters from radar
    y: array
        y coordinates in meters from radar
    z: array
        z coordinates in meters above radar

    '''
    
    ground_range, z = slant2ground(rng, elev)
    phi = cmp2pol(az)
    x, y = pol2cart(phi, ground_range)
    return x, y, z

    
def get_unambiguous_range(self, sweep, check_uniform=True):
    """
    Return the unambiguous range in meters for a given sweep.

    Raises a LookupError if the unambiguous range is not available, an
    Exception is raised if the velocities are not uniform in the sweep
    unless check_uniform is set to False.

    Parameters
    ----------
    sweep : int
        Sweep number to retrieve data for, 0 based.
    check_uniform : bool
        True to check to perform a check on the unambiguous range that
        they are uniform in the sweep, False will skip this check and
        return the velocity of the first ray in the sweep.

    Returns
    -------
    unambiguous_range : float
        Scalar containing the unambiguous in m/s for a given sweep.

    """
    s = self.get_slice(sweep)
    try:
        unambiguous_range = self.instrument_parameters['unambiguous_range']['data'][s]
    except:
        raise LookupError('unambiguous range unavailable')
    if check_uniform:
        if np.any(unambiguous_range != unambiguous_range[0]):
            raise Exception('Nyquist velocities are not uniform in sweep')
    return float(unambiguous_range[0])


# Get unique sweeps

def get_tilts(radar):
    tilts = radar.fixed_angle['data']
    unique_tilts = np.unique(tilts)
    return tilts, unique_tilts


def get_sweeps(radar, field):

    tilts, unique_tilts = get_tilts(radar)

    arrays = [radar.get_field(i, field) for i in range(len(tilts))]
    has_data = [not np.all(a.mask) for a in arrays]
    rng = radar.range['data']

    sweeps = []

    for i, tilt in enumerate(unique_tilts):

        matches = np.nonzero((tilts == tilt) & has_data)[0]
        if matches.size == 0:
            continue

        '''Use nyquist velocity to obtain a single sweep for one tilt'''
        j = 0 
        if matches.size > 1:
            nyq_vels = [radar.get_nyquist_vel(i, check_uniform=False) for i in matches]

            # non-Doppler fields: pick the one with smallest prf
            if field in ['total_power',
                        'reflectivity', 
                        'differential_reflectivity',
                        'cross_correlation_ratio',
                        'differential_phase']: 

                j = matches[np.argmin(nyq_vels)]

            # Doppler fields: pick the one with largest prf
            elif field in ['velocity', 
                        'spectrum_width']:

                j = matches[np.argmax(nyq_vels)]

            else:
                raise ValueError("Invalid field")
        
        else:
            j = matches[0]

        elev = radar.get_elevation(j)
        az = radar.get_azimuth(j)

        data = radar.get_field(j, field)
        
        # Convert to regular numpy array filled with NaNs
        data = np.ma.filled(data, fill_value=np.nan)
        
        # Sort by azimuth
        I = np.argsort(az)
        az = az[I]
        elev = elev[I]
        data = data[I,:]

        sweep = {
                'data': data,
                'az': az,
                'rng': rng,
                'elev': elev,
                'fixed_angle': tilt,
                'sweepnum': j
            }
        
        if radar.instrument_parameters is not None:
            unambiguous_range = get_unambiguous_range(radar, j, check_uniform=False) # not a class method
            sweep['unambiguous_range'] = unambiguous_range
        
        sweeps.append(sweep)

    return sweeps


def get_volumes(radar, field='reflectivity', coords='antenna'):
    '''
    Get all sample volumes in a vector, along with coordinates
    
    x1, x2, x3, data = get_volumes(radar, field)
    
    Parameters
    ----------
    radar: Radar
        The Py-ART radar object representing the volume scan
    field: string
        Which field to get, e.g., 'reflectivity'
    coords: string
        Return coordinate system ('antenna' | 'cartesian' | 'geographic')
        
    Returns
    -------
    x1, x2, x3: array
        Coordinate arrays for each sample volume in specified coordinate system
    data: array
        Measurements for requested field for each sample volume    
    
    Dimension orders are:
        antenna:    range, azimuth, elevation
        cartesian:  x, y, z
        geographic: lon, lat, z
    '''
    
    sweeps = get_sweeps(radar, field)

    n = len(sweeps)
    
    X1 = [None] * n
    X2 = [None] * n
    X3 = [None] * n
    DATA = [None] * n    
    
    for j, sweep in enumerate(sweeps):

        DATA[j] = sweep['data']
        
        sweepnum = sweep['sweepnum']
        
        if coords=='antenna':
            elev = radar.get_elevation(sweepnum)
            az = radar.get_azimuth(sweepnum)

            # Dimension order is (az, range). Keep this order and ask
            # meshgrid to use 'ij' indexing
            AZ, RNG = np.meshgrid(sweep['az'], sweep['rng'], indexing='ij')
            ELEV = np.full_like(DATA[j], sweep['elev'].reshape(-1,1))
            
            X1[j], X2[j], X3[j] = RNG, AZ, ELEV            
        elif coords=='cartesian':
            X, Y, Z = radar.get_gate_x_y_z(sweepnum)
            X1[j], X2[j], X3[j] = X, Y, Z
        elif coords=='geographic':
            LAT, LON, ALT = radar.get_gate_lat_lon_alt(sweepnum)            
            X1[j], X2[j], X3[j] = LON, LAT, ALT
        else:
            raise ValueError('Unrecognized coordinate system: %s' % (coords))
    
        if X1[j].size != DATA[j].size:
            raise ValueError()

    
    concat = lambda X: np.concatenate([x.ravel() for x in X])
    
    X1 = concat(X1)
    X2 = concat(X2)
    X3 = concat(X3)
    DATA = concat(DATA)
    
    return X1, X2, X3, DATA



def radarInterpolant( data, az, rng, method="nearest"):
        
    m, n = data.shape
    
    I = np.argsort(az)
    az = az[I]
    data = data[I,:]

    # Replicate first and last radials on opposite ends of array
    # to correctly handle wrapping
    az   = np.hstack((az[-1]-360, az, az[0]+360))
    
    data = np.vstack((data[-1,:],
                      data,
                      data[0,:]))
    
    # Ensure strict monotonicity
    delta = np.hstack((0, np.diff(az)))   # difference between previous and this
    
    az = az + np.where(delta==0, 0.001, 0.0)  # add small amount to each azimuth that
                                              #  is the same as predecessor
    
    # Create interpolating function
    return RegularGridInterpolator((az, rng), data, 
                                   method=method,
                                   bounds_error=False,
                                   fill_value=np.nan)


def radarVolumeInterpolant(data, elev, rng, az, elev_buffer=0.25, method="nearest"):
        
    I = np.argsort(az)
    az = az[I]
    data = data.copy()[:,:,I]

    # Replicate first and last radials on opposite ends of array
    # to correctly handle wrapping
    az   = np.hstack((az[-1]-360, az, az[0]+360))
            
    data = np.concatenate((data[:,:,-1,None],
                           data,
                           data[:,:,0,None]), axis=2)
        
    # Ensure strict monotonicity
    delta = np.hstack((0, np.diff(az)))   # difference between previous and this
    
    az = az + np.where(delta==0, 0.001, 0.0)  # add small amount to each azimuth that
                                              #  is the same as predecessor

    # Replicate first and last elevations offset by elev_buffer to 
    # allow limited extrapolation
    elev = np.hstack((elev[0]-elev_buffer, elev, elev[-1]+elev_buffer))
    data = np.concatenate((data[None,0,:,:],
                           data,
                           data[None,-1,:,:]))
        
    # Create interpolating function
    return RegularGridInterpolator((elev, rng, az), data, 
                                   method=method,
                                   bounds_error=False,
                                   fill_value=np.nan)


class Grid():
    
    def __init__(self, grid):
        self.grid = grid

    def coords(self):
        return [np.linspace(*g) for g in self.grid]

    def points(self, indexing='xy'):
        return np.stack(np.meshgrid(*self.coords(), indexing=indexing)) 

    def shape(self):
        return np.array([g[2] for g in self.grid])

    def size(self):
        return np.prod([g[2] for g in self.grid])


def radar2mat(radars,
              axis=1,
              as_dict=False, 
              **kwargs):
    '''Render one or more radar files as a 4d array'''
    
    #from IPython.core.debugger import set_trace; set_trace()

    radars = always_iterable(radars)
    results = [radar2mat_single(r, **kwargs) for r in radars]
        
    data = np.concatenate([r[0] for r in results], axis=axis) 
    coords = [r[1:] for r in results]
    
    # coords[i] = (fields, elev, y, x) or (fields, elev, range, azimuth)
    
    combined_coords = list(coords[0])
    combined_coords[axis] = np.concatenate([r[axis] for r in coords])

    if as_dict:
        fields = coords[0][0]
        data = {f: v for f, v in zip(fields, data)}
    
    return (data,) + tuple(combined_coords)
    


VALID_FIELDS = ['reflectivity',
                'velocity',
                'spectrum_width',
                'differential_reflectivity',
                'cross_correlation_ratio',
                'differential_phase']


def radar2mat_single(radar,
                     fields = None,
                     coords = 'polar',
                     r_min  = 2125.0,     # default: first range bin of WSR-88D
                     r_max  = 459875.0,   # default: last range bin
                     r_res  = 250,        # default: super-res gate spacing
                     az_res = 0.5,        # default: super-res azimuth resolution
                     dim    = 600,        # num pixels on a side in Cartesian rendering
                     sweeps = None,
                     elevs  = None,
                     ydirection = 'xy',
                     use_ground_range = True,
                     interp_method='nearest',
                     max_interp_dist = 1.0):
    
    '''Render a single radar file as a 3d array'''
    
    '''
    Input parsing and checking
    '''    
        
    if ydirection not in ['xy', 'ij']:
        raise ValueError(f"Invalid ydirection {ydirection}. Must be 'xy' or 'ij'")
    

    # Get available fields
    available_fields = [f for f in VALID_FIELDS if f in radar.fields]
    
    # Assemble list of fields to render, with error checking
    if fields is None:
        fields = available_fields
        
    elif isinstance(fields, (list, np.array)):
        
        fields = np.array(fields) # convert to numpy array
        
        valid     = np.in1d(fields, VALID_FIELDS)
        available = np.in1d(fields, available_fields)

        if not(np.all(valid)):
            raise ValueError("fields %s are not valid" % (fields[valid != True]))

        if not(np.all(available)):
            warnings.warn("requested fields %s were not available" % (fields[available != True]))
        
        fields = fields[available]
        
    else:
        raise ValueError("fields must be None or a list")
        
    '''Get all sweeps for each field'''
    sweepdata = {f: get_sweeps(radar, f) for f in fields}
    
    '''Get list of requested elevation angles'''
    if elevs is not None:
        requested_elevs = elevs   # user requested elevation angles
    else:
        # all available elevations (for first field)
        first_field = fields[0]
        requested_elevs = np.array([s['fixed_angle'] for s in sweepdata[first_field]])
        
        # subselect by sweep index if requested
        if sweeps is not None:
            requested_elevs = requested_elevs[sweeps]
    
    
    '''Select sweeps for each field as close as possible
       desired elevation angles'''
    selected_elevs = dict()
    for f in fields:
        available_elevs = np.array([s['fixed_angle'] for s in sweepdata[f]])
        
        # available and requested are the same, no interpolation needed
        if (len(available_elevs) == len(requested_elevs) and
                np.allclose(available_elevs, requested_elevs)):

            selected_elevs[f] = available_elevs
            
        # Use interp1d to map requested elevation to nearest available elevation
        else:
            inds = np.arange(len(available_elevs))
            elev2ind = interp1d(available_elevs, inds, kind='nearest', fill_value="extrapolate")
            sweeps = np.array(elev2ind(requested_elevs).astype(int))
            selected_elevs[f] = available_elevs[sweeps]

            # Quality check
            interp_dist = np.abs(selected_elevs[f] - requested_elevs)
            if np.any(interp_dist > max_interp_dist):
                raise ValueError('Failed to match at least one requested elevation')

            # Subselect the sweeps
            sweepdata[f] = [sweepdata[f][i] for i in sweeps]
 
        
    '''
    Construct coordinate matrices PHI, R for query points
    '''    
    if coords == 'polar':
        # Query points
        r   = np.arange(r_min, r_max + r_res, r_res)
        phi = np.arange(0., 360, az_res)
        PHI, R = np.meshgrid(phi, r)
        
        # Coordinates of three dimensions in output array
        x1 = selected_elevs[fields[0]]  # use actual elevations of first field
        x2 = r
        x3 = phi
 
    elif coords == 'cartesian':
        x = y = np.linspace (-r_max, r_max, dim)
        if ydirection == 'ij':
            y = np.flip(y)

        [X, Y] = np.meshgrid(x, y)
        [PHI, R] = cart2pol(X, Y)
        PHI = pol2cmp(PHI)  # convert from radians to compass heading
        
        # Coordinates of three dimensions in output array
        x1 = selected_elevs[fields[0]] # use actual elevations of first field
        x2 = y
        x3 = x
        
    else:
        raise ValueError("inavlid coords: %s" % (coords))
    
    from collections import OrderedDict
    
    '''
    Build the output 3D arrays
    ''' 
    data = list()   
    
    m,n = PHI.shape
    
    for f in fields:
    
        nsweeps = len(sweepdata[f])    

        fdata = np.empty((nsweeps, m, n))
        
        for i, sweep in enumerate(sweepdata[f]):
                        
            az = sweep['az']
            rng = sweep['rng']

            if use_ground_range:
                rng, _ = slant2ground(rng, sweep['fixed_angle'])
            
            F = radarInterpolant(sweep['data'], az, rng, method=interp_method)

            fdata[i,:,:] = F((PHI, R))
            
        data.append(fdata)
    
    data = np.stack(data)
    
    return data, fields, x1, x2, x3


NORMALIZERS = {
        'reflectivity':              pltc.Normalize(vmin=  -5, vmax= 35),
        'velocity':                  pltc.Normalize(vmin= -15, vmax= 15),
        'spectrum_width':            pltc.Normalize(vmin=   0, vmax= 10),
        'differential_reflectivity': pltc.Normalize(vmin=  -4, vmax=  8),
        'differential_phase':        pltc.Normalize(vmin=   0, vmax=250),
        'cross_correlation_ratio':   pltc.Normalize(vmin=   0, vmax=  1.1)
}

def volume_mosaic(data, fields):
    '''
    Convert 4d radar data array into mosaic

    Parameters
    ----------
    data: array
        Four-dimensional data array from radar2mat, with shape
        (n_fields, n_elevs, h, w)
    fields: list
        list of field names
    
    Returns
    -------
    mosaic: array
        An image mosaic showing the entire radar scan, with rows
        corresponding to elevations, and columns to radar fields.
        The array represents and RGB image and has shape
        (n_elevs * h, n_fields * w, 3)

    '''
    rgb = list()

    for i, f in enumerate(fields):
        cm = plt.get_cmap(pyart.config.get_field_colormap(f))
        norm = NORMALIZERS[f]
        imdata = ma.masked_invalid(data[i])
        imdata = norm(imdata)
        imdata = np.flip(imdata, 1)
        rgb.append(cm(imdata))

    mosaic = np.hstack([np.vstack(im) for im in rgb])

    return mosaic
