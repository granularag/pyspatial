import ctypes
cimport cython
import numpy as np
cimport numpy as np
from libc.stdint cimport uintptr_t
from libc.math cimport sin, cos, exp, sqrt, floor

try:
    from shapely.coords import required
except:
    from shapely.speedups._speedups import required

cdef extern from "math.h":
    double sin(double theta)
    double cos(double theta)
    double tan(double theta)
    double acos(double x)
    double asin(double x)
    double floor(double x)
    int floor(double x)

import math

cdef inline double pi(): return math.pi
cdef inline double radians(double x): return PI*x/180.
cdef inline double degrees(double x): return 180.*x/PI
cdef PI = pi()


DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t


cdef inline float to_pixel(float a, float A, float a_px_size):
    return (a - A)/a_px_size

@cython.boundscheck(False)
def create_image_array(np.ndarray[DTYPE_t, ndim=2] rast,
                       np.ndarray[DTYPE_t, ndim=2] colors):

    cdef:
        unsigned int xsize = rast.shape[0]
        unsigned int ysize = rast.shape[1]
        unsigned int i, j, c
        np.ndarray[DTYPE_t, ndim=3] img = np.zeros([xsize, ysize, 4], dtype=DTYPE)

    for i in xrange(xsize):
        for j in xrange(ysize):
            img[i, j, :] = colors[rast[i,j]]
    return img

def to_pixels(float lon, float lat, float minLon, float maxLat,
              float lon_px_size, float lat_px_size):
    lon_px = to_pixel(lon, minLon, lon_px_size)
    lat_px = to_pixel(lat, maxLat, lat_px_size)
    return lon_px, lat_px

def grid_for_pixel(int grid_size, np.int_t x, np.int_t y):
    x_grid = x - x % grid_size
    y_grid = y - y % grid_size
    return (x_grid, y_grid)


def sub(tup, float minx, float miny):
    return (tup[0] - minx, tup[1] - miny)


@cython.boundscheck(False)
cpdef adjust_coords(geom, float minx, float miny):
    ob = required(geom)
    array = ob.__array_interface__

    cdef:
         unsigned int xsize = array['shape'][0]
         np.ndarray[np.float64_t, ndim=2] res = np.zeros([xsize, 2], dtype=np.float64)
         unsigned int sm, sn, n, dx

    n = 2
    dx = 0
    if array.get('strides', None):
        sm = array['strides'][0]/sizeof(dx)
        sn = array['strides'][1]/sizeof(dx)
    else:
        sm = n
        sn = 1

    # Make pointer to the coordinate array
    if isinstance(array['data'], ctypes.Array):
        cp = <double *><uintptr_t>ctypes.addressof(array['data'])
    else:
        cp = <double *><uintptr_t>array['data'][0]

    #print xsize, array["shape"][1]

    for i in xrange(xsize):
        res[i, 0] = cp[sm*i] - minx
        res[i, 1] = cp[sm*i+sn] - miny

    return res

cdef double K0 = 0.9996
cdef double E = 0.00669438
cdef double E2 = E * E
cdef double E3 = E2 * E
cdef double E_P2 = E / (1.0 - E)
cdef double SQRT_E = sqrt(1 - E)
cdef double _E = (1 - SQRT_E) / (1 + SQRT_E)
cdef double _E2 = _E * _E
cdef double _E3 = _E2 * _E
cdef double _E4 = _E3 * _E
cdef double _E5 = _E4  * _E
cdef double M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
cdef double M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
cdef double M3 = (15 * E2 / 256 + 45 * E3 / 1024)
cdef double M4 = (35 * E3 / 3072)
cdef double P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
cdef double P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
cdef double P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
cdef double P5 = (1097. / 512 * _E4)
cdef double R = 6378137


cdef int latlon_to_zone_number(double latitude, double longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return floor((longitude + 180.) / 6.) + 1

cdef double zone_number_to_central_longitude(int zone_number):
    return (zone_number - 1) * 6 - 180 + 3


cdef struct LatLon:
    double lat
    double lon


@cython.overflowcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef LatLon from_latlon(double latitude, double longitude):

    lat_rad = radians(latitude)
    lat_sin = sin(lat_rad)
    lat_cos = cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    zone_number = latlon_to_zone_number(latitude, longitude)

    lon_rad = radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = radians(central_lon)

    n = R / sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * sin(2 * lat_rad) +
             M3 * sin(4 * lat_rad) -
             M4 * sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if latitude < 0:
        northing += 10000000

    return LatLon(easting, northing)

# Note, had to elevation because shapely can give coordinates
# in 2 or 3 dimensions when using ops.transform
def to_utm(np.float64_t lon, np.float64_t lat, ele=None):
    """Compute the coordinates in UTM, will ignore elevation"""
    latlon = from_latlon(lat, lon)
    return latlon.lon, latlon.lat

def haversine(tuple coord1, tuple coord2):
    """Given two (lng, lat) tuples, returns the distance between them in
    meters."""
    cdef double lat1
    cdef double lng1
    cdef double lat2
    cdef double lng2
    lng1, lat1 = coord1
    lng2, lat2 = coord2

    if lat1 > 90 or lat1 < -90 or lat2 > 90 or lat2 < -90:
        raise ValueError("Invalid latitude (should be between +/- 90)")
    if lng1 > 180 or lng1 < -180 or lng2 > 180 or lng2 < -180:
        raise ValueError("Invalid longitude (should be between +/- 180)")

    cdef double ph1
    cdef double ph2
    cdef double theta1
    cdef double theta2
    cdef double c
    cdef double arc

    phi1 = (90.0 - lat1) * 0.0174532925
    phi2 = (90.0 - lat2) * 0.0174532925
    theta1 = lng1 * 0.0174532925
    theta2 = lng2 * 0.0174532925

    c = (sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2))
    arc = acos(c)
    return arc * 6367444.7
