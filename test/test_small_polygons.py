import os
from pyspatial.vector import read_geojson, to_geometry
from pyspatial.raster import read_raster
import numpy as np

base_dir = os.path.abspath(os.path.dirname(__file__))
get_path = lambda x: os.path.join(base_dir, "data", x)
vl, vldf = read_geojson(get_path("vector/small_polygon.geojson"))
rd = read_raster(get_path("raster/95000_45000.tif"))
shp = vl[0]


def test_small_polygon():
    shp_px = rd.to_pixels(vl)[0]

    grid = rd.to_geometry_grid(*shp_px.bounds)

    areas = {}
    for i, b in grid.iteritems():
        diff = b.Intersection(to_geometry(shp, proj=rd.proj))
        areas[i] = diff.GetArea()

    total_area = sum(areas.values())
    index = areas.keys()
    if total_area > 0:
        weights = np.array([areas[k]/total_area for k in index])
    else:
        weights = np.zeros(len(index))

    assert abs(total_area - shp.GetArea()) < 1e-8
    assert abs(1 - sum(weights)) < 1e-8
    values = rd.get_values_for_pixels(np.array(index))
    print values
    print weights
