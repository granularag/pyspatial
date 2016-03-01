import os
from pyspatial.vector import read_geojson, to_geometry
from pyspatial.raster import read_raster
import numpy as np
from numpy.testing import assert_array_almost_equal
from test_raster_query import compute_stats

base_dir = os.path.abspath(os.path.dirname(__file__))
get_path = lambda x: os.path.join(base_dir, "data", x)
vl, vldf = read_geojson(get_path("vector/small_polygon.geojson"))
rd = read_raster(get_path("raster/95000_45000.tif"))
shp = vl[0]


def test_small_polygon():
    bboxes = vl.boundingboxes()
    shp_px = rd.to_pixels(bboxes)[0]

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
    assert_array_almost_equal(values, np.array([121, 176, 176, 176],
                                               dtype=np.uint8))

    classes = np.arange(0, 256)
    exp_stats = [0.114183, 0.885817]
    exp_classes = [121, 176]
    for r in rd.query(vl):
        stats = compute_stats(r.values, r.weights)
        assert_array_almost_equal(exp_classes, classes[stats > 0])
        assert_array_almost_equal(exp_stats, stats[stats > 0])
