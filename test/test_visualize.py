import os
from nose.tools import assert_equals
from pyspatial.vector import read_layer
from pyspatial.visualize import get_latlngs

base = os.path.abspath(os.path.dirname(__file__))
get_path = lambda x: os.path.join(base, "data/vector", x)

vl, vldf = read_layer(get_path("cb_2014_us_state_20m.zip"),
                      index="STUSPS")


def test_getlatlngs():
    shp = vl.to_shapely("CA")
    exp = [{'lat': 37.242214717335116, 'lng': -119.61111973321412}]
    assert_equals(get_latlngs(vl["CA"]), exp)
    assert_equals(get_latlngs(shp), exp)
    assert_equals([get_latlngs(vl)[0]], exp)
