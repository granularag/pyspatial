import os
import pickle
import pyspatial.vector as vt
from pyspatial.spatiallib import haversine
from osgeo import ogr
from nose.tools import assert_raises

base = os.path.abspath(os.path.dirname(__file__))
get_path = lambda x: os.path.join(base, "data/vector", x)


def test_read():
    path = ("http://www2.census.gov/geo/tiger/GENZ2014/shp/"
            "cb_2014_us_state_500k.zip")

    assert len(vt.read_layer(path)[0]) == 56

    path = get_path("cb_2014_us_state_500k.zip")
    assert len(vt.read_layer(path)[0]) == 56

    paths = ["http://eric.clst.org/wupl/Stuff/gz_2010_us_040_00_500k.json",
             "s3://granular-raster/test/gz_2010_us_040_00_500k.json",
             get_path("gz_2010_us_040_00_500k.json")]

    for path in paths:
        assert len(vt.read_geojson(path)[0]) == 52

    geojson_str = open(paths[2]).read()
    assert len(vt.read_geojson(geojson_str)[0]) == 52


class TestVectorLayer:
    @classmethod
    def setup_class(cls):
        path1 = get_path("clu/four_shapes_2il_2ca.geojson")
        path2 = get_path("gz_2010_us_040_00_500k.json")
        path3 = get_path("bay_area_counties.geojson")
        path4 = get_path("bay_area_zips.geojson")

        cls.vl1, cls.df1 = vt.read_geojson(path1)
        cls.vl2, cls.df2 = vt.read_geojson(path2)
        cls.counties, cls.df3 = vt.read_geojson(path3, index="NAME")
        cls.zips, cls.df4 = vt.read_geojson(path4, index="ZCTA5CE10")
        p = get_path("clu/four_shapes_2il_2ca.p")
        cls.df = pickle.load(open(p))
        assert isinstance(cls.counties, vt.VectorLayer)
        assert isinstance(cls.counties["San Francisco"], ogr.Geometry)

    def test_read(self):
        states, df = vt.read_layer(get_path("cb_2014_us_state_500k.zip"))
        assert isinstance(states, vt.VectorLayer)
        assert len(states) == 56
        co, co_df = vt.read_layer(get_path("clu/clu_public_a_co095.shp"))
        assert isinstance(co, vt.VectorLayer)

    def test_from_series(self):
        series = self.df["__geometry__"]
        assert isinstance(vt.from_series(series), vt.VectorLayer)

    def test_predicates(self):
        sf = self.counties["San Francisco"]
        assert isinstance(sf, ogr.Geometry)
        sf_ids = self.zips.within(sf, index_only=True)
        assert all(map(lambda x: x.startswith("941"), sf_ids))

    def test_intersection(self):
        sf = self.counties["San Francisco"]
        vl = self.zips.intersection(sf)
        assert abs(vl.unary_union().area - vt.to_shapely(sf).area) < 1e-3

    def test_read_index(self):
        path = get_path("cb_2014_us_state_500k.zip")
        vl, df = vt.read_layer(path, index="STUSPS")
        assert all([a == b for a, b in zip(vl.index, df.index)])
        vl, df = vt.read_layer(path, index=xrange(5, 61))
        assert_raises(ValueError, vt.read_layer, path, 0, xrange(5, 56))

    def test_ipredictes(self):
        path = get_path("cb_2014_us_state_500k.zip")
        vl, df = vt.read_layer(path, index="STUSPS")
        clu_path = get_path("clu/clu_public_a_il189.shp")
        clus, clusdf = vt.read_layer(clu_path)
        assert vl.iintersects(clus[0])[0] == "IL"
        assert len(vl.iwithin(clus[0])) == 0
        assert "IL" not in vl.idisjoint(clus[0])
        assert vl[["CA"]].boundingboxes().icontains(vl["CA"])[0] == "CA"
        assert len(vl[["CA"]].boundingboxes().iwithin(vl["CA"])) == 0

    def test_distance(self):
        clu_path = get_path("clu/clu_public_a_il189.shp")
        clus, clusdf = vt.read_layer(clu_path)
        path = get_path("cb_2014_us_state_500k.zip")
        vl, vldf = vt.read_layer(path, index="STUSPS")
        s = clus[:5].distances(vl["RI"], proj="albers")/1.e3
        assert s.count() == 5
        assert s.min() > 1440 and s.max() < 1441
        d = vl.boundingboxes().distances(vl["MI"], proj='albers')/1.e3
        assert abs(d["CA"] - 1853.3812112445789) < 1e-6
        # Compute the haversine distance to MI
        MI = vl["MI"].Centroid().GetPoints()[0]
        d = (vl.to_wgs84().centroids(format="Series")
             .map(lambda x: haversine(x, MI))/1.e3)
        assert abs(d["WA"] - 2706.922595) < 1e-6
