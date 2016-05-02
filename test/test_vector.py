"""
Copyright (c) 2016, Granular, Inc.
All rights reserved.
License: BSD 3-Clause ("BSD New" or "BSD Simplified")

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions
    and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
    following disclaimer in the documentation and/or other materials provided with the distribution.
  * Neither the name of the nor the names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import pickle
import pyspatial.vector as vt
from pyspatial.utils import projection_from_string, ALBERS_N_AMERICA
from pyspatial.utils import projection_from_epsg
from pyspatial.spatiallib import haversine
from osgeo import ogr
from nose.tools import assert_raises, assert_almost_equal

base = os.path.abspath(os.path.dirname(__file__))
get_path = lambda x: os.path.join(base, "data/vector", x)

rect_str = """
{
        "type": "Polygon",
        "coordinates": [
          [
            [
              -122.58956909179688,
              37.73379707124429
            ],
            [
              -122.58956909179688,
              37.80218877920469
            ],
            [
              -122.4587631225586,
              37.80218877920469
            ],
            [
              -122.4587631225586,
              37.73379707124429
            ],
            [
              -122.58956909179688,
              37.73379707124429
            ]
          ]
        ]
      }"""
rect = ogr.CreateGeometryFromJson(rect_str)

farallon_str = """
{
    "type": "Polygon",
        "coordinates": [
          [
            [
              -123.04550170898436,
              37.6816466602918
            ],
            [
              -123.04550170898436,
              37.72293542866175
            ],
            [
              -122.97409057617188,
              37.72293542866175
            ],
            [
              -122.97409057617188,
              37.6816466602918
            ],
            [
              -123.04550170898436,
              37.6816466602918
            ]
          ]
        ]
}"""

farallon = ogr.CreateGeometryFromJson(farallon_str)


def __test_read():
    path = ("http://www2.census.gov/geo/tiger/GENZ2014/shp/"
            "cb_2014_us_state_500k.zip")

    assert len(vt.read_layer(path)[0]) == 56

    path = get_path("cb_2014_us_state_500k.zip")
    assert len(vt.read_layer(path)[0]) == 56

    paths = ["http://eric.clst.org/wupl/Stuff/gz_2010_us_040_00_500k.json",
             get_path("gz_2010_us_040_00_500k.json")]

    for path in paths:
        assert len(vt.read_geojson(path)[0]) == 52

    geojson_str = open(paths[1]).read()
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
        cls.sf = "San Francisco"
        proj = projection_from_epsg()
        rect.AssignSpatialReference(proj)
        farallon.AssignSpatialReference(proj)
        cls.counties[cls.sf] = cls.counties[cls.sf].Difference(farallon)
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

    def test_ipredicates(self):
        path = get_path("cb_2014_us_state_500k.zip")
        vl, df = vt.read_layer(path, index="STUSPS")
        clu_path = get_path("clu/clu_public_a_il189.shp")
        clus, clusdf = vt.read_layer(clu_path)
        assert vl.iintersects(clus[0])[0] == "IL"
        assert len(vl.iwithin(clus[0])) == 0
        assert "IL" not in vl.idisjoint(clus[0])
        centroid = vl["CA"].Centroid()
        centroid.AssignSpatialReference(vl.proj)
        # California contains its centroid
        assert vl.icontains(centroid)[0] == "CA"
        # Clus are within IL
        assert len(clus.head().iwithin(vl["IL"])) > 0
        # No states within clus
        assert len(vl.iwithin(clus.head()[0])) == 0
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
        # Compute the haversine (great-circle) distance to MI
        MI = vl["MI"].Centroid().GetPoints()[0]
        d = (vl.to_wgs84().centroids(format="Series")
             .map(lambda x: haversine(x, MI))/1.e3)
        assert abs(d["WA"] - 2706.922595) < 1e-6

    def test_to_json(self):
        with open(get_path("RI.json")) as inf:
            exp = inf.read()
        act, _ = vt.read_layer(get_path("cb_2014_us_state_20m.zip"),
                               index="STUSPS")
        assert exp == act[["RI"]].to_json()

    def test_set_theoretic(self):
        proj = projection_from_string(ALBERS_N_AMERICA)
        counties = self.counties.transform(proj)
        sf = [self.sf]

        union_exp = 183.026345584
        union_act = counties[sf].union(rect)[self.sf]
        assert_almost_equal(union_act.GetArea()/1e6, union_exp)

        intersection_exp = 31.0072128793
        intersection_act = counties[sf].intersection(rect)[self.sf]

        assert_almost_equal(intersection_act.GetArea()/1e6, intersection_exp)

        symdifference_exp = 152.019132704
        symdifference_act = counties[sf].difference(rect, kind="symmetric")
        assert_almost_equal(symdifference_act[self.sf].GetArea()/1e6,
                            symdifference_exp)

        ldifference_exp = 95.5399654593
        ldifference_act = counties[sf].difference(rect, kind="left")
        assert_almost_equal(ldifference_act[self.sf].GetArea()/1e6,
                            ldifference_exp)

        rdifference_exp = 56.4791672452
        rdifference_act = counties[sf].difference(rect, kind="right")
        assert_almost_equal(rdifference_act[self.sf].GetArea()/1e6,
                            rdifference_exp)


def test_intersects():
    shape, _ = vt.read_geojson(get_path('test_shape.json'))

    with open(get_path('test_soils.json')) as f:
        soils = f.read()
        soils = vt.read_geojson(soils)[0]

    soils.intersects(shape[0])
    assert (shape[0].IsValid())


def test_to_json():
    shape, _ = vt.read_geojson(get_path('test_shape.json'))
    # This shape requires more precision when serializing
    j = shape.to_json(precision=15)
    jj, _ = vt.read_geojson(j)
    assert jj[0].IsValid()
