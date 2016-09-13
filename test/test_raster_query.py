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

# Scipy stack
import numpy as np
import pandas as pd

# Spatial
from pyspatial.vector import read_layer, read_geojson
from pyspatial.raster import read_catalog

from nose.tools import timed

cwd = os.getcwd()
base = os.path.abspath(os.path.dirname(__file__))


def get_path(x):
    return os.path.join(base, "data", x)


def compute_stats(values, weights):
    # values = np.array(row, dtype=np.uint8)
    counts = np.bincount(values, weights=weights, minlength=256)
    zeros = np.zeros(counts.shape)
    total = 1. * np.sum(counts)
    return counts/total if total > 0 else zeros


class TestRasterQuery:
    @classmethod
    def setup_class(cls):
        os.chdir(base)

        # Let's add some CLUs for Washington County, IL
        cls.vl, _ = read_layer(get_path("vector/clu/clu_public_a_il189.shp"),
                               index="uniqueid")
        cls.one_shape_id = "3850036893056913813r"
        cls.vl_one_shape = cls.vl[[cls.one_shape_id]]
        cls.areas = cls.vl.areas()
        cls.areas = cls.areas/cls.areas.sum()

        # Read expected output computed via single-tile computation
        # from a file, so we can compare against that.
        cls.df_expected = pd.read_csv(get_path("expected.csv"),
                                      index_col=0)
        cls.df_expected.rename(columns=lambda x: int(x), inplace=True)
        cls.df_expected.applymap(lambda x: int(x))

        # For single shape output, we only want to compare our data against that shape.
        cls.df_expected_one_shape = cls.df_expected.loc[cls.one_shape_id]

    @classmethod
    def teardown_class(cls):
        os.chdir(cwd)

    def make_dataframe(self, generator):
        index = []
        res = []

        for r in generator:
            res.append(compute_stats(r.values, r.weights))
            index.append(r.id)

        return pd.DataFrame(res, index=index)

    def test_shapes_outside_raster_should_be_filtered(self):
        # 2 shapes in IL that should be in range,
        # plus 2 shapes in CA that should be out of range.
        p = get_path("vector/clu/four_shapes_2il_2ca.geojson")
        vl_outside, vl_df = read_geojson(p)
        assert (len(vl_outside.keys()) == 4)

        dataset_catalog_file = get_path("../catalog/cdl_2014.json")
        rd = read_catalog(dataset_catalog_file)
        df = self.make_dataframe(rd.query(vl_outside))
        assert (len(df.index) == 4)
        sums = df.sum(axis=1).map(int)
        assert df[sums < 1e-6].shape[0] == 2
        assert df[(sums - 1).map(np.abs) < 1e-6].shape[0] == 2

    # Compute term frequency for cdl_2014 on a tiled dataset for all shapes,
    # and compare against our saved single-tile computation.
    @timed(60)
    def test_term_frequency_tiled_all_shapes(self):
        dataset_catalog_file = get_path("../catalog/cdl_2014.json")
        rd = read_catalog(dataset_catalog_file)
        df = self.make_dataframe(rd.query(self.vl))

        assert (len(df.index) == 23403)

        # Compare against expected output computed via single-tile computation
        # read from a file.
        df_distance = (self.df_expected - df).applymap(np.abs)
        corn_error = (df_distance[1]*self.areas).sum()
        soy_error = (df_distance[5]*self.areas).sum()
        assert(corn_error < 0.02), corn_error
        assert(soy_error < 0.02), soy_error

    # Removed this functionality as it needs more thought
    # Compute term frequency for cdl_2014 on a tiled dataset for all shapes,
    # and compare against our saved single-tile computation.
    #@timed(45)
    #def test_term_frequency_tiled_all_shapes_with_index(self):
    #    dataset_catalog_file = get_path("../catalog/cdl_2014_with_index.json")
    #    rd = read_catalog(dataset_catalog_file)
    #    df = self.make_dataframe(rd.query(self.vl))

    #    assert (len(df.index) == 23403)

        # Compare against expected output computed via single-tile computation
        # read from a file.
    #    df_distance = (self.df_expected - df).applymap(np.abs)
    #    corn_error = (df_distance[1]*self.areas).sum()
    #    soy_error = (df_distance[5]*self.areas).sum()
    #    assert(corn_error < 0.02), corn_error
    #    assert(soy_error < 0.02), soy_error

    # Compute term frequency for cdl_2014 on an untiled dataset for all shapes,
    # and compare against our saved single-tile computation.
    @timed(60)
    def test_term_frequency_untiled_all_shapes(self):
        dataset_catalog_file = get_path("../catalog/cdl_2014_untiled.json")
        rd = read_catalog(dataset_catalog_file)
        df = self.make_dataframe(rd.query(self.vl))

        assert (len(df.index) == 23403)

        # Compare against expected output computed via single-tile computation
        # read from a file.
        df_distance = (self.df_expected - df).applymap(np.abs)
        corn_error = (df_distance[1]*self.areas).sum()
        soy_error = (df_distance[5]*self.areas).sum()
        assert(corn_error < 0.02), corn_error
        assert(soy_error < 0.02), soy_error

    # Compute term frequency for cdl_2014 on a tiled dataset for one shape,
    # and compare against our saved single-tile computation.
    @timed(0.5)
    def test_term_frequency_tiled_one_shape(self):
        dataset_catalog_file = get_path("../catalog/cdl_2014.json")
        rd = read_catalog(dataset_catalog_file)
        df = self.make_dataframe(rd.query(self.vl_one_shape))

        # Compare against expected output computed via single-tile computation
        # read from a file.
        df_distance = (self.df_expected_one_shape - df.loc[self.one_shape_id])\
            .apply(np.abs)
        assert(df_distance[1] < 0.05)
        assert(df_distance[5] < 0.05)

    # Compute term frequency for cdl_2014 on an untiled dataset for one shape,
    # and compare against our saved single-tile computation.
    @timed(0.5)
    def test_term_frequency_untiled_one_shape(self):
        dataset_catalog_file = get_path("../catalog/cdl_2014_untiled.json")
        rd = read_catalog(dataset_catalog_file)
        df = self.make_dataframe(rd.query(self.vl_one_shape))
        # Compare against expected output computed via single-tile computation
        # read from a file.
        df_distance = (self.df_expected_one_shape - df.loc[self.one_shape_id])\
            .apply(np.abs)

        assert(df_distance[1] < 0.05)
        assert(df_distance[5] < 0.05)

    @timed(5)
    def test_co_ne_border(self):
        p = get_path("vector/clu/clu_public_a_co095.shp")
        vl, _ = read_layer(p)
        p = get_path("vector/co_ne_border.geojson")
        co_ne_border, df = read_geojson(p)
        vl = vl.within(co_ne_border.bbox())

        rd = read_catalog(get_path("../catalog/co_soil.json"))
        for r in rd.query(vl):
            r
            #compute_stats(r.values, r.weights)

        rd = read_catalog(get_path("../catalog/co_soil_bad.json"))
        failed = False
        try:
            for r in rd.query(vl):
                r
                #compute_stats(r.values, r.weights)
        except IndexError:
            failed = True

        assert failed

    # Test if tilepaths were defined from a different working directory
    # than the python code
    def test_unconventional_tilepath(self):
        dataset_catalog_file = get_path("../catalog/cdl_2014_tilepath.json")
        rd = read_catalog(dataset_catalog_file, workdir='data/raster')
        df = self.make_dataframe(rd.query(self.vl_one_shape))
        assert (len(df.index) == 1)
