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
import pyspatial.raster as rst
import pyspatial.vector as vt
from pyspatial import fileutils
import unittest
import boto

# check if google cloud storage config exists
gs_access_key_id_exists = boto.config.get('Credentials', 'gs_access_key_id') is not None
gs_secret_access_key_exists = boto.config.get('Credentials', 'gs_secret_access_key') is not None

gs_config_exists = gs_access_key_id_exists and gs_secret_access_key_exists

# get the bucket name from env variables if it exists
gs_bucket_name = os.getenv('PS_GS_DEFAULT_BUCKET', 'pyspatial')

base = os.path.abspath(os.path.dirname(__file__))

def delete_gs_key(path):
    storage_uri = boto.storage_uri(gs_bucket_name, 'gs')
    bucket = storage_uri.get_bucket(gs_bucket_name)
    key = bucket.lookup(path)
    key.delete()

@unittest.skipIf(not gs_config_exists, "requires google cloud storage configuration")
def test_raster_io_gs():
    filepath = 'data/raster/prism.tif'
    gs_path = 'gs://%s/%s' % (gs_bucket_name, filepath)

    # write a raster file to gs
    outf = fileutils.open(gs_path, "wb")
    with open(os.path.join(base, filepath)) as inf:
        outf.write(inf.read())
    outf.close()

    # read the raster file from gs
    rd = rst.read_raster(gs_path)
    assert isinstance(rd, rst.RasterDataset)

    # clean up
    delete_gs_key(filepath)

@unittest.skipIf(not gs_config_exists, "requires google cloud storage configuration")
def test_vector_json_io_gs():
    filepath = 'data/vector/test_shape.json'
    gs_path = 'gs://%s/%s' % (gs_bucket_name, filepath)

    # write a vector json file to gs
    shape, _ = vt.read_geojson(os.path.join(base, filepath))
    shape.to_json(path=gs_path, precision=15)

    # read the vector json file from gs
    shape_json = vt.fetch_geojson(gs_path)
    geojson, _ = vt.read_geojson(shape_json)

    # ensure it is valid
    assert geojson[0].IsValid()

    # clean up
    delete_gs_key(filepath)

@unittest.skipIf(not gs_config_exists, "requires google cloud storage configuration")
def test_vector_shp_io_gs():
    filepath = 'data/vector/clu/clu_public_a_co095.shp'
    gs_path = 'gs://%s/%s' % (gs_bucket_name, filepath)

    # write a vector shape file to gs
    co, co_df = vt.read_layer(os.path.join(base, filepath))
    co.to_shapefile(gs_path, df=co_df)

    # read the vector shape file from gs
    #co, co_df = vt.read_layer(gs_path)
    #assert isinstance(co, vt.VectorLayer)

    # clean up
    delete_gs_key(filepath)