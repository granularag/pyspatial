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
