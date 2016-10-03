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

import pandas as pd
from osgeo.ogr import Geometry
from shapely.geometry.base import BaseGeometry
import json

from jinja2 import Environment, PackageLoader
from pyspatial.dataset import to_dict, dumps
from pyspatial.utils import projection_from_epsg
from pyspatial import fileutils

STATIC_CSS = "https://granular-labs.s3.amazonaws.com/static/css"
STATIC_JS = "https://granular-labs.s3.amazonaws.com/static/js"
STATIC_IMG = "https://granular-labs.s3.amazonaws.com/static/img"


def to_feature(shp, _id):
    if isinstance(shp, BaseGeometry):
        return {"type": "Feature", "id": _id,
                "geometry": shp.__geo_interface__}

    elif isinstance(shp, Geometry):
        geom = shp.Clone()
        geom.TransformTo(projection_from_epsg())
        return {"type": "Feature", "id": _id,
                "geometry": json.loads(geom.ExportToJson())}
    elif pd.isnull(shp):
        return {"type": "Feature", "id": _id,
                "geometry": {"type": "Point",
                             "coordinates": []}}
    else:
        raise ValueError("Unable create feature dict from shp")


def get_geojson_dict(geo_data):
    features = None
    msg = "Unable to geo_data to convert to dict"

    if isinstance(geo_data, BaseGeometry) or isinstance(geo_data, Geometry):
        geo_data = [geo_data]

    elif isinstance(geo_data, str) or isinstance(geo_data, unicode):
        geo_data = json.loads(geo_data)

    if hasattr(geo_data, "__iter__") and len(geo_data) == 0:
        features = []

    # TODO: Find a better check with isinstance
    # since that would cause a circular reference
    # in vector.py
    elif hasattr(geo_data, "proj"):
        return geo_data.dropna().to_dict()

    elif isinstance(geo_data, pd.Series):
        g0 = geo_data[0]
        if isinstance(g0, Geometry) or isinstance(g0, BaseGeometry):
            features = [to_feature(s, i) for i, s in geo_data.items()]
        else:
            raise ValueError(msg)

    elif isinstance(geo_data, list):
        features = [to_feature(s, i) for i, s in enumerate(geo_data)]
    elif isinstance(geo_data, dict):
        if "type" not in geo_data and "features" not in geo_data:
            raise ValueError("Invalid geo_data")
        return geo_data

    else:
        raise ValueError(msg)

    return {"type": "FeatureCollection", "features": features}


def to_latlng(shp):
    if isinstance(shp, BaseGeometry):
        pt = to_feature(shp.centroid, 0)
    elif isinstance(shp, Geometry):
        pt = to_feature(shp.Centroid(), 0)
    elif pd.isnull(shp):
        return {"lat": None, "lng": None}
    else:
        raise ValueError("Unable to create latlng from shp")

    return dict(zip(["lng", "lat"], pt["geometry"]["coordinates"]))


def get_latlngs(geo_data):
    if isinstance(geo_data, BaseGeometry) or isinstance(geo_data, Geometry):
        return [to_latlng(geo_data)]

    elif hasattr(geo_data, "__iter__") and len(geo_data) == 0:
        return []

    elif hasattr(geo_data, "__iter__"):
        return [to_latlng(s) for i, s in geo_data.items()]

    else:
        raise ValueError("Invalid geo_data")


class HTMLMap(object):
    def __init__(self, lat, lng, zoom=5, data=None, info_cols=None,
                 height="100%", static_js=STATIC_JS, static_css=STATIC_CSS,
                 static_img=STATIC_IMG):
        self.lat = lat
        self.lng = lng
        self.zoom = zoom
        self.data = data
        self.info_cols = info_cols
        self.height = height
        self.base_layer = None
        self.choropleths = {}
        self.overlays = {}
        self.palettes = {}
        self.static_css = static_css
        self.static_js = static_js
        self.static_img = static_img

    def set_baselayer(self, geo_data):
        self.base_layer = get_geojson_dict(geo_data)

    def _render(self, height=None):
        if height is None:
            height = self.height

        env = Environment(loader=PackageLoader('pyspatial', 'templates'))
        data = {"base_layer": self.base_layer,
                "view": {"lat": self.lat, "lng": self.lng, "zoom": self.zoom},
                "overlays": self.overlays,
                "info_cols": self.info_cols,
                "dataset": to_dict(self.data),
                "choropleths": self.choropleths}

        data_json = dumps(data)
        opts = dict(DATA=data_json, height=height,
                    static_js=self.static_js, static_css=self.static_css,
                    static_img=self.static_img)

        self.html = env.get_template("map.html").render(**opts)

    def render_ipython(self, height="500px", width="100%"):
        from IPython.display import HTML
        self._render(height=height)
        """
        Embeds the HTML source of the report directly into an IPython notebook.
        """
        iframe = ('<iframe srcdoc="{srcdoc}" style="width: {width}; '
                  'height: {height}; border: none" id="some_id"></iframe>')
        srcdoc = self.html.replace('"', '&quot;')
        return HTML(iframe.format(width=width, height=height, srcdoc=srcdoc))

    def add_shapes(self, name, shapes, style=None, text=None):
        """Add shapes to the map. Accepts pandas Series or list of
        shapely or ogr.Geometry objects, or a dictionary or string
        matching the geojson spec."""
        if style is None:
            style = {}

        self.overlays[name] = {"shape": get_geojson_dict(shapes),
                               "style": style,
                               "kind": "geoJson",
                               "text": text}

    def add_markers(self, name, shapes, style=None, text=None):
        if style is None:
            style = {}

        self.overlays[name] = {"shape": get_latlngs(shapes),
                               "kind": "marker",
                               "style": style,
                               "text": text}

    def add_text(self, name, points, values, style=None):
        """Not implemented yet"""
        _style = {'background-color': 'none',
                  'font-size': '10pt',
                  'font-weight': 'bold',
                  'margin-top': '0'}

        if style is not None:
            _style.update(style)

        css = ";".join(["%s:%s" % (k, v) for k, v in _style.items()])
        style = "style=\"{css}\"".format(css=css)

    def choropleth(self, column=None, levels=6, palette="Reds"):
        '''
        Create a choropleth.

        #TODO:
        - Add style per choropleth.
        - Add support for different discretization schemes

        Parameters
        ----------
        column: str
            The column to use for the choropleth. Must exist in the data
            provided.

        levels: int, default None
            Number of levels to uses for the scale.  The allowed amount varies
            based on the palette that is chosen.
        palette: string or dict, default will use 'Reds'
             Uses color brewer (http://colorbrewer2.org/).
             If you pass a dict, the keys are the string values in the column,
             and the values are the html hex color for that value.
        '''

        if self.base_layer is None:
            raise ValueError("base_layer not set")

        if self.data is None:
            index = [f["id"] for f in self.base_layer["features"]]
            self.data = pd.DataFrame({"shapes": 1.}, index=index)
            levels = 3
            column = "shapes"

        if column not in self.data.columns:
            raise ValueError("%s not in %s" % (column, self.data.columns))

        if isinstance(palette, dict):
            self.choropleths[column] = palette
        else:
            self.choropleths[column] = {"palette": palette, "levels": levels}

    def save(self, path):
        """Save the html to a file.  Can be local or s3.  If s3, assumes that
        a .boto file exists"""
        with fileutils.open(path, 'wb') as outf:
            self._render()
            outf.write(self.html)
