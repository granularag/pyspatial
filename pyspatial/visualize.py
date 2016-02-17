import pandas as pd
from osgeo.ogr import Geometry
from shapely.geometry.base import BaseGeometry
import json

from jinja2 import Environment, PackageLoader
from dataset import to_dict, dumps
from smart_open import smart_open


def to_feature(shp, _id):
    if isinstance(shp, BaseGeometry):
        return {"type": "Feature", "id": _id,
                "geometry": shp.__geo_interface__}

    elif isinstance(shp, Geometry):
        return {"type": "Feature", "id": _id,
                "geometry": json.loads(shp.ExportToJson())}
    else:
        raise ValueError("Unable to shp to feature")


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
    # Check if vector layer
    elif hasattr(geo_data, "proj"):
        return geo_data.dropna().to_dict()

    elif isinstance(geo_data, pd.Series):
        g0 = geo_data[0]
        if isinstance(g0, Geometry) or isinstance(g0, BaseGeometry):
            features = [to_feature(s, i)
                        for i, s in geo_data.dropna().iteritems()]
        else:
            raise ValueError(msg)

    elif isinstance(geo_data, list):
        print "Its'a list!"
        features = [to_feature(s, i) for i, s in enumerate(geo_data)]
    elif isinstance(geo_data, dict):
        if "type" not in geo_data and "features" not in geo_data:
            raise ValueError("Invalid geo_data")
        return geo_data

    else:
        raise ValueError(msg)

    return {"type": "FeatureCollection", "features": features}


class HTMLMap(object):
    def __init__(self, lat, lng, zoom=5, data=None, info_cols=None):
        self.lat = lat
        self.lng = lng
        self.zoom = zoom
        self.data = data
        self.info_cols = info_cols
        self.base_layer = None
        self.choropleths = {}
        self.overlays = {}
        self.palettes = {}

    def set_baselayer(self, geo_data):
        self.base_layer = get_geojson_dict(geo_data)

    def _render(self):
        env = Environment(loader=PackageLoader('pyspatial', 'templates'))
        data = {"base_layer": self.base_layer,
                "view": {"lat": self.lat, "lng": self.lng, "zoom": self.zoom},
                "overlays": self.overlays,
                "info_cols": self.info_cols,
                "dataset": to_dict(self.data), "choropleths": self.choropleths}

        data_json = dumps(data)
        self.html = env.get_template("map.html").render(DATA=data_json)

    def render_ipython(self, height="500px", width="100%"):
        from IPython.display import HTML
        self._render()
        """
        Embeds the HTML source of the report directly into an IPython notebook.
        """
        iframe = ('<iframe srcdoc="{srcdoc}" style="width: {width}; '
                  'height: {height}; border: none" id="some_id"></iframe>')
        srcdoc = self.html.replace('"', '&quot;')
        return HTML(iframe.format(width=width, height=height, srcdoc=srcdoc))

    def add_shapes(self, name, shapes, style=None):
        if style is None:
            style = {}

        self.overlays[name] = {"shape": get_geojson_dict(shapes),
                               "style": style}

    def add_text(self, name, points, values, style=None):
        _style = {'background-color': 'none',
                  'font-size': '10pt',
                  'font-weight': 'bold',
                  'margin-top': '0'}

        if style is not None:
            _style.update(style)

        css = ";".join(["%s:%s" % (k, v) for k, v in _style.iteritems()])
        style = "style=\"{css}\"".format(css=css)

    def choropleth(self, column=None, levels=6, palette="Reds"):
        '''
        Create a choropleth.

        #TODO:
        - Add style per choropleth.

        Parameters
        ----------
        geo_data: string or dict
            String or dict of GeoJSON, assumes that it is uses
            a FeatureCollection. Each feature must have an 'id'
            attribute, and the index of data will join against
            the 'id' attribute of the features.

        columns: list
            The columns to use for the choropleth. Must exist in the data
            provided or in the data set by add_data. Uses data.columns if None

        n_levels: list, default None
            Number of levels to uses for the scale.  The allowed amount varies
            based on the palette that is chosen (For numerical data only).
            For categorical data, it will automatically choose the number
            of levels (max of 12 supported).
        palette: string or dict, default will autoselect
             Uses color brewer (http://colorbrewer2.org/).
             To get a list of palettes:

             >>> from folium import PALETTES
             >>> PALETTES.keys()

             If you pass a dict, data must be a DataFrame, and the keys
             of the dict must be the columns names in data.  The values
             must be a string of the palette you wish to use.
        info_cols: list, default: None
             Columns to add to the hover control

        filters: dict
             Mapping of column name -> values to filter. Creates UI filters
             on the map.

        Output
        ------
        GeoJSON data layer in obj.template_vars
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
        with smart_open(path, 'wb') as outf:
            self._render()
            outf.write(self.html)
