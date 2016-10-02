# Expanding a Color Table
```bash
$ gdal_translate -of vrt -expand rgba 95000_45000.tif 95000_45000_rgb.vrt
```

# Creating tiles for viewing on the web
```bash
gdal2tiles.py 95000_45000_rgb.vrt /tmp/95000_45000_rgb
```
