# Expanding a Color Table
```bash
$ gdal_translate -of vrt -expand rgba 95000_45000.tif 95000_45000_rgb.vrt
```

# Creating tiles for viewing on the web
  * The -z is the zoom levels to generate
```bash
gdal2tiles.py -z 7-13 95000_45000_rgb.vrt /tmp/95000_45000_rgb
```

# Error handling GDAL
http://pcjericks.github.io/py-gdalogr-cookbook/gdal_general.html#install-gdal-ogr-error-handler
```python
try:
    from osgeo import ogr, osr, gdal
except:
    sys.exit('ERROR: cannot find GDAL/OGR modules')

# example GDAL error handler function
def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print 'Error Number: %s' % (err_num)
    print 'Error Type: %s' % (err_class)
    print 'Error Message: %s' % (err_msg)

if __name__=='__main__':

    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)

    # Raise a dummy error
    gdal.Error(1, 2, 'test error')

    #uninstall error handler
    gdal.PopErrorHandler()
```
