import gdal
from osgeo import gdal
import os
import gdal
from osgeo import gdal_array
from osgeo import osr
from osgeo import ogr
import subprocess
import numpy as np
from osgeo import gdal
import subprocess
import collections
from matplotlib import pyplot as plt
from shapely import geometry, wkt
import json


gdal.AllRegister()

gdaldem = "gdaldem.exe slope"
src = "puszcza_bukowa_szmaragdowe.asc"
dst = "szmaragdowe_geotif_slope.tif"

subprocess.run(gdaldem + " -of Gtiff -b 1 -s 1.0 " + src + " " + dst)

driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(dst)
band = file.GetRasterBand(1)
slope = band.ReadAsArray()

georef = file.GetGeoTransform()
# print(f'Raster params: {georef}')

# reklasyfikacja
slope[np.where(slope <= 5)] = 1
slope[np.where((5 < slope) & (slope <= 10))] = 2
slope[np.where((10 < slope) & (slope <= 20))] = 3
slope[np.where(slope > 20)] = 4

for x in np.unique(slope):
    tot_num_pixels = np.sum(slope == x)
    area = tot_num_pixels * georef[1]
    print(x, area/10000, 'ha')

multipoint = ogr.Geometry(ogr.wkbMultiPoint)
point = ogr.Geometry(ogr.wkbPoint)
for i in range(slope.shape[0]):
    for j in range(slope.shape[1]):
        point.AddPoint(i, j, slope[i, j].astype("double"))
        multipoint.AddGeometry(point)

# Tworzenie GeoJSON
# https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
out_driver = ogr.GetDriverByName('GeoJSON')
out_data_source = out_driver.CreateDataSource('test.geojson')
out_layer = out_data_source.CreateLayer('test.geojson', geom_type=ogr.wkbMultiPoint)
feature_defn = out_layer.GetLayerDefn()
out_feature = ogr.Feature(feature_defn)
out_feature.SetGeometry(multipoint)
out_layer.CreateFeature(out_feature)
out_feature = None
out_data_source = None

# zapis pliku
file_tif = driver.Create('szmaragdowe_geotif_slope_reclass.tif', file.RasterXSize, file.RasterYSize, 1)
file_tif.GetRasterBand(1).WriteArray(slope)

# układ współrzędnych pobierany z pliku źródłowego
proj = file.GetProjection()
georef = file.GetGeoTransform()
file_tif.SetProjection(proj)
file_tif.SetGeoTransform(georef)

plt.imshow(slope, interpolation='nearest', vmin=1)
plt.colorbar()
plt.show()
