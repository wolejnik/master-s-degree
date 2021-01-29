import subprocess

subprocess.run("gdal_translate -a_srs EPSG:32633 -of GPKG szmaragdowe_geotif_slope.tif szmaragdowe.gpkg")
subprocess.run("gdal_translate -a_srs EPSG:2180 -of GTiff szmaragdowe_geotif_slope.tif szmaragdowe.tif")
subprocess.run("gdal_contour -i 1.0 -a elev szmaragdowe.tif szmaragdowe.shp")