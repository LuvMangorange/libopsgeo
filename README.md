# libopsgeo - Geospatial Processing Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Features

### Data Conversion
- Excel → Point Shapefile (`table2point`)
- Vector → Raster (`shp2tif`)
- Raster → Point Features (`tif2point`)
- GeoJSON ↔ Shapefile Conversion (`geojson2shp`)

### Spatial Analysis
- IDW Interpolation (`interpolate`)
- Raster Reclassification (`classify`)
- Raster Cropping (`extract`)
- Multispectral Composition (`group_tif`)

### Coordinate Transformation
- Batch Reprojection (`reproject`)
- Raster Resampling (`resample`)

### Visualization
- Color Ramp Generation (`calc_gradient_color`)
- Raster Colormapping (`colorize`)

## Installation

```bash
# Install dependencies
pip install numpy pandas geopandas gdal pyproj tqdm

# Install library (development mode)
git clone https://github.com/yourusername/lib-opsgeo.git
cd lib-opsgeo
pip install -e .
