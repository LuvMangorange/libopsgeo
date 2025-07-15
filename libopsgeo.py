"""
Autor: HuPengcheng hpc0813@outlook.com
Date: 2021-04-17 11:13:56
LastEditTime: 2024-06-27 09:41:21
Description: 
"""

# Import the json module for working with JSON data
import json

# Import the math module for mathematical operations
# import datetime
import math
# Import the os module for interacting with the operating system
import os
# Import the time module for time-related functions
import time

# Import literal_eval from ast module
# from ast import literal_eval
# Import type hints for function and class definitions
from typing import Callable, Dict, List, Optional, Union

# Import geopandas for working with geospatial data
import geopandas as gpd

# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation and analysis
import pandas as pd
# Import distance function from geopy for calculating distances
from geopy.distance import distance
# Import GDAL, OGR, and OSR modules from osgeo for geospatial data processing
from osgeo import gdal, ogr, osr
# Import tqdm for creating progress bars
from tqdm import tqdm


class OpsGeo:
    """
    A class for performing various geospatial operations.

    Attributes:
        src_path (Optional[Union[list, str]]): Source file path or list of paths.
        src_dir (Optional[str]): Source directory path.
        dst_path (Optional[str]): Destination file path.
        dst_dir (Optional[str]): Destination directory path.
        dst_name (Optional[str]): Destination file name.
        tmpl_path (Optional[str]): Template file path.
        mask_path (Optional[str]): Mask file path.
    """

    def __init__(
        self,
        src_path: Optional[Union[list, str]] = None,
        src_dir: Optional[str] = None,
        dst_path: Optional[str] = None,
        dst_dir: Optional[str] = None,
        dst_name: Optional[str] = None,
        tmpl_path: Optional[str] = None,
        mask_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpsGeo class.

        Args:
            src_path (Optional[Union[list, str]]): Source file path or list of paths.
            src_dir (Optional[str]): Source directory path.
            dst_path (Optional[str]): Destination file path.
            dst_dir (Optional[str]): Destination directory path.
            dst_name (Optional[str]): Destination file name.
            tmpl_path (Optional[str]): Template file path.
            mask_path (Optional[str]): Mask file path.
        """
        self.src_path = src_path
        self.src_dir = src_dir
        self.dst_path = dst_path
        self.dst_dir = dst_dir
        self.dst_name = dst_name
        self.tmpl_path = tmpl_path
        self.mask_path = mask_path
        # Initialize the destination path
        self.__init_dst_path()

    def __init_dst_path(self):
        """
        Initialize the destination path based on the provided parameters.

        Raises:
            ValueError: If 'dst_name' input is missing or destination path input is invalid.
        """
        if self.dst_path is not None:
            # Extract the directory and name from the destination path
            self.dst_dir = os.path.dirname(self.dst_path)
            self.dst_name = os.path.basename(self.dst_path)
        else:
            try:
                # Set the destination name if not provided
                self.dst_name = (
                    os.path.basename(self.src_path) if self.dst_name is None else self.dst_name
                )
            except TypeError as exc:
                # Raise an error if 'dst_name' input is missing
                raise ValueError("missing the input of the 'dst_name'") from exc

            # Set the destination directory if not provided
            self.dst_dir = os.path.dirname(self.src_path) if self.dst_dir is None else self.dst_dir
            if self.dst_name is None and self.dst_dir is None:
                # Raise an error if destination path input is invalid
                raise ValueError("error input of destination path")

            # Construct the destination path
            self.dst_path = os.path.join(self.dst_dir, self.dst_name)

    @staticmethod
    def calc_gradient_color(frgb: tuple, trgb: tuple, step: int) -> list:
        """
        Calculate a list of RGB colors representing a gradient between two given colors.

        Args:
            frgb (tuple): Starting RGB color.
            trgb (tuple): Target RGB color.
            step (int): Number of steps in the gradient.

        Returns:
            list: A list of RGB tuples representing the gradient colors.
        """
        # Generate a list of RGB colors for the gradient
        colors_rgb = [
            (
                int(frgb[0] + (trgb[0] - frgb[0]) / step * i),
                int(frgb[1] + (trgb[1] - frgb[1]) / step * i),
                int(frgb[2] + (trgb[2] - frgb[2]) / step * i),
            )
            for i in range(step + 1)
        ]
        return colors_rgb

    def table2point(
        self,
        data: dict,
        longitude: Optional[list] = None,
        latitude: Optional[list] = None,
        EPSG: Optional[int] = 4326,
    ) -> None:
        """
        Convert tabular data with longitude and latitude information into a point shapefile.

        Args:
            data (dict): Tabular data containing longitude and latitude information.
            longitude (Optional[list]): List of longitude values. Defaults to data["Longitude"].
            latitude (Optional[list]): List of latitude values. Defaults to data["Latitude"].
            EPSG (Optional[int]): EPSG code for the coordinate reference system. Defaults to 4326.
        """
        if longitude is None:
            # Use the 'Longitude' key from the data if longitude is not provided
            longitude = data["Longitude"]
        if latitude is None:
            # Use the 'Latitude' key from the data if latitude is not provided
            latitude = data["Latitude"]

        # Create a GeoDataFrame from the tabular data
        gdf = gpd.GeoDataFrame(
            pd.DataFrame(data),
            geometry=gpd.points_from_xy(longitude, latitude),
        )
        # Set the coordinate reference system of the GeoDataFrame
        gdf.crs = f"EPSG:{EPSG}"
        # gdf.to_crs(crs="EPSG:4326")

        # Save the GeoDataFrame as a shapefile
        gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

    def interpolate(
        self,
        zfield: str,
        algorithm: Optional[str] = "nearest:radius=20",
        scale: Optional[float] = None,
        bounds: Optional[str] = None,
    ) -> None:
        """
        Perform spatial interpolation on a dataset and save the result as a raster file.

        Args:
            zfield (str): Name of the field containing the values to interpolate.
            algorithm (Optional[str]): Interpolation algorithm. Defaults to "nearest:radius=20".
            scale (Optional[float]): Pixel size for the output raster. Defaults to None.
            bounds (Optional[str]): Path to a vector file defining the output bounds. Defaults to None.
        """
        # Define the GDAL Grid options
        options = {
            "format": "GTiff",
            "outputType": gdal.GDT_Float32,
            # 'cols': cols,
            # 'rows': rows,
            # 'algorithm': "linear:power=2:smothing=0.0:radius=1.0:max_points=10:min_points=0:nodata=0.0",
            # 'algorithm': "linear:radius=70",
            "algorithm": algorithm,
            "zfield": zfield,
        }

        if scale is not None:
            # Open the source dataset
            src_ds = ogr.Open(self.src_path)
            # Get the extent of the source dataset
            xmin, xmax, ymin, ymax = src_ds.GetLayer().GetExtent()

            # Calculate the number of columns based on the scale
            xnum = distance((ymax, xmin), (ymax, xmax)).meters / scale
            cols = int(xnum + 1) if int(xnum) < xnum else int(xnum)
            # Calculate the new maximum x-coordinate
            xmax = list(distance(kilometers=scale * cols / 1000).destination([ymax, xmin], 90))[1]

            # Calculate the number of rows based on the scale
            ynum = distance((ymax, xmin), (ymin, xmin)).meters / scale
            rows = int(ynum) + 1 if int(ynum) < ynum else int(ynum)
            # Calculate the new minimum y-coordinate
            ymin = list(distance(kilometers=scale * rows / 1000).destination([ymax, xmin], 180))[0]
            # Update the GDAL Grid options with the new dimensions and bounds
            options.update(
                {
                    "width": cols,
                    "height": rows,
                    "outputBounds": [xmin, ymin, xmax, ymax],
                }
            )

        if bounds is not None:
            # Open the bounds dataset
            bounds_ds = ogr.Open(bounds)
            # Get the extent of the bounds dataset
            xmin, xmax, ymin, ymax = bounds_ds.GetLayer().GetExtent()
            # Update the GDAL Grid options with the new bounds
            options.update(
                {
                    "outputBounds": [xmin, ymin, xmax, ymax],
                }
            )

        if self.tmpl_path is not None:
            # Open the template dataset
            tmpl_ds = gdal.Open(self.tmpl_path, gdal.gdalconst.GA_ReadOnly)
            # Get the width and height of the template dataset
            tmpl_x = tmpl_ds.RasterXSize
            tmpl_y = tmpl_ds.RasterYSize

            # Update the GDAL Grid options with the template dimensions
            options.update({"width": tmpl_x, "height": tmpl_y})

        # Perform the spatial interpolation using GDAL Grid
        gdal.Grid(
            destName=self.dst_path,
            srcDS=self.src_path,
            options=gdal.GridOptions(**options),
        )

    def classify(self, thresholds: Dict[int, list]):
        """
        Reclassify a single-band raster based on the provided thresholds.

        Args:
            thresholds (Dict[int, list]): A dictionary mapping category numbers to threshold ranges.
        """
        # Open the source raster dataset in read-only mode
        src_ds = gdal.Open(self.src_path, gdal.GA_ReadOnly)
        if src_ds.RasterCount > 1:
            # Raise an error if the raster has more than one band
            raise ValueError("Only single-band data is supported")

        # Get the first band of the source raster
        src_band = src_ds.GetRasterBand(1)
        # Read the raster data as a numpy array
        src_array = src_band.ReadAsArray()
        # 3. 进行重分类：根据你的需求设定不同类别的阈值，并将栅格值映射到新的类别上。

        # Create a new raster array initialized with -1
        dst_array = np.full(src_array.shape, -1)

        # Iterate over each category and its threshold
        for category, threshold in thresholds.items():
            if threshold[0] in [min, 'min']:
                # Set the minimum threshold to the minimum value of the raster
                threshold[0] = src_array.min()
            if threshold[1] in [max, "max"]:
                # Set the maximum threshold to the maximum value of the raster
                threshold[1] = src_array.max()
            # Create a mask for pixels within the threshold range
            mask = np.logical_and(src_array >= threshold[0], src_array <= threshold[1])
            # Reclassify the pixels within the mask to the corresponding category
            dst_array[mask] = category

        # 4. 创建输出栅格文件，并将重分类后的数组写入其中：
        # Get the GeoTIFF driver
        driver = gdal.GetDriverByName("GTiff")
        # Create the destination raster dataset
        dst_ds = driver.Create(
            self.dst_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Int16
        )
        # Set the geotransform of the destination raster
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        # Set the projection of the destination raster
        dst_ds.SetProjection(src_ds.GetProjection())

        # Get the first band of the destination raster
        dst_band = dst_ds.GetRasterBand(1)
        # Write the reclassified array to the destination band
        dst_band.WriteArray(dst_array)
        # Flush the cache to ensure data is written to disk
        dst_band.FlushCache()
        # 5. 最后，关闭输入栅格文件和清理资源：
        del dst_ds
        src_band = None
        src_ds = None

    def colorize(
        self,
        ColorEntry: Optional[Dict[int, Union[tuple, list]]] = None,
        ColorRamps: Optional[List[Dict]] = None,
    ):
        """
        Apply a color table to a single-band raster.

        Args:
            ColorEntry (Optional[Dict[int, Union[tuple, list]]]): A dictionary mapping pixel values to RGB colors.
            ColorRamps (Optional[List[Dict]]): A list of dictionaries defining color ramps.
        """
        if ColorEntry is None and ColorRamps is None:
            # Raise an error if no color inputs are provided
            raise ValueError("Color inputs is missing")
        if ColorEntry is not None and ColorRamps is not None:
            # If both color entry and color ramps are provided, ignore color entry
            ColorEntry = None
        if not self.src_path.endswith(('.tif', 'tiff')):
            # Raise an error if the input file is not a GeoTIFF
            raise ValueError("input error!")
        # Open the source raster dataset in update mode
        src_ds = gdal.Open(self.src_path, gdal.GA_Update)
        if src_ds.RasterCount > 1:
            # Raise an error if the raster has more than one band
            raise ValueError("Only single-band data is supported")
        # Read the raster data as a numpy array and convert it to uint16
        array = src_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)
        # Get the GeoTIFF driver
        driver = gdal.GetDriverByName('gtiff')
        # Create the destination raster dataset
        dst_ds = driver.Create(
            self.dst_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_UInt16
        )
        # Set the geotransform of the destination raster
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        # Set the projection of the destination raster
        dst_ds.SetProjection(src_ds.GetProjection())
        # Get the first band of the destination raster
        dst_band = dst_ds.GetRasterBand(1)
        # Write the raster data to the destination band
        dst_band.WriteArray(array)
        # Create a new color table
        color_table = gdal.ColorTable()
        if ColorRamps is not None and ColorEntry is None:
            # Create color ramps based on the provided ColorRamps
            for ColorRamp in ColorRamps:
                color_table.CreateColorRamp(
                    list(ColorRamp.keys())[0],
                    list(ColorRamp.values())[0],
                    list(ColorRamp.keys())[1],
                    list(ColorRamp.values())[1],
                )
        if ColorRamps is None and ColorEntry is not None:
            # Set color entries based on the provided ColorEntry
            for category, threshold in ColorEntry.items():
                color_table.SetColorEntry(category, threshold)
        # Set the color table of the destination band
        dst_band.SetRasterColorTable(color_table)
        # Set the color interpretation of the destination band
        dst_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        # Flush the cache to ensure data is written to disk
        dst_ds.FlushCache()

    def shp2tif(
        self,
        field: Optional[str] = None,
        bands: Optional[int] = 1,
        burn_values: Optional[int] = 0,
        all_touch: Optional[bool] = False,
        NoData: Optional[int] = None,
    ) -> None:
        """
        Convert a shapefile to a raster file.

        Args:
            field (Optional[str]): Name of the field to use for rasterization. Defaults to None.
            bands (Optional[int]): Number of bands in the output raster. Defaults to 1.
            burn_values (Optional[int]): Value to burn into the raster. Defaults to 0.
            all_touch (Optional[bool]): Whether to rasterize all pixels touched by a feature. Defaults to False.
            NoData (Optional[int]): NoData value for the output raster. Defaults to None.
        """
        # Open the source shapefile
        src_ds = ogr.Open(self.src_path)
        # Get the layer from the source shapefile
        src_layer = src_ds.GetLayer()
        # featureCount = layer.GetFeatureCount()

        # Open the template raster dataset in read-only mode
        tmpl_ds = gdal.Open(self.tmpl_path, gdal.gdalconst.GA_ReadOnly)
        # Get the width of the template raster
        tmpl_x = tmpl_ds.RasterXSize
        # Get the height of the template raster
        tmpl_y = tmpl_ds.RasterYSize
        # Get the first band of the template raster
        tmpl_band = tmpl_ds.GetRasterBand(1)

        if field is not None:
            # Generate the destination path based on the field name
            self.dst_path = f"{self.src_path[:-4]}_{field}.tif"
            # Create the destination raster dataset
            dst_ds = gdal.GetDriverByName("GTiff").Create(
                self.dst_path, tmpl_x, tmpl_y, 1, tmpl_band.DataType
            )
            # Set the geotransform of the destination raster
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
            # Set the projection of the destination raster
            dst_ds.SetProjection(src_ds.GetProjection())
            # Get the first band of the destination raster
            dst_band = dst_ds.GetRasterBand(1)

            if tmpl_band.GetNoDataValue() is not None:
                # Set the NoData value of the destination band to the template's NoData value
                dst_band.SetNoDataValue(tmpl_band.GetNoDataValue())
            if NoData is not None:
                # Set the NoData value of the destination band to the provided NoData value
                dst_band.SetNoDataValue(NoData)
            # Flush the cache to ensure data is written to disk
            dst_band.FlushCache()

            # Rasterize the shapefile layer
            gdal.RasterizeLayer(
                dst_ds,
                [bands],
                src_layer,
                burn_values=[burn_values],
                options=[
                    "ALL_TOUCHED=" + str(all_touch),
                    "ATTRIBUTE=" + field,
                ],
            )
        else:
            # Create the destination raster dataset
            dst_ds = gdal.GetDriverByName("GTiff").Create(
                self.dst_path, tmpl_x, tmpl_y, 1, tmpl_band.DataType
            )
            # Set the geotransform of the destination raster
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
            # Set the projection of the destination raster
            dst_ds.SetProjection(src_ds.GetProjection())
            # Get the first band of the destination raster
            dst_band = dst_ds.GetRasterBand(1)

            if tmpl_band.GetNoDataValue() is not None:
                # Set the NoData value of the destination band to the template's NoData value
                dst_band.SetNoDataValue(tmpl_band.GetNoDataValue())
            if NoData is not None:
                # Set the NoData value of the destination band to the provided NoData value
                dst_band.SetNoDataValue(NoData)

            # Flush the cache to ensure data is written to disk
            dst_band.FlushCache()
            # Rasterize the shapefile layer
            gdal.RasterizeLayer(
                dst_ds,
                [bands],
                src_layer,
                burn_values=[burn_values],
                options=["ALL_TOUCHED=" + str(all_touch)],
            )

    def resample(
        self,
        pSize: Optional[Union[float, int]] = None,
        scale: Optional[Union[float, int]] = None,
        EPSG: Optional[int] = None,
        NoData: Optional[int] = None,
    ) -> None:
        """
        Resample a raster dataset.

        Args:
            pSize (Optional[Union[float, int]]): Pixel size for the output raster. Defaults to None.
            scale (Optional[Union[float, int]]): Scaling factor for the pixel size. Defaults to None.
            EPSG (Optional[int]): EPSG code for the output coordinate reference system. Defaults to None.
            NoData (Optional[int]): NoData value for the output raster. Defaults to None.
        """
        if pSize is None and scale is None and self.tmpl_path is None:
            # Raise an error if no valid input is provided
            raise ValueError("error input")

        # Open the source raster dataset in read-only mode
        src_ds = gdal.Open(self.src_path, gdal.GA_ReadOnly)
        # Get the projection of the source raster
        src_proj = src_ds.GetProjection()
        # Get the geotransform of the source raster
        src_trans = src_ds.GetGeoTransform()
        # Get the width of the source raster
        src_x = src_ds.RasterXSize
        # Get the height of the source raster
        src_y = src_ds.RasterYSize
        # Get the number of bands in the source raster
        src_bands = src_ds.RasterCount

        if self.tmpl_path is not None:
            # Open the template raster dataset in read-only mode
            tmpl_ds = gdal.Open(self.tmpl_path, gdal.GA_ReadOnly)
            # Get the projection of the template raster
            tmpl_proj = tmpl_ds.GetProjection()
            # Get the geotransform of the template raster
            tmpl_trans = tmpl_ds.GetGeoTransform()
            # Get the width of the template raster
            tmpl_x = tmpl_ds.RasterXSize
            # Get the height of the template raster
            tmpl_y = tmpl_ds.RasterYSize

            # Set the output dimensions and projection based on the template
            dst_cols, dst_rows = [tmpl_x, tmpl_y]
            dst_proj = tmpl_proj
            dst_trans = tmpl_trans

        else:
            # Get the EPSG code of the source raster
            src_epsg = osr.SpatialReference(wkt=src_proj).GetAttrValue("AUTHORITY", 1)
            # Create a spatial reference object for the source raster
            src_sr = osr.SpatialReference()
            src_sr.ImportFromEPSG(int(src_epsg))
            # Define the UK OSNG, see <http://spatialreference.org/ref/epsg/27700/>

            # Set the output EPSG code to the source EPSG code
            dst_epsg = int(src_epsg)
            # Create a spatial reference object for the output raster
            dst_sr = osr.SpatialReference()
            dst_sr.ImportFromEPSG(dst_epsg)

            # Create a coordinate transformation object
            tx = osr.CoordinateTransformation(dst_sr, src_sr)
            # Up to here, all  the projection have been defined, as well as a transformation from the from to the  to :)

            # Calculate the boundaries of the new dataset in the target projection
            (ulx, uly, ulz) = tx.TransformPoint(src_trans[0], src_trans[3])
            (lrx, lry, lrz) = tx.TransformPoint(
                src_trans[0] + src_trans[1] * src_x, src_trans[3] + src_trans[5] * src_y
            )

            # Calculate the pixel spacing
            pixel_spacing = src_trans[1] / scale if scale is not None else pSize
            # Calculate the new geotransform
            dst_trans = (
                ulx,
                pixel_spacing,
                src_trans[2],
                uly,
                src_trans[4],
                -pixel_spacing,
            )

            # Calculate the number of columns and rows for the output raster
            dst_cols = int((lrx - ulx) / pixel_spacing)
            dst_rows = int((uly - lry) / pixel_spacing)

            # Get the output projection as a WKT string
            dst_proj = dst_sr.ExportToWkt()

        # Get the GeoTIFF driver
        driver = gdal.GetDriverByName("GTiff")

        # Create the destination raster dataset
        dst_ds = driver.Create(
            self.dst_path,
            dst_cols,
            dst_rows,
            src_bands,
            src_ds.GetRasterBand(1).DataType,
        )
        # Set the geotransform of the destination raster
        dst_ds.SetGeoTransform(dst_trans)
        # Set the projection of the destination raster
        dst_ds.SetProjection(dst_proj)

        # Set the NoData value for each band
        for b in range(src_bands):
            src_band = src_ds.GetRasterBand(b + 1)
            dst_band = dst_ds.GetRasterBand(b + 1)
            if src_band.GetNoDataValue() is not None:
                # Set the NoData value of the destination band to the source's NoData value
                dst_band.SetNoDataValue(src_band.GetNoDataValue())
            if NoData is not None:
                # Set the NoData value of the destination band to the provided NoData value
                dst_band.SetNoDataValue(NoData)
        # Define the GDAL Warp options
        options = gdal.WarpOptions(
            srcSRS=src_proj,
            dstSRS=dst_proj,
            resampleAlg=gdal.GRA_Bilinear,
        )
        # Perform the resampling using GDAL Warp
        gdal.Warp(
            destNameOrDestDS=dst_ds,
            srcDSOrSrcDSTab=src_ds,
            options=options,
        )

        if EPSG is not None:
            # Create a spatial reference object for the new EPSG code
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(EPSG)

            # Define the GDAL Warp options for the reprojection
            opts = gdal.WarpOptions(
                dstSRS=dst_srs.ExportToWkt(),
                resampleAlg=gdal.GRA_NearestNeighbour,
            )
            # Perform the reprojection using GDAL Warp
            gdal.Warp(
                destNameOrDestDS=self.dst_path,
                srcDSOrSrcDSTab=self.dst_path,
                options=opts,
            )

    def reproject(
        self,
        EPSG: int = 4326,
    ) -> None:
        """
        Reproject a raster or vector dataset to a new coordinate reference system.

        Args:
            EPSG (int): EPSG code for the target coordinate reference system. Defaults to 4326.
        """

        if os.path.basename(self.src_path).endswith(".tif") or os.path.basename(
            self.src_path
        ).endswith(".shp"):
            # Create a spatial reference object for the target EPSG code
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(EPSG)

            # Define the GDAL Warp options for the reprojection
            opts = gdal.WarpOptions(
                dstSRS=dst_srs.ExportToWkt(),
                resampleAlg=gdal.GRA_NearestNeighbour,
            )
            # Perform the reprojection using GDAL Warp
            gdal.Warp(
                destNameOrDestDS=self.dst_path,
                srcDSOrSrcDSTab=self.src_path,
                options=opts,
            )

        else:  # format is geojson or other vector file
            # Read the vector dataset using geopandas
            gdf = gpd.read_file(self.src_path)
            # Reproject the GeoDataFrame to the target coordinate reference system
            gdf.to_crs(crs="EPSG:" + str(EPSG))
            if os.path.basename(self.src_path).endswith(".json") or os.path.basename(
                self.src_path
            ).endswith(".geojson"):
                # Save the GeoDataFrame as a GeoJSON file
                gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")
            else:
                # Save the GeoDataFrame as a shapefile
                gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

    def tif2polygon(
        self,
    ) -> None:
        """
        Convert a raster file into a polygon shapefile.
        """

        # Open the source raster dataset
        src_ds = gdal.Open(self.src_path)
        # Create a spatial reference object for the source raster
        src_prj = osr.SpatialReference()
        src_prj.ImportFromWkt(src_ds.GetProjection())

        # Get the ESRI Shapefile driver
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(self.dst_path):
            # Delete the existing destination file if it exists
            drv.DeleteDataSource(self.dst_path)
        # Create the destination shapefile
        dst_ds = drv.CreateDataSource(self.dst_path)

        # Define a field for the polygon attributes
        FLD = ogr.FieldDefn("value", ogr.OFTReal)
        FLD.SetWidth(20)
        FLD.SetPrecision(20)
        # Get the first band of the source raster
        src_band = src_ds.GetRasterBand(1)
        # Create a layer in the destination shapefile
        dst_layer = dst_ds.CreateLayer(
            os.path.basename(self.dst_path).split(".")[0],
            srs=src_prj,
            geom_type=ogr.wkbMultiPolygon,
        )
        # Create the field in the destination layer
        dst_layer.CreateField(FLD)
        # Get the index of the field in the destination layer
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex("value")
        # Convert the raster to polygons using GDAL Polygonize
        gdal.Polygonize(src_band, None, dst_layer, dst_field, [], callback=None)

    def tif2point(
        self,
        field_name: Optional[Union[list, str]] = None,
        EPSG: Optional[int] = None,
    ) -> None:
        """
        Convert a raster file into a point vector file.

        Args:
            field_name (Optional[Union[list, str]]): Name(s) of the field(s) to store raster values. Defaults to None.
            EPSG (Optional[int]): EPSG code for the output coordinate reference system. Defaults to None.
        """
        # Open the source raster dataset
        src_ds = gdal.Open(self.src_path)
        # Get the width of the source raster
        src_x = src_ds.RasterXSize
        # Get the height of the source raster
        src_y = src_ds.RasterYSize
        # Get the geotransform of the source raster
        src_trans = src_ds.GetGeoTransform()
        # Get the projection of the source raster
        src_proj = src_ds.GetProjection()
        # Get the number of bands in the source raster
        src_bands = src_ds.RasterCount
        # Get the EPSG code of the source raster
        src_epsg = osr.SpatialReference(wkt=src_proj).GetAttrValue("AUTHORITY", 1)

        # Initialize a dictionary to store the data
        data = {}
        # Calculate the longitude values for each pixel
        data.update(
            {
                "Longitude": (
                    np.full((src_y, src_x), src_trans[0])
                    + (np.full((src_y, src_x), 1) * np.arange(src_x) * src_trans[1])
                    + (np.full((src_y, src_x), 1) * np.transpose([np.arange(src_y)]) * src_trans[2])
                ).reshape(-1)
            }
        )

        # Calculate the latitude values for each pixel
        data.update(
            {
                "Latitude": (
                    (
                        np.full((src_y, src_x), src_trans[3])
                        + (np.full((src_y, src_x), 1) * np.arange(src_x) * src_trans[4])
                        + (
                            np.full((src_y, src_x), 1)
                            * np.transpose([np.arange(src_y)])
                            * src_trans[5]
                        )
                    ).reshape(-1)
                )
            }
        )

        if src_bands == 1:
            # Set the field name if not provided
            field_name = field_name if field_name is not None else "value"
            # Add the raster values to the data dictionary
            data.update({"value": src_ds.GetRasterBand(1).ReadAsArray().reshape(-1)})
        else:
            # Set the field names if not provided
            field_name = (
                field_name if field_name is not None else [f"value_{b}" for b in range(src_bands)]
            )
            for b in range(src_bands):
                # Add the raster values for each band to the data dictionary
                data.update(
                    {f"{field_name[b]}": src_ds.GetRasterBand(b + 1).ReadAsArray().reshape(-1)}
                )

        # Create a GeoDataFrame from the data
        gdf = gpd.GeoDataFrame(
            pd.DataFrame(data),
            geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]),
        )

        if EPSG is None:
            # Set the coordinate reference system of the GeoDataFrame to the source EPSG code
            gdf.crs = f"EPSG:{src_epsg}"
        else:
            # Reproject the GeoDataFrame to the target coordinate reference system
            gdf.to_crs(crs="EPSG:4326")

        if self.dst_path.endswith("shp"):
            # Save the GeoDataFrame as a shapefile
            gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

        if self.dst_path.endswith("json") or self.dst_path.endswith("geojson"):
            # Save the GeoDataFrame as a GeoJSON file
            gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")

    def group_tif(
        self,
        band_names: Optional[list] = None,
        NoData: Optional[int] = None,
    ) -> None:
        """
        Group multiple single-band GeoTIFF files into a multi-band GeoTIFF file.

        Args:
            band_names (Optional[list]): List of names for each band in the output raster. Defaults to None.
            NoData (Optional[int]): NoData value for the output raster. Defaults to None.
        """
        if self.src_path is not None and isinstance(self.src_path, list):
            # Open all source raster datasets if src_path is a list
            src_ds = [gdal.Open(file) for file in self.src_path]
        if self.src_dir is not None:
            # Open all source raster datasets in the source directory
            src_ds = [
                gdal.Open(os.path.join(self.src_dir, file)) for file in os.listdir(self.src_dir)
            ]

        # Get the first source raster dataset
        src_ds1 = src_ds[0]
        # Get the geotransform of the first source raster
        src_trans = src_ds1.GetGeoTransform()
        # Get the projection of the first source raster
        src_proj = src_ds1.GetProjection()
        # Get the width of the first source raster
        src_x_size = src_ds1.RasterXSize
        # Get the height of the first source raster
        src_y_size = src_ds1.RasterYSize
        # Get the data type of the first source raster
        src_data_type = src_ds1.GetRasterBand(1).DataType

        # Get the GeoTIFF driver
        driver = gdal.GetDriverByName("GTiff")

        # Create the destination multi-band raster dataset
        dst_ds = driver.Create(self.dst_path, src_x_size, src_y_size, len(src_ds), src_data_type)
        # Set the geotransform of the destination raster
        dst_ds.SetGeoTransform(src_trans)
        # Set the projection of the destination raster
        dst_ds.SetProjection(src_proj)

        # Get the number of bands in the output raster
        bands = len(src_ds)
        for b in range(bands):
            # Get the source band
            src_band = src_ds[b].GetRasterBand(1)
            # Get the destination band
            dst_band = dst_ds.GetRasterBand(b + 1)
            if src_band.GetNoDataValue() is not None:
                # Set the NoData value of the destination band to the source's NoData value
                dst_band.SetNoDataValue(src_band.GetNoDataValue())
            if NoData is not None:
                # Set the NoData value of the destination band to the provided NoData value
                dst_band.SetNoDataValue(NoData)
            if band_names:
                # Set the description of the destination band to the corresponding band name
                dst_band.SetDescription(band_names[b])

            # Write the source band data to the destination band
            dst_band.WriteArray(src_band.ReadAsArray())

        # Release the destination raster dataset
        del dst_ds

    def mosaic_tif(self) -> None:
        """
        Mosaic multiple GeoTIFF files into a single GeoTIFF file.
        """
        if self.src_path is not None and isinstance(self.src_path, list):
            # Use the provided list of source files
            src_lst = self.src_path
        if self.src_dir is not None:
            # Get all files in the source directory
            src_lst = [os.path.join(self.src_dir, file) for file in os.listdir(self.src_dir)]

        # Initialize a list to store the GeoTIFF files to mosaic
        mosaic_lst = []
        for file in src_lst:
            if file.endswith("tif"):
                # Add the GeoTIFF file to the mosaic list
                mosaic_lst.append(file)
                # Open the GeoTIFF file
                ds = gdal.Open(file)
                # Get the projection of the GeoTIFF file
                proj = ds.GetProjection()
                if proj == '' or proj is None:
                    # Raise an error if the GeoTIFF file has no projection
                    raise ValueError(f"missing projection of the image:{file}")

        # Mosaic the GeoTIFF files using GDAL Warp
        gdal.Warp(
            self.dst_path,
            mosaic_lst,
            format="GTiff",
            resampleAlg=gdal.GRA_Bilinear,
        )

    def read_tif(self) -> dict:
        """
        Read a GeoTIFF file and return its metadata and data.

        Returns:
            dict: A dictionary containing the metadata and data of the GeoTIFF file.
        """
        # Open the source raster dataset
        src_ds = gdal.Open(self.src_path)
        # Get the width of the source raster
        src_x = src_ds.RasterXSize
        # Get the height of the source raster
        src_y = src_ds.RasterYSize
        # Get the geotransform of the source raster
        src_trans = src_ds.GetGeoTransform()
        # Get the projection of the source raster
        src_proj = src_ds.GetProjection()
        # Get the number of bands in the source raster
        src_bands = src_ds.RasterCount

        # Initialize a dictionary to store the metadata
        result = {
            "cols": src_x,
            "rows": src_y,
            "trans": src_trans,
            "proj": src_proj,
        }
        if src_ds.GetRasterBand(1).GetNoDataValue() is not None:
            # Add the NoData value to the metadata if it exists
            result.update({"NoData": src_ds.GetRasterBand(1).GetNoDataValue()})
        else:
            # Add None as the NoData value if it does not exist
            result.update({"NoData": None})

        # Initialize a dictionary to store the raster data
        data = {}
        for b in range(src_bands):
            # Read the raster data for each band and add it to the data dictionary
            data.update({b + 1: src_ds.GetRasterBand(b + 1).ReadAsArray()})

        # Add the raster data to the result dictionary
        result.update({"data": data})

        return result

    def read_json(self) -> pd.DataFrame:
        """
        Read a GeoJSON file and return it as a GeoDataFrame.

        Returns:
            pd.DataFrame: A GeoDataFrame containing the data from the GeoJSON file.
        """
        return gpd.read_file(self.src_path)

    def save2tif(
        self,
        data: Union[dict, np.ndarray],
        NoData: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save data to a GeoTIFF file.

        Args:
            data (Union[dict, np.ndarray]): Data to save. Can be a dictionary or a numpy array.
            NoData (Optional[int]): NoData value for the output raster. Defaults to None.
        """
        if isinstance(data, np.ndarray) and self.tmpl_path is None:
            # Raise an error if data is a numpy array and no template path is provided
            raise ValueError("error input")

        if self.tmpl_path is not None:
            # Open the template raster dataset in read-only mode
            tmpl_ds = gdal.Open(self.tmpl_path, gdal.GA_ReadOnly)
            # Get the projection of the template raster
            tmpl_proj = tmpl_ds.GetProjection()
            # Get the geotransform of the template raster
            tmpl_trans = tmpl_ds.GetGeoTransform()
            # Get the width of the template raster
            tmpl_x = tmpl_ds.RasterXSize
            # Get the height of the template raster
            tmpl_y = tmpl_ds.RasterYSize

            # Set the output dimensions and projection based on the template
            cols, rows, proj, trans = [tmpl_x, tmpl_y, tmpl_proj, tmpl_trans]

        elif isinstance(data, dict):
            # Get the output dimensions and projection from the data dictionary
            cols = data.get("cols")
            rows = data.get("rows")
            trans = data.get("trans")
            proj = data.get("proj")
        else:
            # Get the output dimensions and projection from the keyword arguments
            cols = kwargs.get("cols")
            rows = kwargs.get("rows")
            trans = kwargs.get("trans")
            proj = kwargs.get("proj")

        # Get the GeoTIFF driver
        driver = gdal.GetDriverByName("GTiff")

        if isinstance(data, np.ndarray) or isinstance(data["data"], np.ndarray):
            # Set the number of bands to 1 if data is a numpy array
            bands = 1
        elif isinstance(data["data"], dict):
            # Set the number of bands to the number of keys in the data dictionary
            bands = len(data["data"].keys())
        else:
            # Raise an error if data is in an unsupported format
            raise ValueError("error input")

        # Create the destination raster dataset
        dst_ds = driver.Create(self.dst_path, cols, rows, bands=bands, eType=gdal.GDT_Float64)
        # Set the geotransform of the destination raster
        dst_ds.SetGeoTransform(trans)
        # Set the projection of the destination raster
        dst_ds.SetProjection(proj)

        if isinstance(data, np.ndarray):
            # Write the numpy array to the destination raster
            dst_ds.GetRasterBand(1).WriteArray(data)
            if NoData is not None:
                # Set the NoData value of the destination raster
                dst_ds.GetRasterBand(1).SetNoDataValue(NoData)
        elif isinstance(data["data"], np.ndarray):
            # Write the numpy array from the data dictionary to the destination raster
            dst_ds.GetRasterBand(1).WriteArray(data["data"])
            if NoData is not None:
                # Set the NoData value of the destination raster
                dst_ds.GetRasterBand(1).SetNoDataValue(NoData)
        else:
            for b in range(bands):
                # Write the raster data for each band from the data dictionary to the destination raster
                dst_ds.GetRasterBand(b + 1).WriteArray(data["data"].get(b + 1))
                if data.get("NoData") is not None:
                    # Set the NoData value of the destination raster
                    dst_ds.GetRasterBand(b + 1).SetNoDataValue(data["NoData"])
                if NoData is not None:
                    # Set the NoData value of the destination raster
                    dst_ds.GetRasterBand(b + 1).SetNoDataValue(NoData)

        # Flush the cache to ensure data is written to disk
        dst_ds.FlushCache()
        # Release the destination raster dataset
        del dst_ds

    def shp2geojson(
        self,
    ) -> None:
        """
        Convert a shapefile to a GeoJSON file.
        """
        # Read the shapefile using geopandas
        gdf = gpd.read_file(self.src_path)
        # Save the GeoDataFrame as a GeoJSON file
        gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")

    def merge_geojson(
        self,
    ) -> None:
        """
        Merge multiple GeoJSON files into a single GeoJSON file.
        """
        # Initialize a dictionary to store the merged data
        data = {}
        # Get the local variables
        names = locals()
        # Get the number of source files
        files = len(self.src_path)
        for f in range(files):
            # Load the GeoJSON file
            names["gdf" + str(f)] = json.load(open(self.src_path[f], encoding="gb18030"))
            if f == 0:
                # Initialize the data dictionary with the first GeoJSON file
                data.update(names["gdf" + str(f)])
            else:
                # Get the number of features in the data dictionary
                features = len(data["features"])
                for i in range(features):
                    # Merge the properties of each feature
                    data["features"][i]["properties"].update(
                        names["gdf" + str(f)]["features"][i]["properties"]
                    )
        # Create a GeoDataFrame from the merged data
        gdf = gpd.read_file(json.dumps(data))
        # Save the GeoDataFrame as a GeoJSON file
        gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")

    def geojson2shp(
        self,
        Merge: Optional[bool] = False,
    ) -> None:
        """
        Convert a GeoJSON file or merge multiple GeoJSON files into a shapefile.

        Args:
            Merge (Optional[bool]): Whether to merge multiple GeoJSON files. Defaults to False.
        """

        if Merge is False:
            # Load the GeoJSON file
            data = json.load(open(self.src_path, encoding="utf-8"))
            # Create a GeoDataFrame from the GeoJSON data
            gdf = gpd.read_file(json.dumps(data))
            # Save the GeoDataFrame as a shapefile
            gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

        else:
            # Initialize a dictionary to store the merged data
            data = {}
            # Get the local variables
            names = locals()
            # Get the number of source files
            files = len(self.src_path)
            for f in range(files):
                # Load the GeoJSON file
                names["gdf" + str(f)] = json.load(open(self.src_path[f], encoding="gb18030"))
                if f == 0:
                    # Initialize the data dictionary with the first GeoJSON file
                    data.update(names["gdf" + str(f)])
                else:
                    # Get the number of features in the data dictionary
                    features = len(data["features"])
                    for i in range(features):
                        # Merge the properties of each feature
                        data["features"][i]["properties"].update(
                            names["gdf" + str(f)]["features"][i]["properties"]
                        )

            # Create a GeoDataFrame from the merged data
            gdf = gpd.read_file(json.dumps(data))
            # Save the GeoDataFrame as a shapefile
            gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

    def extract(
        self,
        LOC: Optional[str] = None,
        EPSG: Optional[int] = None,
    ) -> str:
        """
        Extract a subset of a raster dataset using a vector mask.

        Args:
            LOC (Optional[str]): SQL WHERE clause to filter features in the mask. Defaults to None.
            EPSG (Optional[int]): EPSG code for the output coordinate reference system. Defaults to None.

        Returns:
            str: Path to the output raster file.
        """
        # Open the source raster dataset
        src_ds = gdal.Open(self.src_path)
        # Get the data type of the first band of the source raster
        data_type = gdal.GetDataTypeName(src_ds.GetRasterBand(1).DataType)
        # Get the NoData value of the first band of the source raster
        src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
        # Define the target NoData value
        target_nodata = -3.4028234663852886e38
        # Define the tolerance for floating-point comparison
        tolerance = 1e-6
        if data_type == "Byte" and math.isclose(src_nodata, target_nodata, rel_tol=tolerance):
            # Set the NoData value to 99 if the data type is Byte and the source NoData value is close to the target
            NoData_value = 99
        else:
            # Set the NoData value to the source NoData value
            NoData_value = src_nodata
        # Define the GDAL Warp options
        options = {
            # 'warpMemoryLimit':500,
            "format": "GTiff",
            "cutlineDSName": self.mask_path,  # vector file type:shp,jeojson
            "cropToCutline": True,
            "copyMetadata": True,
            "dstNodata": NoData_value,
            # 'resampleAlg': 'bilinear',
        }

        if LOC is not None:
            # Add the SQL WHERE clause to the GDAL Warp options
            options.update({"cutlineWhere": LOC})

        if EPSG is not None:
            # Add the target EPSG code to the GDAL Warp options
            options.update({"dstSRS": f"EPSG:{EPSG}"})

        # Create a GDAL Warp options object
        opts = gdal.WarpOptions(**options)
        # Perform the raster extraction using GDAL Warp
        gdal.Warp(destNameOrDestDS=self.dst_path, srcDSOrSrcDSTab=self.src_path, options=opts)
