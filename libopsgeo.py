"""
Autor: HuPengcheng hpc0813@outlook.com
Date: 2021-04-17 11:13:56
LastEditTime: 2024-06-27 09:41:21
Description: 
"""

import json

# import datetime
import math
import os
import time

# from ast import literal_eval
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import mariadb
import numpy as np
import pandas as pd
from geopy.distance import distance
from osgeo import gdal, ogr, osr
from tqdm import tqdm


class OpsGeo:

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
        self.src_path = src_path
        self.src_dir = src_dir
        self.dst_path = dst_path
        self.dst_dir = dst_dir
        self.dst_name = dst_name
        self.tmpl_path = tmpl_path
        self.mask_path = mask_path
        self.__init_dst_path()

    def __init_dst_path(self):
        if self.dst_path is not None:
            self.dst_dir = os.path.dirname(self.dst_path)
            self.dst_name = os.path.basename(self.dst_path)
        else:
            try:
                self.dst_name = (
                    os.path.basename(self.src_path) if self.dst_name is None else self.dst_name
                )
            except TypeError as exc:
                raise ValueError("missing the input of the 'dst_name'") from exc

            self.dst_dir = os.path.dirname(self.src_path) if self.dst_dir is None else self.dst_dir
            if self.dst_name is None and self.dst_dir is None:
                raise ValueError("error input of destination path")

            self.dst_path = os.path.join(self.dst_dir, self.dst_name)

    @staticmethod
    def calc_gradient_color(frgb: tuple, trgb: tuple, step: int) -> list:
        """
        获取渐变色rgb
        :param frgb: 起始颜色rgb
        :param trbg: 目标颜色rgb
        :param step: 步数
        :return: 渐变色rgb
        """
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
        if longitude is None:
            longitude = data["Longitude"]
        if latitude is None:
            latitude = data["Latitude"]

        gdf = gpd.GeoDataFrame(
            pd.DataFrame(data),
            geometry=gpd.points_from_xy(longitude, latitude),
        )
        gdf.crs = f"EPSG:{EPSG}"
        # gdf.to_crs(crs="EPSG:4326")

        gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

    def interpolate(
        self,
        zfield: str,
        algorithm: Optional[str] = "nearest:radius=20",
        scale: Optional[float] = None,
        bounds: Optional[str] = None,
    ) -> None:
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
            src_ds = ogr.Open(self.src_path)
            xmin, xmax, ymin, ymax = src_ds.GetLayer().GetExtent()

            xnum = distance((ymax, xmin), (ymax, xmax)).meters / scale
            cols = int(xnum + 1) if int(xnum) < xnum else int(xnum)
            xmax = list(distance(kilometers=scale * cols / 1000).destination([ymax, xmin], 90))[1]

            ynum = distance((ymax, xmin), (ymin, xmin)).meters / scale
            rows = int(ynum) + 1 if int(ynum) < ynum else int(ynum)
            ymin = list(distance(kilometers=scale * rows / 1000).destination([ymax, xmin], 180))[0]
            options.update(
                {
                    "width": cols,
                    "height": rows,
                    "outputBounds": [xmin, ymin, xmax, ymax],
                }
            )

        if bounds is not None:
            bounds_ds = ogr.Open(bounds)
            xmin, xmax, ymin, ymax = bounds_ds.GetLayer().GetExtent()
            options.update(
                {
                    "outputBounds": [xmin, ymin, xmax, ymax],
                }
            )

        if self.tmpl_path is not None:

            tmpl_ds = gdal.Open(self.tmpl_path, gdal.gdalconst.GA_ReadOnly)
            tmpl_x = tmpl_ds.RasterXSize
            tmpl_y = tmpl_ds.RasterYSize

            options.update({"width": tmpl_x, "height": tmpl_y})

        gdal.Grid(
            destName=self.dst_path,
            srcDS=self.src_path,
            options=gdal.GridOptions(**options),
        )

    def classify(self, thresholds: Dict[int, list]):
        src_ds = gdal.Open(self.src_path, gdal.GA_ReadOnly)
        if src_ds.RasterCount > 1:
            raise ValueError("Only single-band data is supported")

        src_band = src_ds.GetRasterBand(1)
        src_array = src_band.ReadAsArray()
        # 3. 进行重分类：根据你的需求设定不同类别的阈值，并将栅格值映射到新的类别上。

        # 创建新的栅格数组，初始化为-1
        dst_array = np.full(src_array.shape, -1)

        # 遍历每个类别的阈值，并将符合条件的像素重分类到相应类别
        for category, threshold in thresholds.items():
            if threshold[0] in [min, 'min']:
                threshold[0] = src_array.min()
            if threshold[1] in [max, "max"]:
                threshold[1] = src_array.max()
            mask = np.logical_and(src_array >= threshold[0], src_array <= threshold[1])
            dst_array[mask] = category

        # 4. 创建输出栅格文件，并将重分类后的数组写入其中：
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(
            self.dst_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Int16
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        dst_band = dst_ds.GetRasterBand(1)
        dst_band.WriteArray(dst_array)
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
        if ColorEntry is None and ColorRamps is None:
            raise ValueError("Color inputs is missing")
        if ColorEntry is not None and ColorRamps is not None:
            ColorEntry = None
        if not self.src_path.endswith(('.tif', 'tiff')):
            raise ValueError("input error!")
        src_ds = gdal.Open(self.src_path, gdal.GA_Update)
        if src_ds.RasterCount > 1:
            raise ValueError("Only single-band data is supported")
        array = src_ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)
        driver = gdal.GetDriverByName('gtiff')
        dst_ds = driver.Create(
            self.dst_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_UInt16
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_band = dst_ds.GetRasterBand(1)
        dst_band.WriteArray(array)
        color_table = gdal.ColorTable()
        if ColorRamps is not None and ColorEntry is None:
            for ColorRamp in ColorRamps:
                color_table.CreateColorRamp(
                    list(ColorRamp.keys())[0],
                    list(ColorRamp.values())[0],
                    list(ColorRamp.keys())[1],
                    list(ColorRamp.values())[1],
                )
        if ColorRamps is None and ColorEntry is not None:
            for category, threshold in ColorEntry.items():
                color_table.SetColorEntry(category, threshold)
        dst_band.SetRasterColorTable(color_table)
        dst_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        dst_ds.FlushCache()

    def shp2tif(
        self,
        field: Optional[str] = None,
        bands=Optional[1],
        burn_values=Optional[0],
        all_touch: Optional[bool] = False,
        NoData: Optional[int] = None,
    ) -> None:
        src_ds = ogr.Open(self.src_path)
        src_layer = src_ds.GetLayer()
        # featureCount = layer.GetFeatureCount()

        tmpl_ds = gdal.Open(self.tmpl_path, gdal.gdalconst.GA_ReadOnly)
        tmpl_x = tmpl_ds.RasterXSize
        tmpl_y = tmpl_ds.RasterYSize
        tmpl_band = tmpl_ds.GetRasterBand(1)

        if field is not None:
            self.dst_path = f"{self.src_path[:-4]}_{field}.tif"
            dst_ds = gdal.GetDriverByName("GTiff").Create(
                self.dst_path, tmpl_x, tmpl_y, 1, tmpl_band.DataType
            )
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
            dst_ds.SetProjection(src_ds.GetProjection())
            dst_band = dst_ds.GetRasterBand(1)

            if tmpl_band.GetNoDataValue() is not None:
                dst_band.SetNoDataValue(tmpl_band.GetNoDataValue())
            if NoData is not None:
                dst_band.SetNoDataValue(NoData)
            dst_band.FlushCache()

            gdal.RasterizeLayer(
                dst_ds,
                bands,
                src_layer,
                burn_values=burn_values,
                options=[
                    "ALL_TOUCHED=" + str(all_touch),
                    "ATTRIBUTE=" + field,
                ],
            )
        else:
            dst_ds = gdal.GetDriverByName("GTiff").Create(
                self.dst_path, tmpl_x, tmpl_y, 1, tmpl_band.DataType
            )
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
            dst_ds.SetProjection(src_ds.GetProjection())
            dst_band = dst_ds.GetRasterBand(1)

            if tmpl_band.GetNoDataValue() is not None:
                dst_band.SetNoDataValue(tmpl_band.GetNoDataValue())
            if NoData is not None:
                dst_band.SetNoDataValue(NoData)

            dst_band.FlushCache()
            gdal.RasterizeLayer(
                dst_ds,
                bands,
                src_layer,
                burn_values=burn_values,
                options=["ALL_TOUCHED=" + str(all_touch)],
            )

    def resample(
        self,
        pSize: Optional[Union[float, int]] = None,
        scale: Optional[Union[float, int]] = None,
        EPSG: Optional[int] = None,
        NoData: Optional[int] = None,
    ) -> None:
        if pSize is None and scale is None and self.tmpl_path is None:
            raise ValueError("error input")

        src_ds = gdal.Open(self.src_path, gdal.GA_ReadOnly)
        src_proj = src_ds.GetProjection()
        src_trans = src_ds.GetGeoTransform()
        src_x = src_ds.RasterXSize
        src_y = src_ds.RasterYSize
        src_bands = src_ds.RasterCount

        if self.tmpl_path is not None:
            # get template raster information
            tmpl_ds = gdal.Open(self.tmpl_path, gdal.GA_ReadOnly)
            tmpl_proj = tmpl_ds.GetProjection()
            tmpl_trans = tmpl_ds.GetGeoTransform()
            tmpl_x = tmpl_ds.RasterXSize
            tmpl_y = tmpl_ds.RasterYSize

            dst_cols, dst_rows = [tmpl_x, tmpl_y]
            dst_proj = tmpl_proj
            dst_trans = tmpl_trans

        else:
            src_epsg = osr.SpatialReference(wkt=src_proj).GetAttrValue("AUTHORITY", 1)
            src_sr = osr.SpatialReference()
            src_sr.ImportFromEPSG(int(src_epsg))
            # Define the UK OSNG, see <http://spatialreference.org/ref/epsg/27700/>

            dst_epsg = int(src_epsg)
            dst_sr = osr.SpatialReference()
            dst_sr.ImportFromEPSG(dst_epsg)

            tx = osr.CoordinateTransformation(dst_sr, src_sr)
            # Up to here, all  the projection have been defined, as well as a transformation from the from to the  to :)

            # Work out the boundaries of the new dataset in the target projection
            (ulx, uly, ulz) = tx.TransformPoint(src_trans[0], src_trans[3])
            (lrx, lry, lrz) = tx.TransformPoint(
                src_trans[0] + src_trans[1] * src_x, src_trans[3] + src_trans[5] * src_y
            )

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

            dst_cols = int((lrx - ulx) / pixel_spacing)
            dst_rows = int((uly - lry) / pixel_spacing)

            dst_proj = dst_sr.ExportToWkt()

        # create destination raster
        driver = gdal.GetDriverByName("GTiff")

        dst_ds = driver.Create(
            self.dst_path,
            dst_cols,
            dst_rows,
            src_bands,
            src_ds.GetRasterBand(1).DataType,
        )
        dst_ds.SetGeoTransform(dst_trans)
        dst_ds.SetProjection(dst_proj)

        # set nodata_value
        for b in range(src_bands):
            src_band = src_ds.GetRasterBand(b + 1)
            dst_band = dst_ds.GetRasterBand(b + 1)
            if src_band.GetNoDataValue() is not None:
                dst_band.SetNoDataValue(src_band.GetNoDataValue())
            if NoData is not None:
                dst_band.SetNoDataValue(NoData)
        options = gdal.WarpOptions(
            srcSRS=src_proj,
            dstSRS=dst_proj,
            resampleAlg=gdal.GRA_Bilinear,
        )
        gdal.Warp(
            destNameOrDestDS=dst_ds,
            srcDSOrSrcDSTab=src_ds,
            options=options,
        )

        if EPSG is not None:
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(EPSG)

            opts = gdal.WarpOptions(
                dstSRS=dst_srs.ExportToWkt(),
                resampleAlg=gdal.GRA_NearestNeighbour,
            )
            gdal.Warp(
                destNameOrDestDS=self.dst_path,
                srcDSOrSrcDSTab=self.dst_path,
                options=opts,
            )

    def reproject(
        self,
        EPSG: int = 4326,
    ) -> None:
        """reproject"""

        if os.path.basename(self.src_path).endswith(".tif") or os.path.basename(
            self.src_path
        ).endswith(".shp"):
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(EPSG)

            opts = gdal.WarpOptions(
                dstSRS=dst_srs.ExportToWkt(),
                resampleAlg=gdal.GRA_NearestNeighbour,
            )
            gdal.Warp(
                destNameOrDestDS=self.dst_path,
                srcDSOrSrcDSTab=self.src_path,
                options=opts,
            )

        else:  # format is geojson or other vector file
            gdf = gpd.read_file(self.src_path)
            gdf.to_crs(crs="EPSG:" + EPSG)
            if os.path.basename(self.src_path).endswith(".json") or os.path.basename(
                self.src_path
            ).endswith(".geojson"):
                gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")
            else:
                gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

    def tif2polygon(
        self,
    ) -> None:
        """convert raster file into polygon file"""

        src_ds = gdal.Open(self.src_path)
        src_prj = osr.SpatialReference()
        src_prj.ImportFromWkt(src_ds.GetProjection())

        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(self.dst_path):
            drv.DeleteDataSource(self.dst_namedst_path)
        dst_ds = drv.CreateDataSource(self.dst_path)

        FLD = ogr.FieldDefn("value", ogr.OFTReal)
        FLD.SetWidth(20)
        FLD.SetPrecision(20)
        src_band = src_ds.GetRasterBand(1)
        dst_layer = dst_ds.CreateLayer(
            os.path.basename(self.dst_path).split(".")[0],
            srs=src_prj,
            geom_type=ogr.wkbMultiPolygon,
        )
        dst_layer.CreateField(FLD)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex("value")
        gdal.Polygonize(src_band, None, dst_layer, dst_field, [], callback=None)

    def tif2point(
        self,
        field_name: Optional[Union[list, str]] = None,
        EPSG: Optional[int] = None,
    ) -> None:
        """convert the format of the file from raster into vector"""
        src_ds = gdal.Open(self.src_path)
        src_x = src_ds.RasterXSize
        src_y = src_ds.RasterYSize
        src_trans = src_ds.GetGeoTransform()
        src_proj = src_ds.GetProjection()
        src_bands = src_ds.RasterCount
        src_epsg = osr.SpatialReference(wkt=src_proj).GetAttrValue("AUTHORITY", 1)

        data = {}
        data.update(
            {
                "Longitude": (
                    np.full((src_y, src_x), src_trans[0])
                    + (np.full((src_y, src_x), 1) * np.arange(src_x) * src_trans[1])
                    + (np.full((src_y, src_x), 1) * np.transpose([np.arange(src_y)]) * src_trans[2])
                ).reshape(-1)
            }
        )

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
            field_name = field_name if field_name is not None else "value"
            data.update({"value": src_ds.GetRasterBand(1).ReadAsArray().reshape(-1)})
        else:
            field_name = (
                field_name if field_name is not None else [f"value_{b}" for b in range(src_bands)]
            )
            for b in range(src_bands):
                data.update(
                    {f"{field_name[b]}": src_ds.GetRasterBand(b + 1).ReadAsArray().reshape(-1)}
                )

        gdf = gpd.GeoDataFrame(
            pd.DataFrame(data),
            geometry=gpd.points_from_xy(data["Longitude"], data["Latitude"]),
        )

        if EPSG is None:
            gdf.crs = f"EPSG:{src_epsg}"
        else:
            gdf.to_crs(crs="EPSG:4326")

        if self.dst_path.endswith("shp"):
            gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

        if self.dst_path.endswith("json") or self.dst_path.endswith("geojson"):
            gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")

    def group_tif(
        self,
        band_names: Optional[list] = None,
        NoData: Optional[int] = None,
    ) -> None:
        if self.src_path is not None and isinstance(self.src_path, list):
            src_ds = [gdal.Open(file) for file in self.src_dir]
        if self.src_dir is not None:
            src_ds = [
                gdal.Open(os.path.join(self.src_dir, file)) for file in os.listdir(self.src_dir)
            ]

        src_ds1 = src_ds[0]
        src_trans = src_ds1.GetGeoTransform()
        src_proj = src_ds1.GetProjection()
        src_x_size = src_ds1.RasterXSize
        src_y_size = src_ds1.RasterYSize
        src_data_type = src_ds1.GetRasterBand(1).DataType

        driver = gdal.GetDriverByName("GTiff")

        dst_ds = driver.Create(self.dst_path, src_x_size, src_y_size, len(src_ds), src_data_type)
        dst_ds.SetGeoTransform(src_trans)
        dst_ds.SetProjection(src_proj)

        bands = len(src_ds)
        for b in range(bands):
            src_band = src_ds[b].GetRasterBand(1)
            dst_band = dst_ds.GetRasterBand(b + 1)
            if src_band.GetNoDataValue() is not None:
                dst_band.SetNoDataValue(src_band.GetNoDataValue())
            if NoData is not None:
                dst_band.SetNoDataValue(NoData)
            if band_names:
                dst_band.SetDescription(band_names[b])

            dst_band.WriteArray(src_band.ReadAsArray())

        del dst_ds

    def mosaic_tif(self) -> None:
        if self.src_path is not None and isinstance(self.src_path, list):
            src_lst = self.src_path
        if self.src_dir is not None:
            src_lst = [os.path.join(self.src_dir, file) for file in os.listdir(self.src_dir)]

        mosaic_lst = []
        for file in src_lst:
            if file.endswith("tif"):
                mosaic_lst.append(file)
                ds = gdal.Open(file)
                proj = ds.GetProjection()
                if proj == '' or proj is None:
                    raise ValueError(f"missing projection of the image:{file}")

        gdal.Warp(
            self.dst_path,
            mosaic_lst,
            format="GTiff",
            resampleAlg=gdal.GRA_Bilinear,
        )

    def read_tif(self) -> dict:
        """read image"""
        src_ds = gdal.Open(self.src_path)
        src_x = src_ds.RasterXSize
        src_y = src_ds.RasterYSize
        src_trans = src_ds.GetGeoTransform()
        src_proj = src_ds.GetProjection()
        src_bands = src_ds.RasterCount

        result = {
            "cols": src_x,
            "rows": src_y,
            "trans": src_trans,
            "proj": src_proj,
        }
        if src_ds.GetRasterBand(1).GetNoDataValue() is not None:
            result.update({"NoData": src_ds.GetRasterBand(1).GetNoDataValue()})
        else:
            result.update({"NoData": None})

        data = {}
        for b in range(src_bands):
            data.update({b + 1: src_ds.GetRasterBand(b + 1).ReadAsArray()})

        result.update({"data": data})

        return result

    def read_json(self) -> pd.DataFrame:
        """read image"""
        return gpd.read_file(self.src_path)

    def save2tif(
        self,
        data: Union[dict, np.ndarray],
        NoData: Optional[int] = None,
        **kwargs,
    ) -> None:
        if isinstance(data, np.ndarray) and self.tmpl_path is None:
            raise ValueError("error input")

        if self.tmpl_path is not None:
            tmpl_ds = gdal.Open(self.tmpl_path, gdal.GA_ReadOnly)
            tmpl_proj = tmpl_ds.GetProjection()
            tmpl_trans = tmpl_ds.GetGeoTransform()
            tmpl_x = tmpl_ds.RasterXSize
            tmpl_y = tmpl_ds.RasterYSize

            cols, rows, proj, trans = [tmpl_x, tmpl_y, tmpl_proj, tmpl_trans]

        elif isinstance(data, dict):
            cols = data.get("cols")
            rows = data.get("rows")
            trans = data.get("trans")
            proj = data.get("proj")
        else:
            cols = kwargs.get("cols")
            rows = kwargs.get("rows")
            trans = kwargs.get("trans")
            proj = kwargs.get("proj")

        driver = gdal.GetDriverByName("GTiff")

        if isinstance(data, np.ndarray) or isinstance(data["data"], np.ndarray):
            bands = 1
        elif isinstance(data["data"], dict):
            bands = len(data["data"].keys())
        else:
            raise ValueError("error input")

        dst_ds = driver.Create(self.dst_path, cols, rows, bands=bands, eType=gdal.GDT_Float64)
        dst_ds.SetGeoTransform(trans)
        dst_ds.SetProjection(proj)

        if isinstance(data, np.ndarray):
            dst_ds.GetRasterBand(1).WriteArray(data)
            if NoData is not None:
                dst_ds.GetRasterBand(1).SetNoDataValue(NoData)
        elif isinstance(data["data"], np.ndarray):
            dst_ds.GetRasterBand(1).WriteArray(data["data"])
            if NoData is not None:
                dst_ds.GetRasterBand(1).SetNoDataValue(NoData)
        else:
            for b in range(bands):
                dst_ds.GetRasterBand(b + 1).WriteArray(data["data"].get(b + 1))
                if data["NoData"] is not None:
                    dst_ds.GetRasterBand(b + 1).SetNoDataValue(data["NoData"])
                if NoData is not None:
                    dst_ds.GetRasterBand(b + 1).SetNoDataValue(NoData)

        dst_ds.FlushCache()
        del dst_ds

    def shp2geojson(
        self,
    ) -> None:
        gdf = gpd.read_file(self.src_path)
        gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")

    def merge_geojson(
        self,
    ) -> None:
        """merge geojson data"""
        data = {}
        names = locals()
        files = len(self.src_path)
        for f in range(files):
            names["gdf" + str(f)] = json.load(open(self.src_path[f], encoding="gb18030"))
            if f == 0:
                data.update(names["gdf" + str(f)])
            else:
                features = len(data["features"])
                for i in range(features):
                    data["features"][i]["properties"].update(
                        names["gdf" + str(f)]["features"][i]["properties"]
                    )
        gdf = gpd.read_file(json.dumps(data))
        gdf.to_file(self.dst_path, driver="GeoJSON", encoding="utf-8")

    def geojson2shp(
        self,
        Merge: Optional[bool] = False,
    ) -> None:
        """convert the geojson file into vector file"""

        if Merge is False:
            data = json.load(open(self.src_path, encoding="utf-8"))
            gdf = gpd.read_file(json.dumps(data))
            gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

        else:
            data = {}
            names = locals()
            files = len(self.src_path)
            for f in range(files):
                names["gdf" + str(f)] = json.load(open(self.src_path[f], encoding="gb18030"))
                if f == 0:
                    data.update(names["gdf" + str(f)])
                else:
                    features = len(data["features"])
                    for i in range(features):
                        data["features"][i]["properties"].update(
                            names["gdf" + str(f)]["features"][i]["properties"]
                        )

            gdf = gpd.read_file(json.dumps(data))
            gdf.to_file(self.dst_path, driver="ESRI Shapefile", encoding="utf-8")

    def extract(
        self,
        LOC: Optional[str] = None,
        EPSG: Optional[int] = None,
    ) -> str:
        """extract raster by shapefile"""
        src_ds = gdal.Open(self.src_path)
        data_type = gdal.GetDataTypeName(src_ds.GetRasterBand(1).DataType)
        src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
        target_nodata = -3.4028234663852886e38
        tolerance = 1e-6
        if data_type == "Byte" and math.isclose(src_nodata, target_nodata, rel_tol=tolerance):
            NoData_value = 99
        else:
            NoData_value = src_nodata
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
            options.update({"cutlineWhere": LOC})

        if EPSG is not None:
            options.update({"dstSRS": f"EPSG:{EPSG}"})

        opts = gdal.WarpOptions(**options)
        gdal.Warp(destNameOrDestDS=self.dst_path, srcDSOrSrcDSTab=self.src_path, options=opts)
