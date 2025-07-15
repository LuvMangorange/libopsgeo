'''
Autor: HuPengcheng hpc0813@outlook.com
Date: 2024-04-18 11:35:51
LastEditTime: 2024-08-30 10:58:14
Description: 
'''

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from libopsgeo import OpsGeo
from utils.libopsfile import OpsFile


class Test_GEO(object):
    def setup_class(self):
        print("\n", "---test operates geo data start---")

    def test_resample(self):
        input_tif = r'G:\TEST\NDVI_2023_07_08.tif
        OpsGeo(src_path=input_tif, dst_name="NDVI_2023_07_08_test2.tif").resample(
            pSize=1, EPSG=4326
        )

    def test_reproject(self):
        src_dir = r"C:\Users\\DownloadsRVI2024"
        for tif in os.listdir(src_dir):
            dst_dir = r'C:\Users\Downloads\RVI\2024'
            OpsGeo(src_path=os.path.join(src_dir, tif), dst_dir=dst_dir).reproject(EPSG=4326)

    def test_read_tif(self):
        src_path = r'G:\TEST\GNDVI_2019_01_17_hexing2.tif'
        result = OpsGeo(src_path=src_path).read_tif()
        print(result)

    def test_tif2point(self):
        tif_path = r'G:\TEST\NDVI_2023_07_08_test.tif'
        OpsGeo(src_path=tif_path, dst_name='test.shp').tif2point()

    def test_grouptif(self):
        OpsGeo(
            src_dir=r"C:\Users\data\UAV_GROUP",
            dst_dir=r"C:\Users\result",
            dst_name="group.tif",
        ).group_tif(band_names=["NIR", 'R', 'RE'])

    def test_mosaic_tif(self):

        OpsGeo(
            src_dir=r"F:\test\G",
            dst_dir=r"F:\test\G",
            dst_name="mosaic.tif",
        ).mosaic_tif()

    def test_extract(self):
        tif_path = r'G:\TEST\GNDVI_2019_01_17.tif'
        shp_path = r'G:\TEST\area.json'
        OpsGeo(
            src_path=tif_path,
            mask_path=shp_path,
            dst_name="GNDVI_2019_01_17.tif",
        ).extract(
            EPSG='4326',
        )

    def test_gradient_color(self):
        frgb = (241, 39, 17)
        trgb = (137, 255, 0)
        step = 5
        colors = OpsGeo.calc_gradient_color(frgb=frgb, trgb=trgb, step=step)
        print(colors)

    def teardown_class(self):
        print("\n", "---test operates geo data  end---")
