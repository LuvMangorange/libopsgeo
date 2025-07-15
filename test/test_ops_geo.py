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

    def test_table2point(self):

        filepath = r"C:\Users\HuPengcheng\Downloads\test\2_大有公土壤信息.xlsx"
        sheetname = [
            "A区",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
        ]

        dataframe = pd.DataFrame()
        for sn in sheetname:
            temp_dataframe = OpsFile.read_table(filepath, header=1, sheetname=sn)
            dataframe = pd.concat([dataframe, temp_dataframe])

        data_dict = dataframe.dropna().to_dict(orient="list")
        data_dict.update(
            {
                "经度": list(
                    map(lambda x: OpsGeo.rad2dec(x.split(",")[0]), data_dict.get("取样点经纬度"))
                ),
                "纬度": list(
                    map(lambda x: OpsGeo.rad2dec(x.split(",")[1]), data_dict.get("取样点经纬度"))
                ),
            }
        )

        data_dict.update(
            {
                "氮": list(map(lambda x: x.split("、")[0], data_dict.get("养分含量"))),
                "磷": list(map(lambda x: x.split("、")[1], data_dict.get("养分含量"))),
                "钾": list(map(lambda x: x.split("、")[2], data_dict.get("养分含量"))),
            }
        )

        data_dict.update({"PH值": list(map(lambda x: round(float(x), 2), data_dict.get("PH值")))})
        data_dict.pop("序号")

        field_data = {
            'soilID': data_dict.get("土壤编号"),
            'customer': data_dict.get("客户名称"),
            'phone': data_dict.get("联系方式"),
            'exmDT': data_dict.get("取样日期"),
            'testDT': data_dict.get("检测日期"),
            'texture': data_dict.get("土壤质地"),
            'testID': data_dict.get("检测编号"),
            'pH': data_dict.get("PH值"),
            'EC': data_dict.get("EC值"),
            'N': data_dict.get("氮"),
            'P': data_dict.get("磷"),
            'K': data_dict.get("钾"),
            'lon': data_dict.get("经度"),
            'lat': data_dict.get("纬度"),
        }

        savePath = r"C:\Users\HuPengcheng\Downloads\test\禾兴测土数据.shp"

        OpsGeo(dst_path=savePath).table2point(
            data=field_data,
            longitude=field_data.get("lon"),
            latitude=field_data.get("lat"),
        )

    def test_interpolate(self):
        point_file = r"C:\Users\HuPengcheng\Downloads\test\禾兴测土数据.shp"
        # template_tif = "C:\\Users\\HuPengcheng\\Downloads\\禾兴测土数据\\tif1.tif"
        OpsGeo(src_path=point_file, dst_name="k.tif").interpolate(
            # algorithm='linear:radius=20',
            algorithm='invdistnn:power=2',
            # algorithm='invdist:power=3.6:smoothing=0.2:radius1=0.0:radius2=0.0:angle=0.0:max_points=0:min_points=0:nodata=0.0',
            zfield="k",
            # xy_num=[50,]#unit numbers
            scale=10,  # unit meter
            # template_path=template_tif,
        )

    def test_shp2tif(self):
        point_file = "C:\\Users\\HuPengcheng\\Downloads\\禾兴测土数据\\禾兴测土数据.shp"
        templatefile = "C:\\Users\\HuPengcheng\\Downloads\\禾兴测土数据\\禾兴测土数据_P.tif"
        OpsGeo(
            src_path=point_file,
            dst_name='禾兴测土数据.tif',
            tmpl_path=templatefile,
        ).shp2tif(
            # bands=[1],
            # burn_values=[0],
            field="pH",
            # all_touch="False",
        )

    def test_resample(self):
        # input_tif = r'G:\TEST\NDVI_2019_01_17.tif'
        input_tif = r'G:\TEST\NDVI_2023_07_08.tif'
        # template_path = "C:\\Users\\HuPengcheng\\Downloads\\禾兴测土数据\\禾兴测土数据_P.tif"
        # dst_path = r'G:\TEST\test.tif'
        OpsGeo(src_path=input_tif, dst_name="NDVI_2023_07_08_test2.tif").resample(
            pSize=1, EPSG=4326
        )
        # OpsGeo(src_path=input_tif, tmpl_path=r'G:\TEST\NDVI_2019_01_17.tif',dst_name="NDVI_2023_07_08_test.tif").resample()

    def test_reproject(self):
        src_dir = r"C:\Users\HuPengcheng\Downloads\RVI2024"
        for tif in os.listdir(src_dir):
            dst_dir = r'C:\Users\HuPengcheng\Downloads\RVI\2024'
            OpsGeo(src_path=os.path.join(src_dir, tif), dst_dir=dst_dir).reproject(EPSG=4326)

    def test_read_tif(self):
        src_path = r'G:\TEST\GNDVI_2019_01_17_hexing2.tif'
        result = OpsGeo(src_path=src_path).read_tif()
        print(result)

    def test_read_json(self):
        src_path = r'C:\Users\HuPengcheng\Documents\Code\Data\data_soil_hexing\boundary\地块列表+带标注.json'
        result = OpsGeo(src_path=src_path).read_json()
        print(result)

    def test_tif2point(self):
        tif_path = r'G:\TEST\NDVI_2023_07_08_test.tif'
        OpsGeo(src_path=tif_path, dst_name='test.shp').tif2point()

    def test_geojson2shp(self):
        json_path = r'C:\Users\HuPengcheng\Documents\Code\Data\data_soil_hexing\boundary\地块列表+带标注.json'
        OpsGeo(src_path=json_path, dst_name='boundary.shp').geojson2shp()

    def test_grouptif(self):
        # input_files = [
        #     r".\data\UAV_GROUP\DJI_20240620170615_0491_MS_NIR.TIF",
        #     r".\data\UAV_GROUP\DJI_20240620170615_0491_MS_R.TIF",
        #     r".\data\UAV_GROUP\DJI_20240620170615_0491_MS_RE.TIF",
        # ]

        OpsGeo(
            src_dir=r"C:\Users\HuPengcheng\Documents\Code\LIB\lib_opsgeo\data\UAV_GROUP",
            dst_dir=r"C:\Users\HuPengcheng\Documents\Code\LIB\lib_opsgeo\result",
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
            dst_name="GNDVI_2019_01_17_hexing2.tif",
        ).extract(
            EPSG='4326',
            # LOC=['name','东胜区'],
        )

    def test_classify(self):
        tif_path = r'C:\Users\HuPengcheng\Documents\Code\Data\data_soil_hexing\properties\P2.tif'
        dst_dir = r'C:\Users\HuPengcheng\Documents\Code\Data\data_soil_hexing\properties'
        OpsGeo(
            src_path=tif_path,
            dst_dir=dst_dir,
            dst_name="P2.tif",
        ).classify(
            thresholds={
                1: [40, max],
                2: [20, 40],
                3: [10, 20],
                4: [5, 10],
                5: [3, 5],
                6: [min, 3],
            }
        )

    def test_colorize(self):
        tif_path = r'C:\Users\HuPengcheng\Documents\Code\Data\data_soil_hexing\properties\EC2.tif'
        dst_path = r'C:\Users\HuPengcheng\Documents\Code\Data\data_soil_hexing\properties\EC3.tif'
        OpsGeo(
            src_path=tif_path,
            dst_path=dst_path,
        ).colorize(
            # ColorEntry={
            #     1: (241, 39, 17),
            #     2: (137, 255, 0),
            # },
            ColorRamps=[
                {
                    1: (241, 39, 17),
                    2: (137, 255, 0),
                }
            ],
        )

    def test_gradient_color(self):
        frgb = (241, 39, 17)
        trgb = (137, 255, 0)
        step = 5
        colors = OpsGeo.calc_gradient_color(frgb=frgb, trgb=trgb, step=step)
        print(colors)

    def teardown_class(self):
        print("\n", "---test operates geo data  end---")
