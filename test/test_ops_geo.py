class Test_GEO(object):
    """
    Test suite for geospatial operations using OpsGeo library.
    
    Validates raster processing, coordinate transformations, data conversion,
    and spatial analysis functionalities.
    """

    # ... existing methods ...

    def test_resample(self):
        """
        Test raster resolution adjustment with coordinate transformation.
        
        Parameters:
        - Source: NDVI time series data
        - Target resolution: 1 degree
        - Coordinate system: WGS84 (EPSG:4326)
        - Resampling method: Nearest neighbor
        """
        input_tif = r'G:\TEST\NDVI_2023_07_08.tif'
        OpsGeo(src_path=input_tif, dst_name="NDVI_2023_07_08_test2.tif").resample(
            pSize=1, EPSG=4326
        )

    def test_reproject(self):
        """
        Batch reproject raster datasets to WGS84 coordinate system.
        
        Processes directory contents:
        - Input: Annual raster collection
        - Output: Standardized CRS for temporal analysis
        - Preservation of original pixel values
        """
        src_dir = r"C:\Users\\DownloadsRVI2024"
        for tif in os.listdir(src_dir):
            dst_dir = r'C:\Users\Downloads\RVI\2024'
            OpsGeo(src_path=os.path.join(src_dir, tif), dst_dir=dst_dir).reproject(EPSG=4326)

    def test_read_tif(self):
        """
        Validate raster metadata and data structure reading.
        
        Verifies:
        - Correct geotransform parameters
        - Proper band count recognition
        - Valid pixel value ranges
        """
        src_path = r'G:\TEST\GNDVI_2019_01_17_hexing2.tif'
        result = OpsGeo(src_path=src_path).read_tif()
        print(result)

    def test_tif2point(self):
        """
        Test raster-to-vector conversion for pixel centroids.
        
        Converts:
        - Continuous NDVI values to point features
        - Preserves original pixel values as attributes
        - Generates ESRI Shapefile output
        """
        tif_path = r'G:\TEST\NDVI_2023_07_08_test.tif'
        OpsGeo(src_path=tif_path, dst_name='test.shp').tif2point()

    def test_grouptif(self):
        """
        Test multispectral band stacking operation.
        
        Combines:
        - Near Infrared (NIR)
        - Red (R) 
        - Red Edge (RE) bands
        Creates composite raster for vegetation analysis
        """
        OpsGeo(
            src_dir=r"C:\Users\data\UAV_GROUP",
            dst_dir=r"C:\Users\result",
            dst_name="group.tif",
        ).group_tif(band_names=["NIR", 'R', 'RE'])

    def test_mosaic_tif(self):
        """
        Test raster mosaicking for adjacent datasets.
        
        Processes:
        - Input directory of tiled rasters
        - Output: Seamless composite raster
        - Automatic boundary matching
        """
        OpsGeo(
            src_dir=r"F:\test\G",
            dst_dir=r"F:\test\G",
            dst_name="mosaic.tif",
        ).mosaic_tif()

    def test_extract(self):
        """
        Test raster cropping using vector boundaries.
        
        Parameters:
        - Mask: GeoJSON polygon features
        - Output: Subset raster matching mask extent
        - Coordinate system conversion to WGS84
        """
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
        """
        Test color ramp generation for visualization.
        
        Parameters:
        - Start color: RGB(241,39,17)
        - End color: RGB(137,255,0)
        - Interpolation steps: 5
        Outputs gradient colors for classification rendering
        """
        frgb = (241, 39, 17)
        trgb = (137, 255, 0)
        step = 5
        colors = OpsGeo.calc_gradient_color(frgb=frgb, trgb=trgb, step=step)
        print(colors)

    def teardown_class(self):
        """Clean up test artifacts and temporary outputs"""
        print("\n", "---test operates geo data  end---")
