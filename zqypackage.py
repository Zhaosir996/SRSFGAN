from osgeo import gdal


def read_image(file_name):
    dataset = gdal.Open(file_name)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    image_proj = dataset.GetProjection()
    image_geotrans = dataset.GetGeoTransform()
    image_band = dataset.RasterCount
    image_data = dataset.ReadAsArray(0, 0, width, height)
    return image_data, image_band, image_proj, image_geotrans


def write_image(filename, proj, trans, image_data, image_band):
    datatype = gdal.GDT_Float32
    im_bands = image_band
    if im_bands > 1:
        im_width, im_height = image_data[0].shape

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(filename, im_height, im_width, im_bands, datatype)
        dataset.SetGeoTransform(trans)
        dataset.SetProjection(proj)

        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(image_data[i])
    else:
        im_width, im_height = image_data.shape

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(filename, im_height, im_width, im_bands, datatype)
        dataset.SetGeoTransform(trans)
        dataset.SetProjection(proj)
        dataset.GetRasterBand(1).WriteArray(image_data)
    del dataset
