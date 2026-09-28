from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image

from msr.api.app import create_app
from msr.io.raster import read_rgb_raster
from test_api import FakeRuntime


@pytest.mark.parametrize('suffix,format', [('jpg','JPEG'),('png','PNG'),('webp','WEBP'),('bmp','BMP'),('tif','TIFF'),('jp2','JPEG2000')])
def test_formats_reach_api_and_decode_rgb(tmp_path: Path, suffix: str, format: str):
    pixels = np.zeros((32,48,3), dtype=np.uint8)
    pixels[:,:24] = [50,150,60]
    pixels[:,24:] = [200,70,90]
    path = tmp_path / f'scene.{suffix}'
    Image.fromarray(pixels).save(path, format=format)
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path/'jobs'))
    result = client.post('/api/predict',files={'image':(path.name,path.read_bytes(),'application/octet-stream')})
    assert result.status_code == 200
    decoded = read_rgb_raster(runtime.calls[0]['input_path'])
    assert decoded.rgb.shape == (3,32,48)
    assert not decoded.georeferenced
    assert np.abs(decoded.rgb[:,10,10]-pixels[10,10]).max() < 6


@pytest.mark.parametrize('mode',['L','P'])
def test_grayscale_and_palette_clipboard_png_are_images_not_elevation(tmp_path: Path, mode: str):
    image = Image.new(mode,(20,20),100)
    if mode == 'P':
        palette = [0]*768
        palette[300:303] = [20,170,40]
        image.putpalette(palette)
    path = tmp_path/'clipboard.png'
    image.save(path)
    decoded = read_rgb_raster(path,max_dimension=10)
    assert decoded.rgb.shape == (3,10,10)
    assert not decoded.georeferenced
    np.testing.assert_equal(decoded.rgb[:,0,0], [100]*3 if mode=='L' else [20,170,40])


def test_elevation_tiff_does_not_silently_become_an_rgb_import(tmp_path: Path):
    path = tmp_path/'elevation.tif'
    with rasterio.open(path,'w',driver='GTiff',width=10,height=10,count=1,dtype='float32') as dst:
        dst.write(np.ones((1,10,10),dtype=np.float32)*1600)
    with pytest.raises(ValueError,match='Load existing height map'):
        read_rgb_raster(path)


def test_conditional_jp2_terrain_request_and_manual_calibration_exclusion(tmp_path: Path):
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime,results_root=tmp_path))
    response = client.post('/api/predict',files={'image':('scene.jp2',b'fixture')},data={'auto_dem_if_georeferenced':'true'})
    assert response.status_code == 200
    assert runtime.calls[0]['auto_dem_if_georeferenced'] is True
    response = client.post('/api/predict',files={'image':('scene.jp2',b'fixture'),'dem':('dem.tif',b'fixture')},data={'auto_dem_if_georeferenced':'true'})
    assert response.status_code == 400
    assert len(runtime.calls)==1
