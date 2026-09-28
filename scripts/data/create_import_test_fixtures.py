"""Encode one bundled urban RGB image for the browser import regression check."""
from pathlib import Path
from PIL import Image

output = Path('outputs/runtime/import-review')
output.mkdir(exist_ok=True, parents=True)
image = Image.open('viewer/public/demo/copenhagen_rgb.jpg').convert('RGB').resize((256, 256))
for suffix, image_format in [('png','PNG'), ('jpg','JPEG'), ('webp','WEBP'), ('bmp','BMP'), ('jp2','JPEG2000'), ('tif','TIFF')]:
    image.save(output / f'urban-import.{suffix}', format=image_format)
(output / 'unsupported.txt').write_text('Not an image', encoding='utf-8')
(output / 'empty.png').write_bytes(b'')
print('Created urban image fixtures in', output)
