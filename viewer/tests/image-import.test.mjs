import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prepareImageImport, clipboardImageType, clipboardImageFile, MAX_IMAGE_BYTES } from '../app/image-import.ts';
import { sceneScaleLength } from '../app/scene-scale.ts';

test('TIFF import preserves the original bytes, filename and absent or generic MIME type', async () => {
  for (const type of ['', 'application/octet-stream']) {
    const file = new File([new Uint8Array([73,73,42,0,42,99])], 'terrain.TIFF', {type});
    assert.equal(prepareImageImport(file),file);
    assert.deepEqual(new Uint8Array(await file.arrayBuffer()),new Uint8Array([73,73,42,0,42,99]));
  }
});
test('shared validation accepts every supported extension but rejects wrong, empty and oversize inputs', () => {
  for (const ext of ['jpg','jpeg','png','tif','tiff','webp','bmp','jp2']) {
    const file = new File(['fixture'],`scene.${ext}`);
    assert.equal(prepareImageImport(file),file);
  }
  assert.throws(()=>prepareImageImport(new File(['data'],'bad.txt',{type:'image/png'})),/unsupported/);
  assert.throws(()=>prepareImageImport(new File([],'empty.png')),/empty/);
  assert.throws(()=>prepareImageImport({name:'large.tif',size:MAX_IMAGE_BYTES+1}),/512 MB/);
});
test('clipboard files receive an API-compatible name, without converting TIFF content', async () => {
  const unnamed = new File(['unchanged'],'image',{type:'image/png'});
  assert.equal(prepareImageImport(unnamed).name,'image.png');
  const chosen = clipboardImageType(['text/html','image/png','image/tiff']);
  assert.equal(chosen,'image/tiff');
  const file = clipboardImageFile(new Blob(['original geotiff'],{type:chosen}));
  assert.match(file.name,/\.tif$/); assert.equal(await file.text(),'original geotiff');
  assert.equal(clipboardImageType(['text/plain']),undefined);
});
test('scale ticks follow the sampled perspective span and never label invalid views', () => {
  for (const [span,expected] of [[127,100],[48,20],[5.5,5],[.37,.2],[.0012,.001]]) assert.equal(sceneScaleLength(span),expected);
  for (const span of [NaN,Infinity,0,-1]) assert.equal(sceneScaleLength(span),null);
});
