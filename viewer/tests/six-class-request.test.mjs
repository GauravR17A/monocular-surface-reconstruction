import test from 'node:test';
import assert from 'node:assert/strict';
import { createClassificationLoader, stateForScene } from '../app/six-class-request.ts';

const base = 'http://127.0.0.1:8000';
const source = { job_id: 'a'.repeat(32) };
const path = `/results/classification_${'b'.repeat(32)}`;
const names = ['ground', 'buildings', 'water', 'roads', 'low_vegetation', 'trees'];
function payload() {
  return { width: 7, height: 1, height_pipeline_changed: false, valid_pixels: 6, ignored_pixels: 1,
    source_job_id: source.job_id, source_demo_id: null, resampled: false,
    labels_url: `${path}/classes.bin`, raster_url: `${path}/classes.tif`, metadata_url: `${path}/metadata.json`,
    classes: names.map((name, id) => ({ name, id, pixels: 1, coverage_percent: 100 / 6, area_m2: null, color: [1,2,3] })) };
}
function mockFetch(report = payload(), labels = [0,1,2,3,4,5,255]) {
  const calls = [];
  const fetcher = async (url, options) => {
    calls.push({ url, options });
    return url.endsWith('/api/classify') ? Response.json(report) : new Response(new Uint8Array(labels));
  };
  return { fetcher, calls };
}

test('automatic request loads six categories and never calls height prediction', async () => {
  const {fetcher, calls} = mockFetch();
  const result = await createClassificationLoader(fetcher)(base, source);
  assert.deepEqual([...result.labels], [0,1,2,3,4,5,255]);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].options.body.get('job_id'), source.job_id);
  assert.equal(calls[0].options.body.has('image'), false);
  assert.ok(calls.every(({url}) => !url.includes('predict')));
});

test('concurrent mounts and revisiting a scene share a single inference', async () => {
  const {fetcher, calls} = mockFetch();
  const load = createClassificationLoader(fetcher);
  const [a,b] = await Promise.all([load(base, source), load(base, source)]);
  assert.equal(a, b);
  assert.equal(await load(base, source), a);
  assert.equal(calls.length, 2);
});

test('six-class failure is retriable without changing the original image or heights', async () => {
  const valid = mockFetch(); let fail = true;
  const load = createClassificationLoader((url, options) => fail ? Promise.resolve(Response.json({detail:'GPU temporarily unavailable'}, {status:500})) : valid.fetcher(url, options));
  await assert.rejects(load(base, source), error => error.status === 'error' && /GPU/.test(error.message));
  fail = false;
  assert.equal((await load(base, source)).classes.length, 6);
});

test('unsupported 16-bit input is explicit, not a fabricated six-class map', async () => {
  const load = createClassificationLoader(async () => Response.json({detail:'Requires 8-bit RGB'}, {status:422}));
  await assert.rejects(load(base, source), error => error.status === 'unsupported' && /8-bit/.test(error.message));
});

test('scene identity prevents late results or replaced textures leaking into a new scene', () => {
  const old = {sceneKey:'old.jpg', status:'ready', result:payload()};
  assert.equal(stateForScene(old, 'old.jpg'), old);
  assert.deepEqual(stateForScene(old, 'new.jpg'), {sceneKey:'new.jpg', status:'waiting', result:null});
  assert.equal(stateForScene(old, '').status, 'idle');
});

for (const [name, change] of [
  ['wrong source', p => {p.source_job_id = 'c'.repeat(32);}],
  ['wrong class order', p => {p.classes.reverse();}],
  ['external artifact', p => {p.labels_url = 'https://foreign/classes.bin';}],
  ['different artifact directories', p => {p.metadata_url = `/results/classification_${'c'.repeat(32)}/metadata.json`;}],
  ['changed height contract', p => {p.height_pipeline_changed = true;}],
  ['wrong coverage', p => {p.classes[1].coverage_percent = 25;}],
  ['wrong pixel count', p => {p.valid_pixels = 7;}],
  ['oversized map', p => {p.width = 50000;}],
]) test(`reject ${name}`, async () => {
  const report = payload(); change(report);
  await assert.rejects(createClassificationLoader(mockFetch(report).fetcher)(base, source));
});

test('unknown 255 stays unknown, other unsupported IDs are rejected', async () => {
  await assert.rejects(createClassificationLoader(mockFetch(payload(), [0,1,2,3,4,5,6]).fetcher)(base, source));
});
