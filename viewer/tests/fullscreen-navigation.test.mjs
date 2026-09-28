import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fullscreenEscapeAction, fullscreenKeyboard } from '../app/fullscreen-navigation.ts';

test('first Escape leaves Explore; second Escape leaves fullscreen', () => {
  assert.equal(fullscreenEscapeAction('explore',true,false),'leave-explore');
  assert.equal(fullscreenEscapeAction('orbit',true,false),'leave-fullscreen');
});
test('holding Escape does not cascade through navigation layers', () => {
  for (const mode of ['explore','orbit','inspect','sightline']) assert.equal(fullscreenEscapeAction(mode,true,true),'none');
});
test('windowed Explore exits normally and windowed Orbit has no fullscreen action', () => {
  assert.equal(fullscreenEscapeAction('explore',false,false),'leave-explore');
  assert.equal(fullscreenEscapeAction('orbit',false,false),'none');
});
test('other fullscreen modes can exit without getting trapped', () => {
  assert.equal(fullscreenEscapeAction('inspect',true,false),'leave-fullscreen');
  assert.equal(fullscreenEscapeAction('sightline',true,false),'leave-fullscreen');
});
test('keyboard capture is used only when both lock and unlock exist', () => {
  assert.equal(fullscreenKeyboard({}),null);
  assert.equal(fullscreenKeyboard({keyboard:{lock:async()=>{}}}),null);
  const keyboard={lock:async()=>{},unlock:()=>{}};
  assert.equal(fullscreenKeyboard({keyboard}),keyboard);
});
