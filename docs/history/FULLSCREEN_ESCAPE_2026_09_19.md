# Two-step fullscreen Escape

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Behavior

- Fullscreen + Explore (Fly or Walk): a short Escape press releases mouse capture and returns to Orbit without leaving fullscreen.
- The next Escape press in Orbit exits fullscreen.
- Repeated keydown events from a held key do not cascade through both app actions. The browser's long-hold Escape safety exit remains available.
- Windowed Explore still returns to Orbit with Escape; the fullscreen button still exits directly.

## Browser handling

Native fullscreen uses `navigator.keyboard.lock(['Escape'])`. Only Escape is captured; the lock is released on fullscreen exit or component cleanup. Without this browser API, or if its request is rejected, the viewer expands within the app window instead. This fallback preserves the two-step behavior but does not hide browser/desktop chrome. A ResizeObserver keeps the canvas aligned with that expanded viewport.

References: [Chrome fullscreen/Keyboard Lock guide](https://developer.chrome.com/blog/better-full-screen-mode), [requestFullscreen](https://developer.mozilla.org/en-US/docs/Web/API/Element/requestFullscreen).

## Verification

- TypeScript and focused ESLint passed.
- 68 targeted viewer tests passed, including five fullscreen routing/feature-detection tests.
- Chromium browser with real Escape key presses: native fullscreen Fly -> Orbit (still native fullscreen) -> windowed Orbit passed.
- Same native-fullscreen sequence with Walk passed; camera restoration and mouse-capture release checked.
- Browser with Keyboard Lock unavailable: full-window fallback, Fly -> Orbit -> normal layout passed.
- Browser with Keyboard Lock rejecting: fallback activated without leaving native fullscreen stuck; Escape restored normal layout.
- Browser reported no uncaught errors during these checks.

No model weights, inference, prediction values, or dataset files changed.
