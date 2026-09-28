export type EscapeAction = 'leave-explore' | 'leave-fullscreen' | 'none';

/** One physical Escape press removes one navigation layer; key repeats remove none. */
export function fullscreenEscapeAction(mode: string, fullscreen: boolean, repeat: boolean): EscapeAction {
  if (repeat) return 'none';
  if (mode === 'explore') return 'leave-explore';
  return fullscreen ? 'leave-fullscreen' : 'none';
}

export type EscapeKeyboard = { lock: (codes: string[]) => Promise<void>; unlock: () => void };
export function fullscreenKeyboard(navigator: { keyboard?: Partial<EscapeKeyboard> }): EscapeKeyboard | null {
  const keyboard = navigator.keyboard;
  return typeof keyboard?.lock === 'function' && typeof keyboard?.unlock === 'function' ? keyboard as EscapeKeyboard : null;
}
