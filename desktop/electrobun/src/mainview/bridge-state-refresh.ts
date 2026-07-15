export const BRIDGE_STATE_REFRESH_INTERVAL_MS = 750;

type SetInterval = (callback: () => void, delay: number) => number;

export function startBridgeStateRefresh(
  refresh: () => void,
  setIntervalFn: SetInterval = window.setInterval.bind(window),
): number {
  return setIntervalFn(refresh, BRIDGE_STATE_REFRESH_INTERVAL_MS);
}
