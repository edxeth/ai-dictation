import { describe, expect, test } from "bun:test";

import {
  BRIDGE_STATE_REFRESH_INTERVAL_MS,
  startBridgeStateRefresh,
} from "./bridge-state-refresh";

describe("bridge state refresh", () => {
  test("periodically refreshes state when SSE updates are unavailable", () => {
    let scheduled: (() => void) | null = null;
    let refreshCount = 0;

    const timerId = startBridgeStateRefresh(
      () => {
        refreshCount += 1;
      },
      (callback, delay) => {
        expect(delay).toBe(BRIDGE_STATE_REFRESH_INTERVAL_MS);
        scheduled = callback;
        return 42;
      },
    );

    expect(timerId).toBe(42);
    expect(scheduled).not.toBeNull();
    scheduled!();
    scheduled!();
    expect(refreshCount).toBe(2);
  });
});
