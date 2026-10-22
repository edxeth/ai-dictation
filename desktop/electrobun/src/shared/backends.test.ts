import { describe, expect, test } from "bun:test";

import { BACKEND_LABELS, nextBackend, normalizeBackendName } from "./backends";

describe("backend names", () => {
  test("labels every persisted backend in switching order", () => {
    // Same keys and order as BACKEND_LABELS in src/local_ai_dictation/backend_state.py.
    expect(Object.entries(BACKEND_LABELS)).toEqual([
      ["whisper", "Whisper"],
      ["parakeet", "Parakeet Ultra"],
      ["parakeet-redux", "Parakeet Redux"],
      ["bark", "Bark 3.0"],
      ["bark-pro", "Bark 3.0 Pro"],
      ["willow", "Willow Voice"],
    ]);
  });

  test("cycles through every backend and wraps around", () => {
    const visited = ["whisper"];
    for (let step = 0; step < 6; step += 1) {
      visited.push(nextBackend(normalizeBackendName(visited.at(-1))));
    }
    expect(visited).toEqual([
      "whisper",
      "parakeet",
      "parakeet-redux",
      "bark",
      "bark-pro",
      "willow",
      "whisper",
    ]);
  });

  test.each([undefined, null, 42, "", "bark-ultra", "toString"])(
    "falls back to Whisper for %p",
    (value) => {
      expect(normalizeBackendName(value)).toBe("whisper");
    },
  );
});
