import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { backendToggleCommand, bridgeEndpoint, readPreferredBackend } from "./backend-command";

describe("backend bridge command", () => {
  test("targets the bridge configured for the GUI", () => {
    expect(backendToggleCommand("local-ai-dictation", "http://127.0.0.1:40125")).toEqual([
      "local-ai-dictation",
      "backend",
      "toggle",
      "--restart-bridge",
      "--host",
      "127.0.0.1",
      "--port",
      "40125",
    ]);
  });

  test("uses the bridge defaults for an invalid URL", () => {
    expect(bridgeEndpoint("not a url")).toEqual({ host: "127.0.0.1", port: "8765" });
  });
});

describe("persisted backend", () => {
  test.each(["whisper", "parakeet", "parakeet-redux", "bark", "bark-pro", "willow"])(
    "reads the saved %s selection",
    (backend) => {
      const directory = mkdtempSync(join(tmpdir(), "dictation-backend-"));
      try {
        const path = join(directory, "backend.json");
        writeFileSync(path, JSON.stringify({ backend }));
        expect(readPreferredBackend(path)).toBe(backend);
      } finally {
        rmSync(directory, { recursive: true });
      }
    },
  );

  test.each(["{", "null", '{"backend":"unknown"}', '{"backend":42}'])(
    "falls back for invalid state %s",
    (contents) => {
      const directory = mkdtempSync(join(tmpdir(), "dictation-backend-"));
      try {
        const path = join(directory, "backend.json");
        writeFileSync(path, contents);
        expect(readPreferredBackend(path)).toBe("whisper");
      } finally {
        rmSync(directory, { recursive: true });
      }
    },
  );

  test("falls back when no selection was saved", () => {
    const directory = mkdtempSync(join(tmpdir(), "dictation-backend-"));
    try {
      expect(readPreferredBackend(join(directory, "missing.json"))).toBe("whisper");
    } finally {
      rmSync(directory, { recursive: true });
    }
  });
});
