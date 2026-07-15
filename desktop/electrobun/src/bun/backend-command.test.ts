import { describe, expect, test } from "bun:test";

import { backendToggleCommand, bridgeEndpoint } from "./backend-command";

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
