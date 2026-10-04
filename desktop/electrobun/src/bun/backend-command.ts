import { readFileSync } from "node:fs";

export type BridgeEndpoint = {
  host: string;
  port: string;
};

export function bridgeEndpoint(bridgeUrl: string): BridgeEndpoint {
  try {
    const parsed = new URL(bridgeUrl);
    return {
      host: parsed.hostname || "127.0.0.1",
      port: parsed.port || "8765",
    };
  } catch {
    return { host: "127.0.0.1", port: "8765" };
  }
}

export function backendToggleCommand(cli: string, bridgeUrl: string): string[] {
  const endpoint = bridgeEndpoint(bridgeUrl);
  return [
    cli,
    "backend",
    "toggle",
    "--restart-bridge",
    "--host",
    endpoint.host,
    "--port",
    endpoint.port,
  ];
}

/** Read the saved model, using Whisper when the state file is missing or invalid. */
export function readPreferredBackend(statePath: string): "whisper" | "parakeet" | "willow" {
  try {
    const payload: unknown = JSON.parse(readFileSync(statePath, "utf8"));
    if (
      payload !== null &&
      typeof payload === "object" &&
      "backend" in payload &&
      (payload.backend === "parakeet" || payload.backend === "willow")
    ) {
      return payload.backend;
    }
  } catch {
    return "whisper";
  }
  return "whisper";
}
