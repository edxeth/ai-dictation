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
